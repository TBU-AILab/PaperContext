#Looking at the current algorithm, I can identify several improvements:
#
#1. **Better CMA-ES implementation** with proper path evolution and step-size adaptation
#2. **Multi-restart strategy** with decreasing budgets for global exploration vs local refinement
#3. **Better SHADE parameters** and population size reduction (L-SHADE)
#4. **Surrogate-assisted pre-screening** to save function evaluations
#5. **Improved local search** with Nelder-Mead fallback for low dimensions
#
import numpy as np
from datetime import datetime, timedelta
import time

def run(func, dim, bounds, max_time):
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def time_left():
        return max_time * 0.98 - elapsed()
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    best = float('inf')
    best_params = None
    evals = 0
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def bounce_back(mutant, parent):
        result = mutant.copy()
        for d in range(dim):
            if result[d] < lower[d]:
                result[d] = (lower[d] + parent[d]) / 2.0
            elif result[d] > upper[d]:
                result[d] = (upper[d] + parent[d]) / 2.0
        return result

    # ==================== L-SHADE ====================
    def run_lshade(budget_fraction=0.70):
        nonlocal best, best_params
        
        N_init = min(max(12 * dim, 50), 200)
        N_min = 4
        pop_size = N_init
        max_evals_shade = None  # we use time
        
        # LHS initialization
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
        
        fitness = np.array([eval_func(population[i]) for i in range(pop_size) if time_left() > max_time * (1.0 - budget_fraction)])
        if len(fitness) < pop_size:
            population = population[:len(fitness)]
            pop_size = len(fitness)
        
        if pop_size < 4:
            return
        
        # Opposition-based initialization
        opp_pop = lower + upper - population
        opp_fitness = []
        for i in range(pop_size):
            if time_left() <= max_time * (1.0 - budget_fraction):
                break
            opp_fitness.append(eval_func(opp_pop[i]))
        
        if opp_fitness:
            n_opp = len(opp_fitness)
            all_pop = np.vstack([population, opp_pop[:n_opp]])
            all_fit = np.concatenate([fitness, np.array(opp_fitness)])
            order = np.argsort(all_fit)[:pop_size]
            population = all_pop[order]
            fitness = all_fit[order]
        
        H = 100
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.3)
        k = 0
        
        archive = []
        archive_max = pop_size
        
        generation = 0
        evals_at_start = evals
        
        time_threshold = max_time * (1.0 - budget_fraction)
        
        while time_left() > time_threshold and pop_size >= N_min:
            generation += 1
            S_CR = []
            S_F = []
            S_delta = []
            
            trial_pop = []
            trial_fit = []
            success_indices = []
            
            # Sort population
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]
            
            for i in range(pop_size):
                if time_left() <= time_threshold:
                    break
                
                ri = np.random.randint(H)
                
                if M_CR[ri] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                while F_i <= 0:
                    F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                F_i = min(F_i, 1.0)
                
                # current-to-pbest/1
                p = max(2.0 / pop_size, 0.05) + 0.15 * np.random.random()
                p = min(p, 0.25)
                n_pbest = max(1, int(p * pop_size))
                pbest_idx = np.random.randint(n_pbest)
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size - 1)
                if r2 >= i:
                    r2 += 1
                if r2 == r1:
                    r2 = (r2 + 1) % pool_size
                    if r2 == i:
                        r2 = (r2 + 1) % pool_size
                
                if r2 < pop_size:
                    xr2 = population[r2]
                else:
                    xr2 = archive[r2 - pop_size]
                
                mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - xr2)
                mutant = bounce_back(mutant, population[i])
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                trial_f = eval_func(trial)
                
                if trial_f <= fitness[i]:
                    delta = fitness[i] - trial_f
                    trial_pop.append((i, trial, trial_f))
                    if delta > 0:
                        S_CR.append(CR_i)
                        S_F.append(F_i)
                        S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    population[i] = trial
                    fitness[i] = trial_f
            
            # Update memory
            if S_CR and S_F:
                weights = np.array(S_delta)
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                    s_f = np.array(S_F)
                    s_cr = np.array(S_CR)
                    
                    mean_cr = np.sum(weights * s_cr)
                    if np.max(s_cr) == 0:
                        M_CR[k] = -1
                    else:
                        M_CR[k] = mean_cr
                    M_F[k] = np.sum(weights * s_f ** 2) / (np.sum(weights * s_f) + 1e-30)
                    k = (k + 1) % H
            
            # Linear population size reduction (L-SHADE)
            total_time = max_time * budget_fraction
            time_used = elapsed() - (max_time - max_time * budget_fraction - (max_time * (1.0 - budget_fraction) - time_threshold + max_time * budget_fraction))
            ratio = min(1.0, (elapsed() - (max_time - max_time * 0.98)) / (total_time + 1e-30))
            # Simple ratio based on generation count proxy
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * min(generation / max(1, 3 * N_init), 1.0))))
            
            if new_pop_size < pop_size:
                order = np.argsort(fitness)
                population = population[order[:new_pop_size]]
                fitness = fitness[order[:new_pop_size]]
                pop_size = new_pop_size
                archive_max = pop_size
                while len(archive) > archive_max:
                    archive.pop(np.random.randint(len(archive)))
    
    # ==================== CMA-ES ====================
    def run_cmaes(x0, sigma0, time_budget):
        nonlocal best, best_params
        
        deadline = elapsed() + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        
        xmean = x0.copy()
        sigma = sigma0
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        best_local = float('inf')
        stag = 0
        
        while elapsed() < deadline:
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k_i in range(lam):
                arx[k_i] = xmean + sigma * (B @ (D * arz[k_i]))
                arx[k_i] = clip(arx[k_i])
            
            # Evaluate
            arfitness = np.zeros(lam)
            for k_i in range(lam):
                if elapsed() >= deadline:
                    return
                arfitness[k_i] = eval_func(arx[k_i])
            
            counteval += lam
            
            # Sort
            arindex = np.argsort(arfitness)
            
            if arfitness[arindex[0]] < best_local:
                best_local = arfitness[arindex[0]]
                stag = 0
            else:
                stag += 1
            
            # Recombination
            xold = xmean.copy()
            selected = arx[arindex[:mu]]
            xmean = np.sum(weights[:, None] * selected, axis=0)
            
            # CSA
            zmean = np.sum(weights[:, None] * arz[arindex[:mu]], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
            
            # Covariance matrix adaptation
            artmp = (selected - xold) / sigma
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Sigma adaptation
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.mean(ranges))
            
            # Update B and D
            if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            # Termination conditions
            if sigma < 1e-12:
                break
            if stag > 20 + 3 * n:
                break
            if sigma * np.max(D) < 1e-11:
                break
    
    # ==================== Nelder-Mead ====================
    def nelder_mead(x0, time_budget, initial_scale=0.05):
        nonlocal best, best_params
        
        deadline = elapsed() + time_budget
        n = dim
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += initial_scale * ranges[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if elapsed() >= deadline:
                return
            f_simplex[i] = eval_func(simplex[i
