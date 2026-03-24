#Looking at the current algorithm, I can identify several improvements:
#
#1. **Better adaptive DE (L-SHADE with linear population size reduction)**
#2. **Multi-restart strategy with different scales**
#3. **Improved CMA-ES local search with proper adaptation**
#4. **Better time budget allocation**
#5. **Surrogate-assisted pre-screening for expensive evaluations**
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
    
    # Bounce-back clipping
    def bounce_back(mutant, parent):
        result = mutant.copy()
        for d in range(dim):
            if result[d] < lower[d]:
                result[d] = (lower[d] + parent[d]) / 2.0
            elif result[d] > upper[d]:
                result[d] = (upper[d] + parent[d]) / 2.0
        return result

    def run_lshade(time_budget_frac):
        nonlocal best, best_params
        deadline = elapsed() + max_time * time_budget_frac
        
        N_init = min(max(12 * dim, 50), 200)
        N_min = 4
        pop_size = N_init
        
        # LHS initialization
        population = np.zeros((pop_size, dim))
        for d_i in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                population[i, d_i] = lower[d_i] + (perm[i] + np.random.random()) / pop_size * ranges[d_i]
        
        fitness = np.array([eval_func(population[i]) for i in range(pop_size) if time_left() > 0])
        if len(fitness) < pop_size:
            population = population[:len(fitness)]
            pop_size = len(fitness)
        
        # Opposition-based learning
        if time_left() > 0:
            opp_pop = lower + upper - population
            opp_fit = []
            for i in range(min(pop_size, len(opp_pop))):
                if time_left() <= 0:
                    break
                opp_fit.append(eval_func(opp_pop[i]))
            n_opp = len(opp_fit)
            if n_opp > 0:
                all_pop = np.vstack([population, opp_pop[:n_opp]])
                all_fit = np.concatenate([fitness, opp_fit])
                order = np.argsort(all_fit)[:pop_size]
                population = all_pop[order]
                fitness = all_fit[order]
        
        H = 100
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.3)
        k_idx = 0
        
        archive = []
        archive_max = int(N_init * 2.6)
        
        gen = 0
        max_gen_estimate = max(1, int((deadline - elapsed()) / (pop_size * 0.001 + 0.01)))
        
        while elapsed() < deadline and time_left() > 0 and pop_size >= N_min:
            gen += 1
            S_CR, S_F, S_delta = [], [], []
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            # Adaptive p range
            p_min = max(2.0 / pop_size, 0.05)
            p_max = 0.2
            
            for i in range(pop_size):
                if elapsed() >= deadline or time_left() <= 0:
                    break
                
                ri = np.random.randint(H)
                
                # Generate CR
                if M_CR[ri] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                # Generate F (Cauchy)
                F_i = -1
                while F_i <= 0:
                    F_i = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if F_i >= 1.0:
                        F_i = 1.0
                        break
                
                # current-to-pbest/1
                p = np.random.uniform(p_min, p_max)
                n_pbest = max(1, int(np.ceil(p * pop_size)))
                sorted_idx = np.argsort(fitness)
                pbest_idx = sorted_idx[np.random.randint(n_pbest)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size - 1)
                # Avoid i and r1
                exclude = sorted([i, r1]) if i != r1 else [i]
                for ex in exclude:
                    if r2 >= ex:
                        r2 += 1
                # Clamp
                if r2 >= pool_size:
                    r2 = pool_size - 1
                
                if r2 < pop_size:
                    xr2 = population[r2]
                else:
                    xr2 = archive[r2 - pop_size]
                
                mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - xr2)
                mutant = bounce_back(mutant, population[i])
                
                # Binomial crossover
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                trial_f = eval_func(trial)
                
                if trial_f <= fitness[i]:
                    delta = fitness[i] - trial_f
                    if delta > 0:
                        S_CR.append(CR_i)
                        S_F.append(F_i)
                        S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            # Update memory
            if S_CR and S_F:
                weights = np.array(S_delta)
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                else:
                    weights = np.ones(len(S_delta)) / len(S_delta)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                
                mean_sf = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
                M_F[k_idx] = mean_sf
                
                if max(scr) == 0:
                    M_CR[k_idx] = -1
                else:
                    M_CR[k_idx] = np.sum(weights * scr)
                
                k_idx = (k_idx + 1) % H
            
            # Linear population size reduction
            ratio = min(1.0, (elapsed() - (start + max_time - max_time * time_budget_frac)) / (max_time * time_budget_frac) if max_time * time_budget_frac > 0 else 1.0)
            # More robust: use generation count
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * min(gen / max(max_gen_estimate, 1), 1.0))))
            
            if new_size < pop_size:
                order = np.argsort(fitness)[:new_size]
                population = population[order]
                fitness = fitness[order]
                pop_size = new_size
                # Trim archive
                while len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
    
    def run_cmaes_local(sigma_init, time_budget_frac):
        nonlocal best, best_params
        if best_params is None or time_left() <= 0.2:
            return
        
        deadline = elapsed() + max_time * time_budget_frac
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        
        x_mean = best_params.copy()
        sigma = sigma_init
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        eigeneval = 0
        count_eval = 0
        
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        
        while elapsed() < deadline and time_left() > 0.1:
            # Generate offspring
            arxs = []
            arfitness = []
            
            if count_eval - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = count_eval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D_sq, 1e-20))
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            for _ in range(lam):
                if elapsed() >= deadline or time_left() <= 0.05:
                    return
                z = np.random.randn(n)
                x = x_mean + sigma * (B @ (D * z))
                x = clip(x)
                f = eval_func(x)
                arxs.append(x)
                arfitness.append(f)
                count_eval += 1
            
            # Sort
            order = np.argsort(arfitness)
            
            x_old = x_mean.copy()
            x_mean = np.zeros(n)
            for j in range(mu):
                x_mean += weights[j] * arxs[order[j]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (x_mean - x_old) / sigma
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * count_eval / lam)) / chiN < 1.4 + 2.0 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_old) / sigma
            
            # Covariance update
            artmp = np.zeros((mu, n))
            for j in range(mu):
                artmp[j] = (arxs[order[j]] - x_old) / sigma
            
            C = (1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Sigma update
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            
            # Break if sigma too small
            if sigma * np.max(D) < 1e-12 * np.mean(ranges):
                break
    
    def local_pattern_search(time_budget_frac):
        nonlocal best, best_params
        if best_params is None or time_left() <= 0.1:
            return
        
        deadline = elapsed() + max_time * time_budget_frac
        
        x = best_params.copy()
        fx = best
        step = 0.001 * ranges
        
        while elapsed() < deadline and time_left() > 0.05:
            improved = False
            for d in range(dim):
                if elapsed() >= deadline or time_left() <= 0.05:
                    return
                # Try +step
                x_new = x.copy()
                x_new[d] = min(upper[d], x[d] + step[d])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    improved = True
                    continue
                # Try -step
                x_new = x.copy()
                x_new[d] = max(lower[d], x[d] - step[d])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    improved = True
                    continue
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break
    
    # Main strategy: multi-phase with restarts
    # Phase 1: L-SHADE (60% of time)
    run_lshade(
