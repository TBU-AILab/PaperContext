#Looking at the current algorithm, I can identify several improvements:
#
#1. **Better CMA-ES implementation** with proper path evolution and step-size adaptation
#2. **Multiple restart strategy** with decreasing population sizes
#3. **More sophisticated SHADE** with linear population size reduction (L-SHADE)
#4. **Better time allocation** between exploration and exploitation phases
#
import numpy as np
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
    
    def bounce_back(mutant, parent):
        x = mutant.copy()
        for d in range(dim):
            if x[d] < lower[d]:
                x[d] = (lower[d] + parent[d]) / 2.0
            elif x[d] > upper[d]:
                x[d] = (upper[d] + parent[d]) / 2.0
        return x
    
    def eval_func(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # ---- L-SHADE Phase ----
    def run_lshade(time_budget_frac=0.70):
        nonlocal best, best_params
        
        budget_end = elapsed() + max_time * time_budget_frac
        
        N_init = min(max(18 * dim, 50), 300)
        N_min = 4
        pop_size = N_init
        
        # Latin Hypercube Sampling
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
        
        fitness = np.array([eval_func(population[i]) for i in range(pop_size) if time_left() > 0])
        if len(fitness) < pop_size:
            fitness = np.append(fitness, [float('inf')] * (pop_size - len(fitness)))
        
        # Opposition-based initialization
        opp_pop = lower + upper - population
        opp_fit = np.array([eval_func(opp_pop[i]) for i in range(pop_size) if time_left() > 0])
        if len(opp_fit) < pop_size:
            opp_fit = np.append(opp_fit, [float('inf')] * (pop_size - len(opp_fit)))
        
        all_pop = np.vstack([population, opp_pop])
        all_fit = np.concatenate([fitness, opp_fit])
        order = np.argsort(all_fit)[:pop_size]
        population = all_pop[order].copy()
        fitness = all_fit[order].copy()
        
        H = 100
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.3)
        k_idx = 0
        
        archive = []
        archive_max = N_init
        
        gen = 0
        max_gen_estimate = 5000  # rough estimate
        
        while time_left() > 0 and elapsed() < budget_end and pop_size >= N_min:
            gen += 1
            S_CR, S_F, S_delta = [], [], []
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            # Sorted indices for pbest selection
            sorted_idx = np.argsort(fitness)
            
            for i in range(pop_size):
                if time_left() <= 0 or elapsed() >= budget_end:
                    break
                
                ri = np.random.randint(H)
                
                # CR generation
                if M_CR[ri] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                # F generation (Cauchy)
                F_i = -1
                while F_i <= 0:
                    F_i = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if F_i >= 1.0:
                        F_i = 1.0
                        break
                
                # p for current-to-pbest
                p_i = np.random.uniform(2.0 / pop_size, 0.2)
                n_pbest = max(1, int(np.ceil(p_i * pop_size)))
                pbest_idx = sorted_idx[np.random.randint(n_pbest)]
                
                # r1 from pop (not i)
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                # r2 from pop + archive (not i, not r1)
                combined_size = pop_size + len(archive)
                r2 = np.random.randint(combined_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(combined_size)
                
                if r2 < pop_size:
                    xr2 = population[r2]
                else:
                    xr2 = archive[r2 - pop_size]
                
                # Mutation: current-to-pbest/1
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
                w = np.array(S_delta)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                
                M_F[k_idx] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                
                if max(scr) == 0:
                    M_CR[k_idx] = -1
                else:
                    M_CR[k_idx] = np.sum(w * scr ** 2) / (np.sum(w * scr) + 1e-30)
                
                k_idx = (k_idx + 1) % H
            
            # Linear population size reduction
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * gen / max(max_gen_estimate, gen + 1))))
            if new_pop_size < pop_size:
                order = np.argsort(fitness)[:new_pop_size]
                population = population[order]
                fitness = fitness[order]
                pop_size = new_pop_size
    
    # ---- CMA-ES Phase ----
    def run_cmaes(sigma_init=0.2, time_frac_end=0.98):
        nonlocal best, best_params
        
        if best_params is None:
            return
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        xmean = best_params.copy()
        sigma = sigma_init * np.mean(ranges)
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        while time_left() > max_time * (1 - time_frac_end + 0.01):
            # Sample
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            for k in range(lam):
                if time_left() <= 0.05:
                    return
                z = np.random.randn(n)
                arx[k] = xmean + sigma * (B @ (D * z))
                arx[k] = clip(arx[k])
                arf[k] = eval_func(arx[k])
                counteval += 1
            
            # Sort
            arindex = np.argsort(arf)
            xold = xmean.copy()
            
            # Recombination
            xmean = np.zeros(n)
            for j in range(mu):
                xmean += weights[j] * arx[arindex[j]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ ((xmean - xold) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN < 1.4 + 2.0 / (n + 1))
            
            # CCA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
            
            # Covariance matrix adaptation
            artmp = np.zeros((mu, n))
            for j in range(mu):
                artmp[j] = (arx[arindex[j]] - xold) / sigma
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for j in range(mu):
                C += cmu * weights[j] * np.outer(artmp[j], artmp[j])
            
            # Step-size update
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.mean(ranges))
            
            # Update eigen decomposition
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
            
            if sigma < 1e-15 * np.mean(ranges):
                # Restart with smaller sigma
                sigma = 0.05 * np.mean(ranges)
                xmean = best_params.copy()
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                invsqrtC = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
    
    # Execute phases
    run_lshade(time_budget_frac=0.60)
    run_cmaes(sigma_init=0.15, time_frac_end=0.98)
    
    # Final local refinement with small perturbations
    if best_params is not None:
        sigma_final = 0.001 * np.mean(ranges)
        while time_left() > 0.02:
            trial = best_params + np.random.randn(dim) * sigma_final
            eval_func(trial)
    
    return best
