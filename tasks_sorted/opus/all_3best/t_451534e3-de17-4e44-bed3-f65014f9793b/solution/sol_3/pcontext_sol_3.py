#Looking at the results, the best algorithm (24.58) uses L-SHADE with coordinate descent local search. I'll improve it by:
#
#1. **Adding a simplified CMA-ES** as an alternative optimizer that runs in parallel with L-SHADE
#2. **Better local search** using a pattern search with acceleration
#3. **Smarter time allocation** between exploration and exploitation phases
#4. **Opposition-based learning** for population initialization
#5. **More aggressive exploitation** in later restarts
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_sol, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_sol = x.copy()
        return f

    def pattern_search(x0, f0, init_step=0.05, min_step=1e-13, max_evals=None):
        if max_evals is None:
            max_evals = dim * 20
        x_cur = x0.copy()
        f_cur = f0
        step = init_step * ranges.copy()
        n_evals = 0
        
        while n_evals < max_evals and remaining() > 0.1:
            improved = False
            for d in range(dim):
                if remaining() <= 0.05:
                    return x_cur, f_cur
                
                # Positive step
                x_try = x_cur.copy()
                x_try[d] += step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    # Acceleration: try doubling
                    x_cur = x_try
                    f_cur = f_try
                    step[d] *= 1.5
                    improved = True
                    if n_evals < max_evals and remaining() > 0.05:
                        x_try2 = x_cur.copy()
                        x_try2[d] += step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2
                            f_cur = f_try2
                            step[d] *= 1.5
                    continue
                
                # Negative step
                x_try = x_cur.copy()
                x_try[d] -= step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    step[d] *= 1.5
                    improved = True
                    if n_evals < max_evals and remaining() > 0.05:
                        x_try2 = x_cur.copy()
                        x_try2[d] -= step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2
                            f_cur = f_try2
                            step[d] *= 1.5
                else:
                    step[d] *= 0.5
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < min_step:
                    break
        
        return x_cur, f_cur

    def run_cmaes(x0, sigma0, max_evals_cma):
        """Simplified (mu/mu_w, lambda)-CMA-ES"""
        n = dim
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for efficiency
        C_diag = np.ones(n)
        
        n_eval = 0
        
        while n_eval < max_evals_cma and remaining() > 0.2:
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.empty((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * np.sqrt(C_diag) * arz[k]
                arx[k] = clip(arx[k])
            
            # Evaluate
            fit = np.empty(lam)
            for k in range(lam):
                if remaining() <= 0.1:
                    return
                fit[k] = eval_func(arx[k])
                n_eval += 1
            
            # Sort
            idx = np.argsort(fit)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            
            # Update evolution paths
            zmean = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean / np.sqrt(C_diag + 1e-30)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(n_eval/lam+1))) / chiN < 1.4 + 2.0/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * zmean
            
            # Update diagonal covariance
            C_diag = (1 - c1 - cmu_val) * C_diag + c1 * (pc**2 + (1-hsig)*cc*(2-cc)*C_diag)
            for k in range(mu):
                diff = (arx[idx[k]] - old_mean) / sigma
                C_diag += cmu_val * weights[k] * diff**2
            
            C_diag = np.maximum(C_diag, 1e-20)
            
            # Update sigma
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            if sigma < 1e-15:
                break

    # ====== Phase 1: L-SHADE exploration ======
    restart_count = 0
    
    while remaining() > 0.5:
        restart_count += 1
        time_for_this = remaining()
        
        N_init = min(max(25, 8 * dim), 250)
        N_min = max(4, dim)
        pop_size = N_init
        max_nfe_estimate = int(time_for_this * 400)
        nfe_at_start = evals
        
        H = 80
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        if restart_count > 1:
            memory_F = np.full(H, 0.2 + 0.6 * np.random.rand())
            memory_CR = np.full(H, 0.2 + 0.6 * np.random.rand())
        k_idx = 0
        
        archive = []
        archive_max = N_init
        
        # Initialize with opposition-based learning
        pop_half = N_init // 2
        pop1 = np.random.uniform(lower, upper, (pop_half, dim))
        pop2 = lower + upper - pop1  # opposition
        population = np.vstack([pop1, pop2])[:N_init]
        
        if restart_count > 1 and best_sol is not None:
            n_local = max(1, pop_size // 5)
            scale = 0.2 * (0.5 ** min(restart_count - 2, 5))
            for j in range(n_local):
                population[j] = clip(best_sol + scale * ranges * np.random.randn(dim))
        
        fitness = np.array([eval_func(ind) for ind in population])
        
        if remaining() <= 0.5:
            break
        
        generation = 0
        stagnation = 0
        prev_best = best
        
        while remaining() > 0.5:
            generation += 1
            
            nfe_since_start = evals - nfe_at_start
            ratio = min(1.0, nfe_since_start / max(1, max_nfe_estimate))
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                si = np.argsort(fitness)
                population = population[si[:new_pop_size]]
                fitness = fitness[si[:new_pop_size]]
                pop_size = new_pop_size
            
            p_min = 2.0 / pop_size
            p_max = 0.25
            p = p_max - (p_max - p_min) * ratio
            p_best_size = max(2, int(p * pop_size))
            
            ri = np.random.randint(0, H, pop_size)
            
            Fs = np.empty(pop_size)
            for idx in range(pop_size):
                attempts = 0
                while attempts < 20:
                    f_val = memory_F[ri[idx]] + 0.1 * np.random.standard_cauchy()
                    attempts += 1
                    if f_val > 0:
                        Fs[idx] = min(f_val, 1.0)
                        break
                else:
                    Fs[idx] = 0.5
            
            CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0.0, 1.0)
            
            S_F, S_CR, S_delta = [], [], []
            sorted_idx = np.argsort(fitness)
            
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if remaining() <= 0.3:
                    break
                
                pi = sorted_idx[np.random.randint(0, p_best_size)]
                
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                
                combined_size = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(combined_size)
                
                x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - x_r2)
                
                jrand = np.random.randint(dim)
                mask = (np.random.rand(dim) < CRs[i])
                mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                
                below = trial < lower
                above = trial > upper
                trial[below] = (lower[below] + population[i][below]) / 2.0
                trial[above] = (upper[above] + population[i][above]) / 2.0
                
                trial_f = eval_func(trial)
                
                if trial_f <= fitness[i]:
                    delta = fitness[i] - trial_f
                    if trial_f < fitness[i]:
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(Fs[i])
                        S_CR.append(CRs[i])
                        S_delta.append(delta + 1e-30)
                    new_population[i] = trial
                    new_fitness[i] = trial_f
            
            population = new_population
            fitness = new_fitness
            
            if S_F:
                weights = np.array(S_delta)
                weights /= weights.sum()
                sf = np.array(S_F)
                scr = np.array(S_CR)
                mean_F = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
                mean_CR = np.sum(weights * scr)
                memory_F[k_idx % H] = mean_F
                memory_CR[k_idx % H] = mean_CR
                k_idx += 1
            
            if abs(prev_best - best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = best
            
            # Periodic CMA-ES burst
            if generation % 40 == 0 and best_sol is not None and remaining() > 2:
                run_cmaes(best_sol, 0.1 * np.mean(ranges) * max(0.01, 1.0 - ratio), max_evals_cma=dim*8)
            
            # Periodic pattern search
            if generation % 30 == 0 and best_sol is not None and remaining() > 1:
                sigma = 0.02 * max(0.05, 1.0 - ratio)
                best_sol, _ = pattern_search(best_sol, best, init_step=sigma, max_evals=dim*8)
            
            if stagnation > 50 or pop_size <= N_min:
                if best_sol is not None and remaining() > 1:
                    best_sol, _ = pattern_search(best_sol, best, init_step=0.01, max_evals=dim*15)
                    if remaining() > 1:
                        run_cmaes(best_sol, 0.05 * np.mean(ranges), max_evals_cma=dim*15)
                break
    
    # Final intensive local search
    if best_sol is not None and remaining() > 0.2:
        best_sol, _ = pattern_search(best_sol, best, init_step=0.001, min_step=1e-15, max_evals=dim*50)
    
    return best
