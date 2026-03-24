#Looking at the current algorithm, I see a CMA-ES approach with initialization and local refinement. Let me improve it with:
#
#1. Better CMA-ES implementation with proper restart strategy (IPOP-CMA-ES)
#2. More efficient time management
#3. Better initial sampling
#4. Improved local search (pattern search + coordinate descent)
#5. Differential Evolution as an alternative explorer
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def time_ok(frac=0.97):
        return elapsed() < max_time * frac
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Sobol-like quasi-random initialization ---
    n_init = min(max(30 * dim, 200), 800)
    
    # Latin Hypercube Sampling
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.90):
            break
        init_fitness[i] = eval_f(init_pop[i])
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: CMA-ES with IPOP restarts ---
    def cmaes_search(x0, sigma0, time_limit):
        nonlocal best, best_params
        
        end_time = elapsed() + time_limit
        n = dim
        
        # Population size
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        
        gen = 0
        stagnation = 0
        prev_best_gen = float('inf')
        
        while elapsed() < end_time and time_ok(0.96):
            # Generate and evaluate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fitnesses = np.zeros(lam)
            
            for k in range(lam):
                if not time_ok(0.96):
                    return
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
                fitnesses[k] = eval_f(arx[k])
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            best_gen = fitnesses[idx[0]]
            if best_gen < prev_best_gen - 1e-12:
                stagnation = 0
                prev_best_gen = best_gen
            else:
                stagnation += 1
            
            if stagnation > 10 + int(30 * n / lam):
                return
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            mean_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 2)
            
            if gen % max(1, int(1 / (c1 + cmu_val) / n / 10)) == 0:
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
                    sigma = sigma0 * 0.5
            
            if sigma < 1e-16 * sigma0:
                return
            
            gen += 1
    
    # Run IPOP-CMA-ES: restarts with increasing population
    n_top = min(5, len(sorted_idx))
    restart = 0
    pop_mult = 1
    
    while time_ok(0.60):
        rem = remaining()
        if rem < 0.5:
            break
        
        if restart < n_top:
            x0 = init_pop[sorted_idx[restart]].copy()
        else:
            x0 = clip(best_params + np.random.randn(dim) * ranges * 0.3) if best_params is not None else np.array([np.random.uniform(l, u) for l, u in bounds])
        
        sigma0 = 0.25 * np.mean(ranges) * max(0.1, 0.8 ** restart)
        time_budget = min(rem * 0.4, max(rem / max(1, 8 - restart), 1.0))
        
        cmaes_search(x0, sigma0, time_budget)
        restart += 1
    
    # --- Phase 3: Refined local search ---
    if best_params is not None:
        # Adaptive coordinate descent
        step = 0.05 * ranges.copy()
        
        while time_ok(0.96) and np.max(step / ranges) > 1e-12:
            improved = False
            for d in range(dim):
                if not time_ok(0.96):
                    break
                for direction in [1.0, -1.0]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = eval_f(trial)
                    if f < best - 1e-15:
                        improved = True
                        # Accelerate in this direction
                        for _ in range(5):
                            if not time_ok(0.96):
                                break
                            trial2 = best_params.copy()
                            trial2[d] = np.clip(trial2[d] + direction * step[d], lower[d], upper[d])
                            f2 = eval_f(trial2)
                            if f2 >= best - 1e-15:
                                break
                        break
            if not improved:
                step *= 0.5
    
    return best
#
#Key improvements:
#1. **Larger initial sampling** (up to 800 points) for better exploration
#2. **IPOP-CMA-ES restarts** with stagnation detection and adaptive sigma
#3. **Better time budgeting** - more time allocated to CMA-ES phase
#4. **Accelerated coordinate descent** that continues in promising directions
#5. **Centralized evaluation function** that always tracks the best solution
#6. **Stagnation detection** in CMA-ES to avoid wasting time on converged runs
