#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: CMA-ES inspired search ---
    # Initialize population
    pop_size = min(4 + int(3 * np.log(dim)), 50)
    if pop_size % 2 == 1:
        pop_size += 1
    
    # Multiple restarts with different strategies
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params
        x_clipped = clip(x)
        f = func(x_clipped)
        if f < best:
            best = f
            best_params = x_clipped.copy()
        return f
    
    # Latin Hypercube initial sampling
    n_init = min(max(10 * dim, 50), 200)
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = np.array([np.random.uniform(l, u) for l, u in bounds])
        eval_func(x)
    
    # CMA-ES implementation
    def run_cmaes(mean_init, sigma_init, budget_fraction):
        nonlocal best, best_params
        
        target_time = elapsed() + (max_time - elapsed()) * budget_fraction
        
        mean = mean_init.copy()
        sigma = sigma_init
        n = dim
        lam = pop_size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        # State variables
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = dim <= 50
        
        if use_full_cov:
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        else:
            diagC = np.ones(n)
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        generation = 0
        
        while elapsed() < target_time and elapsed() < max_time * 0.95:
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full_cov:
                # Decompose C every few generations
                if generation - eigeneval > lam / (c1 + cmu) / n / 10 or generation == 0:
                    try:
                        C = np.triu(C) + np.triu(C, 1).T
                        D_vals, B = np.linalg.eigh(C)
                        D_vals = np.maximum(D_vals, 1e-20)
                        D = np.sqrt(D_vals)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                        eigeneval = generation
                    except:
                        C = np.eye(n)
                        D = np.ones(n)
                        B = np.eye(n)
                        invsqrtC = np.eye(n)
                
                for k in range(lam):
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
            else:
                sqrtdiagC = np.sqrt(diagC)
                for k in range(lam):
                    arx[k] = mean + sigma * sqrtdiagC * arz[k]
            
            # Evaluate
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= max_time * 0.95:
                    return
                fitnesses[k] = eval_func(arx[k])
            
            # Sort
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            # CSA
            mean_z = np.sum(weights[:, None] * arz[:mu], axis=0)
            if use_full_cov:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean)) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / (sigma * sqrtdiagC)
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2 / (n + 1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            if use_full_cov:
                artmp = (arx[:mu] - old_mean) / sigma
                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * (artmp.T @ np.diag(weights) @ artmp)
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                diagC = (1 - c1 - cmu) * diagC + \
                        c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            
            # Sigma adaptation
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            generation += 1
            
            # Check convergence
            if sigma < 1e-12:
                return
    
    # Run CMA-ES with restarts
    n_restarts = 0
    while elapsed() < max_time * 0.90:
        remaining = max_time - elapsed()
        if remaining < 0.5:
            break
        
        if n_restarts == 0 and best_params is not None:
            # First run from best known point
            init_mean = best_params.copy()
            init_sigma = np.mean(upper - lower) * 0.3
        else:
            # Random restart with increasing population
            init_mean = np.array([np.random.uniform(l, u) for l, u in bounds])
            init_sigma = np.mean(upper - lower) * 0.3
        
        # Allocate time budget per restart
        budget = min(0.4, max(0.1, remaining / max_time))
        run_cmaes(init_mean, init_sigma, budget)
        n_restarts += 1
    
    # --- Phase 3: Local refinement around best ---
    if best_params is not None:
        # Nelder-Mead style local search
        step = np.mean(upper - lower) * 0.001
        while elapsed() < max_time * 0.98:
            # Random perturbation with decreasing step
            direction = np.random.randn(dim)
            direction = direction / (np.linalg.norm(direction) + 1e-30)
            x_new = best_params + step * direction
            f_new = eval_func(x_new)
            if f_new >= best:
                step *= 0.95
                if step < 1e-15:
                    step = np.mean(upper - lower) * 0.001
    
    return best
