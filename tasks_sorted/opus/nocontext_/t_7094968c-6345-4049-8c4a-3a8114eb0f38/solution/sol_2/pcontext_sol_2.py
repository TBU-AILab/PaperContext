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
    if pop_size % 2 != 0:
        pop_size += 1
    
    # Multiple restarts with different strategies
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
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
        params = lower + np.random.rand(dim) * (upper - lower)
        evaluate(params)
    
    # CMA-ES implementation
    def run_cmaes(mean_init, sigma_init, budget_fraction):
        nonlocal best, best_params
        
        target_time = elapsed() + (max_time - elapsed()) * budget_fraction
        
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        mean = mean_init.copy()
        sigma = sigma_init
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = dim <= 100
        
        if use_full_cov:
            C = np.eye(dim)
            eigeneval = 0
            B = np.eye(dim)
            D = np.ones(dim)
        else:
            diagC = np.ones(dim)
        
        generation = 0
        
        while elapsed() < target_time and elapsed() < max_time * 0.95:
            generation += 1
            
            # Sample population
            if use_full_cov:
                if generation % (1 + int(10 * dim / pop_size)) == 0 or generation == 1:
                    try:
                        C = (C + C.T) / 2
                        D_sq, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D_sq, 1e-20))
                    except:
                        C = np.eye(dim)
                        B = np.eye(dim)
                        D = np.ones(dim)
                
                arz = np.random.randn(pop_size, dim)
                arx = np.array([mean + sigma * (B @ (D * z)) for z in arz])
            else:
                sqrtC = np.sqrt(np.maximum(diagC, 1e-20))
                arz = np.random.randn(pop_size, dim)
                arx = np.array([mean + sigma * sqrtC * z for z in arz])
            
            # Evaluate
            fitnesses = np.array([evaluate(x) for x in arx])
            
            if elapsed() >= max_time * 0.95:
                return
            
            # Sort
            idx = np.argsort(fitnesses)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            # Update evolution paths
            mean_diff = (mean - old_mean) / sigma
            
            if use_full_cov:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff)
            else:
                invsqrtC = 1.0 / sqrtC
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC * mean_diff)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*generation)) / chiN) < (1.4 + 2/(dim+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff
            
            # Update covariance
            if use_full_cov:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * (artmp.T @ np.diag(weights) @ artmp)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                diagC = (1 - c1 - cmu) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu * np.sum(weights[:, None] * artmp**2, axis=0)
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Check for stagnation
            if sigma < 1e-12:
                return
    
    # Run CMA-ES with multiple restarts
    # First run from best found point
    if best_params is not None:
        sigma0 = np.mean(upper - lower) / 4
        run_cmaes(best_params.copy(), sigma0, 0.4)
    
    # Restart with increased population and from best
    restart = 0
    while elapsed() < max_time * 0.95:
        restart += 1
        remaining = max_time - elapsed()
        if remaining < 0.1:
            break
        
        # Alternate between random starts and best point perturbation
        if restart % 3 == 0 and best_params is not None:
            mean_init = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
            mean_init = clip(mean_init)
            sigma0 = np.mean(upper - lower) / 10
        else:
            mean_init = lower + np.random.rand(dim) * (upper - lower)
            sigma0 = np.mean(upper - lower) / 3
        
        budget = min(0.3, 1.0)
        run_cmaes(mean_init, sigma0, budget)
    
    # --- Phase 3: Local Nelder-Mead style refinement ---
    if best_params is not None and elapsed() < max_time * 0.92:
        # Simple coordinate descent refinement
        step = (upper - lower) * 0.01
        improved = True
        while improved and elapsed() < max_time * 0.95:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                current = best_params.copy()
                # Try positive step
                current[i] = min(current[i] + step[i], upper[i])
                f = evaluate(current)
                if f < best:
                    improved = True
                    step[i] *= 1.2
                    continue
                # Try negative step
                current = best_params.copy()
                current[i] = max(current[i] - step[i], lower[i])
                f = evaluate(current)
                if f < best:
                    improved = True
                    step[i] *= 1.2
                    continue
                step[i] *= 0.5
    
    return best
