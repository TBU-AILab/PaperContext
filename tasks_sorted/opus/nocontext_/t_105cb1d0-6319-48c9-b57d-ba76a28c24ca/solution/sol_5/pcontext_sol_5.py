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
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- CMA-ES Implementation ---
    def run_cmaes(init_mean, init_sigma, budget_fraction):
        nonlocal best, best_params
        
        time_limit = max_time * budget_fraction
        cma_start = elapsed()
        
        n = dim
        lam = pop_size
        mu = lam // 2
        
        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        # State variables
        mean = init_mean.copy()
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = (n <= 100)
        
        if use_full_cov:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
        else:
            diagC = np.ones(n)
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        gen = 0
        eval_update_eigen = 0
        
        while elapsed() - cma_start < time_limit and elapsed() < max_time - 0.1:
            # Generate offspring
            arz = np.random.randn(lam, n)
            
            if use_full_cov:
                arx = mean + sigma * (arz @ (eigenvectors * np.sqrt(eigenvalues)).T)
            else:
                arx = mean + sigma * arz * np.sqrt(diagC)
            
            # Evaluate
            fitnesses = np.array([eval_func(arx[i]) for i in range(lam)])
            
            if elapsed() >= max_time - 0.1:
                break
            
            # Sort by fitness
            idx = np.argsort(fitnesses)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            # Clip mean to bounds
            mean = clip(mean)
            
            # Update evolution paths
            if use_full_cov:
                invsqrtC = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-20)) @ eigenvectors.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ((mean - old_mean) / sigma / np.sqrt(diagC + 1e-20))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN) < (1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance
            if use_full_cov:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (weights[:, None] * artmp).T @ artmp
                
                eval_update_eigen += lam
                if eval_update_eigen >= lam * 10:
                    eval_update_eigen = 0
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(C)
                        eigenvalues = np.maximum(eigenvalues, 1e-20)
                    except:
                        C = np.eye(n)
                        eigenvalues = np.ones(n)
                        eigenvectors = np.eye(n)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            gen += 1
            
            # Check for convergence / restart condition
            if sigma < 1e-12:
                break
    
    # --- Phase 2: Latin Hypercube Sampling for initial points ---
    n_initial = min(dim * 10, 200)
    
    # Sobol-like sampling via LHS
    for i in range(n_initial):
        if elapsed() >= max_time * 0.05:
            break
        params = np.array([np.random.uniform(l, u) for l, u in bounds])
        eval_func(params)
    
    # --- Phase 3: Run CMA-ES with restarts ---
    restart = 0
    remaining_time = max_time - elapsed()
    
    while elapsed() < max_time - 0.5:
        remaining = max_time - elapsed()
        if remaining < 0.5:
            break
        
        # Different initialization strategies per restart
        if restart == 0 and best_params is not None:
            init_mean = best_params.copy()
            init_sigma = 0.3 * np.mean(upper - lower)
        elif restart % 3 == 0:
            # Random restart
            init_mean = np.array([np.random.uniform(l, u) for l, u in bounds])
            init_sigma = 0.3 * np.mean(upper - lower)
        elif restart % 3 == 1 and best_params is not None:
            # Near best with smaller sigma
            perturbation = np.random.randn(dim) * 0.1 * (upper - lower)
            init_mean = clip(best_params + perturbation)
            init_sigma = 0.1 * np.mean(upper - lower)
        else:
            # Near best with larger sigma
            if best_params is not None:
                init_mean = best_params.copy()
            else:
                init_mean = (lower + upper) / 2.0
            init_sigma = 0.5 * np.mean(upper - lower)
        
        # Budget: divide remaining time, but ensure at least some time per restart
        budget_frac = min(0.4, remaining / max_time)
        run_cmaes(init_mean, init_sigma, budget_frac)
        restart += 1
    
    # --- Phase 4: Local search (Nelder-Mead style) around best ---
    if best_params is not None and elapsed() < max_time - 0.2:
        # Simple coordinate descent refinement
        step_sizes = (upper - lower) * 0.01
        improved = True
        while improved and elapsed() < max_time - 0.1:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time - 0.1:
                    break
                current = best_params.copy()
                
                # Try positive step
                current[i] = min(best_params[i] + step_sizes[i], upper[i])
                f = eval_func(current)
                if f < best:
                    improved = True
                    step_sizes[i] *= 1.5
                    continue
                
                # Try negative step
                current[i] = max(best_params[i] - step_sizes[i], lower[i])
                f = eval_func(current)
                if f < best:
                    improved = True
                    step_sizes[i] *= 1.5
                    continue
                
                step_sizes[i] *= 0.5
    
    return best
