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
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x_clipped = clip_to_bounds(x)
        f = func(x_clipped)
        if f < best:
            best = f
            best_params = x_clipped.copy()
        return f
    
    # Initial random sampling to find good starting points
    n_init = min(dim * 10, 200)
    init_points = []
    init_fits = []
    
    for i in range(n_init):
        if time_left() < 0.1:
            return best
        x = lower + np.random.rand(dim) * (upper - lower)
        f = evaluate(x)
        init_points.append(x.copy())
        init_fits.append(f)
    
    # Also try latin hypercube-like sampling
    n_lhs = min(dim * 5, 100)
    for i in range(n_lhs):
        if time_left() < 0.1:
            return best
        # Sobol-like: use stratified sampling
        x = lower + np.random.rand(dim) * (upper - lower)
        f = evaluate(x)
        init_points.append(x.copy())
        init_fits.append(f)
    
    init_points = np.array(init_points)
    init_fits = np.array(init_fits)
    
    # CMA-ES implementation
    def run_cmaes(x0, sigma0, budget_fraction=1.0):
        nonlocal best, best_params
        
        n = dim
        lam = pop_size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        # State variables
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = (n <= 100)
        
        if use_full_cov:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            invsqrtC = np.eye(n)
        else:
            diag_C = np.ones(n)
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        eigen_update_freq = max(1, int(1 / (10 * n * (c1 + cmu_val))))
        gen = 0
        
        max_cma_time = time_left() * budget_fraction
        cma_start = datetime.now()
        
        while True:
            elapsed = (datetime.now() - cma_start).total_seconds()
            if elapsed > max_cma_time or time_left() < 0.1:
                break
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            
            if use_full_cov:
                arx = mean + sigma * (arz @ (eigenvectors * np.sqrt(eigenvalues)).T)
            else:
                arx = mean + sigma * arz * np.sqrt(diag_C)
            
            # Evaluate
            fitnesses = np.zeros(lam)
            for i in range(lam):
                if time_left() < 0.05:
                    return
                fitnesses[i] = evaluate(arx[i])
            
            # Sort
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, arx[:mu])
            
            # Update evolution paths
            if use_full_cov:
                zmean = np.dot(weights, arz[:mu])
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
            else:
                zmean = np.dot(weights, arz[:mu])
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean / np.sqrt(diag_C)
                # Approximate
                ps_temp = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / sigma
                ps = ps_temp
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance
            if use_full_cov:
                artmp = (arx[:mu] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (artmp.T @ np.diag(weights) @ artmp)
                
                # Update eigen decomposition periodically
                if gen % eigen_update_freq == 0:
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(C)
                        eigenvalues = np.maximum(eigenvalues, 1e-20)
                        invsqrtC = eigenvectors @ np.diag(1.0/np.sqrt(eigenvalues)) @ eigenvectors.T
                    except:
                        C = np.eye(n)
                        eigenvalues = np.ones(n)
                        eigenvectors = np.eye(n)
                        invsqrtC = np.eye(n)
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                diag_C = (1 - c1 - cmu_val) * diag_C + \
                         c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diag_C) + \
                         cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Check for convergence
            if sigma < 1e-12:
                break
            
            if use_full_cov and np.max(eigenvalues) / np.min(eigenvalues) > 1e14:
                break
            
            gen += 1
    
    # Run CMA-ES from best found points with restarts
    sorted_idx = np.argsort(init_fits)
    
    n_restarts = 0
    max_restarts = 10 + dim
    
    while time_left() > 0.5 and n_restarts < max_restarts:
        if n_restarts < len(sorted_idx):
            x0 = init_points[sorted_idx[n_restarts]].copy()
        elif best_params is not None:
            # Restart near best with perturbation
            x0 = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
            x0 = clip_to_bounds(x0)
        else:
            x0 = lower + np.random.rand(dim) * (upper - lower)
        
        # Adaptive sigma
        if n_restarts == 0:
            sigma0 = 0.3 * np.mean(upper - lower)
        elif n_restarts < 3:
            sigma0 = 0.2 * np.mean(upper - lower)
        else:
            sigma0 = (0.1 + 0.4 * np.random.rand()) * np.mean(upper - lower)
        
        remaining = time_left()
        budget = min(remaining * 0.4, remaining - 0.3)
        if budget < 0.3:
            break
        
        run_cmaes(x0, sigma0, budget_fraction=budget / max(time_left(), 0.01))
        n_restarts += 1
    
    # Final local search (Nelder-Mead style) around best point
    if best_params is not None and time_left() > 0.3:
        # Simple pattern search
        step = 0.001 * (upper - lower)
        current = best_params.copy()
        current_fit = best
        
        while time_left() > 0.1:
            improved = False
            for i in range(dim):
                if time_left() < 0.05:
                    return best
                
                # Try positive direction
                trial = current.copy()
                trial[i] = min(trial[i] + step[i], upper[i])
                f = evaluate(trial)
                if f < current_fit:
                    current = trial
                    current_fit = f
                    improved = True
                    continue
                
                # Try negative direction
                trial = current.copy()
                trial[i] = max(trial[i] - step[i], lower[i])
                f = evaluate(trial)
                if f < current_fit:
                    current = trial
                    current_fit = f
                    improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / (upper - lower)) < 1e-10:
                    break
    
    return best
