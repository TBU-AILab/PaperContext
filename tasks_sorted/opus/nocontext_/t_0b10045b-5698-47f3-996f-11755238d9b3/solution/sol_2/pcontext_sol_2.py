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
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x_clipped = clip_to_bounds(x)
        val = func(x_clipped)
        if val < best:
            best = val
            best_params = x_clipped.copy()
        return val
    
    # Latin Hypercube initial sampling
    n_init = min(max(10 * dim, 50), 200)
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        params = lower + np.random.rand(dim) * (upper - lower)
        evaluate(params)
    
    # CMA-ES implementation
    def run_cmaes(mean_init, sigma_init, time_budget):
        nonlocal best, best_params
        
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
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
        
        cma_start = datetime.now()
        generation = 0
        
        while True:
            if elapsed() >= max_time * 0.95:
                return
            
            cma_elapsed = (datetime.now() - cma_start).total_seconds()
            if cma_elapsed >= time_budget:
                return
            
            # Generate and evaluate offspring
            arx = np.zeros((pop_size, dim))
            arfitness = np.zeros(pop_size)
            
            for k in range(pop_size):
                if elapsed() >= max_time * 0.95:
                    return
                
                if use_full_cov:
                    z = np.random.randn(dim)
                    arx[k] = mean + sigma * (B @ (D * z))
                else:
                    arx[k] = mean + sigma * np.sqrt(diagC) * np.random.randn(dim)
                
                arfitness[k] = evaluate(arx[k])
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = clip_to_bounds(np.dot(weights, selected))
            
            # Update evolution paths
            if use_full_cov:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (mean - old_mean) / (sigma * np.sqrt(diagC))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2.0 / (dim + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            if use_full_cov:
                artmp = (selected - old_mean) / sigma
                C = (1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)) * C \
                    + c1 * np.outer(pc, pc) \
                    + cmu * artmp.T @ np.diag(weights) @ artmp
                
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                
                eigeneval += 1
                if eigeneval >= 1.0 / (c1 + cmu) / dim / 10:
                    eigeneval = 0
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                    except:
                        C = np.eye(dim)
                        B = np.eye(dim)
                        D = np.ones(dim)
            else:
                artmp = (selected - old_mean) / sigma
                diagC = (1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)) * diagC \
                    + c1 * pc ** 2 \
                    + cmu * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Check for convergence
            if sigma < 1e-12:
                return
            
            generation += 1
    
    # Run CMA-ES with restarts
    remaining = max_time - elapsed()
    n_restarts = 0
    
    while elapsed() < max_time * 0.90:
        remaining = max_time * 0.90 - elapsed()
        if remaining <= 0.1:
            break
        
        # Allocate time for this restart
        time_for_this = min(remaining * 0.5, remaining / max(1, 3 - n_restarts))
        
        if n_restarts == 0 and best_params is not None:
            init_mean = best_params.copy()
            init_sigma = np.max(upper - lower) * 0.3
        else:
            # Random restart with increasing sigma
            init_mean = lower + np.random.rand(dim) * (upper - lower)
            init_sigma = np.max(upper - lower) * (0.3 * (2 ** n_restarts))
            init_sigma = min(init_sigma, np.max(upper - lower) * 0.5)
        
        run_cmaes(init_mean, init_sigma, time_for_this)
        n_restarts += 1
    
    # --- Phase 2: Local refinement with Nelder-Mead ---
    if best_params is not None and elapsed() < max_time * 0.95:
        # Simple Nelder-Mead
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex around best point
        scale = 0.05 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        fvals = np.zeros(n + 1)
        fvals[0] = best
        
        for i in range(n):
            if elapsed() >= max_time * 0.95:
                return best
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += scale[i]
            simplex[i + 1] = clip_to_bounds(simplex[i + 1])
            fvals[i + 1] = evaluate(simplex[i + 1])
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip_to_bounds(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            if elapsed() >= max_time * 0.95:
                return best
            
            if fr < fvals[0]:
                # Expansion
                xe = clip_to_bounds(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = clip_to_bounds(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if elapsed() >= max_time * 0.95:
                                return best
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = clip_to_bounds(simplex[i])
                            fvals[i] = evaluate(simplex[i])
                else:
                    xc = clip_to_bounds(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if elapsed() >= max_time * 0.95:
                                return best
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = clip_to_bounds(simplex[i])
                            fvals[i] = evaluate(simplex[i])
            
            # Check convergence
            if np.max(np.abs(fvals[0] - fvals[-1])) < 1e-15:
                break
    
    return best
