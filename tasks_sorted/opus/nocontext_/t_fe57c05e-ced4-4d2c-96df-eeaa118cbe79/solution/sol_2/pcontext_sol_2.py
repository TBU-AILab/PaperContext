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
    
    # Start with multiple restarts of a simple CMA-ES
    sigma0 = 0.3  # initial step size relative to bounds range
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    # Latin hypercube sampling for initial points
    def latin_hypercube(n):
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                points[i, d] = lower[d] + (upper[d] - lower[d]) * (perm[i] + np.random.random()) / n
        return points
    
    # Evaluate initial samples
    n_init = min(max(10 * dim, 50), 200)
    init_points = latin_hypercube(n_init)
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        fitness = func(init_points[i])
        if fitness < best:
            best = fitness
            best_params = init_points[i].copy()
    
    if best_params is None:
        best_params = (lower + upper) / 2.0
    
    # CMA-ES implementation
    def cma_es(x0, sigma, budget_fraction=0.9):
        nonlocal best, best_params
        
        target_time = elapsed() + remaining() * budget_fraction
        
        n = len(x0)
        lam = pop_size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal approximation for high dimensions
        use_diag = n > 50
        
        if use_diag:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            eigeneval = 0
            B = np.eye(n)
            D = np.ones(n)
        
        mean = x0.copy()
        sig = sigma
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        gen = 0
        stagnation = 0
        prev_best = best
        
        while elapsed() < target_time:
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_diag:
                sqrt_C = np.sqrt(C_diag)
                for k in range(lam):
                    arx[k] = clip_to_bounds(mean + sig * sqrt_C * arz[k])
            else:
                for k in range(lam):
                    arx[k] = clip_to_bounds(mean + sig * (B @ (D * arz[k])))
            
            # Evaluate
            fitvals = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= target_time:
                    return
                fitvals[k] = func(arx[k])
                if fitvals[k] < best:
                    best = fitvals[k]
                    best_params = arx[k].copy()
            
            # Sort by fitness
            idx = np.argsort(fitvals)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            # Update evolution paths
            if use_diag:
                inv_sqrt_C = 1.0 / np.sqrt(C_diag)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C * (mean - old_mean) / sig
            else:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean)) / sig
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sig
            
            if use_diag:
                # Diagonal CMA update
                artmp = (arx[idx[:mu]] - old_mean) / sig
                C_diag = (1 - c1 - cmu_val) * C_diag + \
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * C_diag) + \
                         cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sig
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (weights[:, None] * artmp).T @ artmp
                
                eigeneval += lam
                if eigeneval >= lam / (c1 + cmu_val) / n / 10:
                    eigeneval = 0
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                    except np.linalg.LinAlgError:
                        # Reset
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
            
            # Update sigma
            sig = sig * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sig = min(sig, np.max(upper - lower))  # cap sigma
            
            gen += 1
            
            # Check stagnation
            if best >= prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = best
            
            if stagnation > 10 + int(30 * n / lam):
                return  # restart
            
            if sig < 1e-12:
                return  # converged
    
    # Run CMA-ES with restarts
    restart_count = 0
    while elapsed() < max_time * 0.95:
        # Choose starting point
        if restart_count == 0:
            x0 = best_params.copy()
            sig = sigma0 * np.mean(upper - lower)
        else:
            # Restart: mix of best known and random
            if np.random.random() < 0.3:
                x0 = best_params.copy() + 0.1 * (upper - lower) * np.random.randn(dim)
                x0 = clip_to_bounds(x0)
                sig = 0.1 * np.mean(upper - lower)
            else:
                x0 = np.array([np.random.uniform(l, h) for l, h in bounds])
                sig = 0.3 * np.mean(upper - lower)
        
        # Allocate time: diminishing fractions
        r = remaining()
        if r < 0.5:
            break
        fraction = min(0.5, max(0.1, 1.0 / (restart_count + 2)))
        
        cma_es(x0, sig, budget_fraction=fraction)
        restart_count += 1
    
    # Final local search (Nelder-Mead style)
    if best_params is not None and remaining() > 0.5:
        # Quick coordinate descent refinement
        step = 0.001 * (upper - lower)
        improved = True
        while improved and elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = func(trial)
                    if f < best:
                        best = f
                        best_params = trial.copy()
                        improved = True
    
    return best
