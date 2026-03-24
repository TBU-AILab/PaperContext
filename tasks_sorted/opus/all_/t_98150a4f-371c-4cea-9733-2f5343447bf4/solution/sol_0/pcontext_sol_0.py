#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * (upper[d] - lower[d])
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(init_points[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_x = init_points[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Sort and pick top candidates
    sorted_idx = np.argsort(init_fitness)
    
    # Run multiple restarts of a simplified CMA-ES / Nelder-Mead hybrid
    def cma_es_search(x0, sigma0, budget_fraction):
        nonlocal best, best_x
        
        target_end = start + timedelta(seconds=max_time * budget_fraction)
        
        n = dim
        lam = max(4 + int(3 * np.log(n)), 10)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        if n > 50:
            diag_cov = np.ones(n)
            use_full = False
        else:
            C = np.eye(n)
            use_full = True
        
        eigeneval = 0
        counteval = 0
        
        while True:
            passed_time = datetime.now()
            if passed_time >= target_end:
                break
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                except:
                    sqrt_C = np.eye(n)
                    C = np.eye(n)
                
                for k in range(lam):
                    arx[k] = mean + sigma * (sqrt_C @ arz[k])
            else:
                sqrt_diag = np.sqrt(np.maximum(diag_cov, 1e-20))
                for k in range(lam):
                    arx[k] = mean + sigma * sqrt_diag * arz[k]
            
            # Clip to bounds
            arx = np.clip(arx, lower, upper)
            
            # Evaluate
            arfitness = np.full(lam, float('inf'))
            for k in range(lam):
                passed_time = datetime.now()
                if passed_time >= target_end:
                    return
                f = func(arx[k])
                arfitness[k] = f
                counteval += 1
                if f < best:
                    best = f
                    best_x = arx[k].copy()
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[arindex[k]]
            
            # Update evolution paths
            diff = mean - old_mean
            
            if use_full:
                try:
                    inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ diff) / sigma
                except:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / sigma
            else:
                inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(diag_cov, 1e-20))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_diag * diff / sigma
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            # Covariance matrix adaptation
            if use_full:
                artmp = np.zeros((mu, n))
                for k in range(mu):
                    artmp[k] = (arx[arindex[k]] - old_mean) / sigma
                
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * sum(weights[k] * np.outer(artmp[k], artmp[k]) for k in range(mu))
                
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
            else:
                artmp = np.zeros((mu, n))
                for k in range(mu):
                    artmp[k] = (arx[arindex[k]] - old_mean) / sigma
                
                diag_cov = (1 - c1 - cmu_val) * diag_cov + \
                    c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_cov) + \
                    cmu_val * sum(weights[k] * artmp[k] ** 2 for k in range(mu))
            
            # Step size update
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Break if sigma is too small
            if sigma < 1e-16:
                break
    
    # Allocate time for CMA-ES restarts
    n_restarts = min(5, max(1, dim // 5))
    remaining_fraction_per_restart = 0.95 / n_restarts
    
    for r in range(n_restarts):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            break
        
        # Pick starting point
        if r == 0:
            x0 = best_x.copy()
        else:
            # Pick from top candidates with some randomness
            idx = sorted_idx[r % min(len(sorted_idx), 10)]
            x0 = init_points[idx].copy()
        
        # Initial sigma
        sigma0 = 0.3 * np.mean(upper - lower) / (r + 1)
        
        elapsed = (datetime.now() - start).total_seconds()
        budget_end = min(elapsed / max_time + remaining_fraction_per_restart, 0.98)
        
        cma_es_search(x0, sigma0, budget_end)
    
    # --- Phase 3: Local refinement with Nelder-Mead style simplex ---
    if best_x is not None:
        passed_time = (datetime.now() - start)
        if passed_time < timedelta(seconds=max_time * 0.95):
            # Simple pattern search refinement
            step = 0.01 * (upper - lower)
            x_curr = best_x.copy()
            f_curr = best
            
            while True:
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    break
                
                improved = False
                for d in range(dim):
                    passed_time = (datetime.now() - start)
                    if passed_time >= timedelta(seconds=max_time * 0.98):
                        break
                    
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[d] = np.clip(x_trial[d] + direction * step[d], lower[d], upper[d])
                        f_trial = func(x_trial)
                        if f_trial < f_curr:
                            x_curr = x_trial
                            f_curr = f_trial
                            if f_curr < best:
                                best = f_curr
                                best_x = x_curr.copy()
                            improved = True
                            break
                
                if not improved:
                    step *= 0.5
                    if np.max(step / (upper - lower)) < 1e-15:
                        break
    
    return best
