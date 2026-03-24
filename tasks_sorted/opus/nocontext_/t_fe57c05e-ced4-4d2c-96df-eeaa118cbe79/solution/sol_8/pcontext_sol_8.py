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
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    
    # Generate LHS-like samples
    init_samples = np.zeros((n_init, dim))
    for d in range(dim):
        perms = np.random.permutation(n_init)
        init_samples[:, d] = (perms + np.random.uniform(0, 1, n_init)) / n_init
        init_samples[:, d] = lower[d] + init_samples[:, d] * (upper[d] - lower[d])
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(init_samples[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_params = init_samples[i].copy()
    
    # --- Phase 2: CMA-ES inspired search ---
    # Use multiple restarts with different population sizes
    
    def cma_es_run(x0, sigma0, time_fraction):
        nonlocal best, best_params
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        
        sigma = sigma0
        mean = x0.copy()
        
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        run_start = datetime.now()
        run_budget = timedelta(seconds=time_fraction)
        gen = 0
        
        while True:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return
            if (datetime.now() - run_start) >= run_budget:
                return
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
            
            # Evaluate
            fitvals = np.zeros(lam)
            for k in range(lam):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    return
                fitvals[k] = func(arx[k])
                if fitvals[k] < best:
                    best = fitvals[k]
                    best_params = arx[k].copy()
            
            # Sort
            arindex = np.argsort(fitvals)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[arindex[k]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance matrix adaptation
            artmp = np.zeros((mu, n))
            for k in range(mu):
                artmp[k] = (arx[arindex[k]] - old_mean) / sigma
            
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Sigma adaptation
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # Update B, D from C
            eigeneval += 1
            if eigeneval >= 1.0 / (c1 + cmu_val) / n / 10 or gen == 0:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    # Reset if eigendecomposition fails
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
                    sigma = sigma0
            
            # Stop if sigma gets too small or too large
            if sigma < 1e-20 or sigma > 1e10:
                return
            
            gen += 1
    
    # Sort initial samples and pick best starting points
    sorted_idx = np.argsort(init_fitness)
    
    # Time remaining
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining <= 0:
        return best
    
    # Multiple CMA-ES restarts from best initial points
    n_restarts = max(1, min(10, int(remaining / 2)))
    time_per_restart = remaining * 0.95 / n_restarts
    
    # Initial sigma based on bounds range
    sigma0 = np.mean(upper - lower) * 0.3
    
    for r in range(n_restarts):
        if (datetime.now() - start).total_seconds() >= max_time * 0.95:
            break
        
        if r < min(5, len(sorted_idx)):
            x0 = init_samples[sorted_idx[r]].copy()
            cur_sigma = sigma0 * (0.5 if r == 0 else 1.0)
        else:
            # Random restart with perturbation from best
            x0 = best_params.copy() + np.random.randn(dim) * sigma0 * 0.5
            x0 = np.clip(x0, lower, upper)
            cur_sigma = sigma0 * 0.5
        
        cma_es_run(x0, cur_sigma, time_per_restart)
    
    # --- Phase 3: Local refinement with Nelder-Mead style simplex ---
    if best_params is not None and (datetime.now() - start).total_seconds() < max_time * 0.90:
        # Simple coordinate-wise refinement
        step = (upper - lower) * 0.01
        improved = True
        while improved:
            if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                break
            improved = False
            for d in range(dim):
                if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                    break
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = func(trial)
                    if f < best:
                        best = f
                        best_params = trial.copy()
                        improved = True
            step *= 0.5
            if np.max(step) < 1e-15:
                break
    
    return best
