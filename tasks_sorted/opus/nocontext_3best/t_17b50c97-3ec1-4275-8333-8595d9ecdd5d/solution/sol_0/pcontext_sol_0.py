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
            best_params = init_points[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Use multiple restarts with CMA-ES
    
    def cmaes_search(x0, sigma0, budget_fraction):
        nonlocal best, best_params
        
        target_end = start + timedelta(seconds=max_time * budget_fraction)
        n = len(x0)
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
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
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        sigma = sigma0
        
        mean = x0.copy()
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        generation = 0
        
        while True:
            passed_time = datetime.now()
            if passed_time >= target_end or passed_time >= start + timedelta(seconds=max_time * 0.98):
                return
            
            # Sample lambda offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
            
            # Evaluate
            fitnesses = np.zeros(lam)
            for k in range(lam):
                passed_time = datetime.now()
                if passed_time >= target_end or passed_time >= start + timedelta(seconds=max_time * 0.98):
                    return
                fitnesses[k] = func(arx[k])
                if fitnesses[k] < best:
                    best = fitnesses[k]
                    best_params = arx[k].copy()
            
            # Sort by fitness
            arindex = np.argsort(fitnesses)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[arindex[k]]
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            artmp = np.zeros((mu, n))
            for k in range(mu):
                artmp[k] = (arx[arindex[k]] - old_mean) / sigma
            
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for k in range(mu):
                C += cmu_val * weights[k] * np.outer(artmp[k], artmp[k])
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            generation += 1
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu_val) / n / 10:
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
            
            # Check for convergence
            if sigma * np.max(D) < 1e-12 * np.max(upper - lower):
                return
    
    # Sort initial points and pick top candidates for CMA-ES restarts
    sorted_idx = np.argsort(init_fitness)
    
    # Determine how many restarts we can do
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining <= 0:
        return best
    
    # First CMA-ES run from best point with moderate sigma
    sigma0 = 0.25 * np.max(upper - lower)
    
    n_restarts = max(1, min(5, int(remaining / 2)))
    
    for r in range(n_restarts):
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time - elapsed
        if remaining < 0.5:
            break
        
        budget_end = min(1.0, (elapsed + remaining / max(1, n_restarts - r)) / max_time)
        
        if r == 0:
            x0 = best_params.copy()
            s0 = sigma0
        else:
            # Pick from top candidates or random perturbation
            if r < len(sorted_idx):
                idx = sorted_idx[min(r, len(sorted_idx) - 1)]
                x0 = init_points[idx].copy()
            else:
                x0 = best_params + np.random.randn(dim) * sigma0 * 0.5
                x0 = np.clip(x0, lower, upper)
            s0 = sigma0 * (0.5 ** (r // 2))
        
        cmaes_search(x0, s0, budget_end)
    
    # --- Phase 3: Local Nelder-Mead style refinement ---
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.5 and best_params is not None:
        # Simple pattern search refinement
        step = 1e-3 * np.max(upper - lower)
        x = best_params.copy()
        
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.99):
                break
            
            improved = False
            for d in range(dim):
                for sign in [1, -1]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step, lower[d], upper[d])
                    f = func(trial)
                    if f < best:
                        best = f
                        best_params = trial.copy()
                        x = trial.copy()
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step *= 0.5
                if step < 1e-15 * np.max(upper - lower):
                    break
    
    return best
