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
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * (upper[d] - lower[d])
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        f = func(init_pop[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_params = init_pop[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Use multiple restarts with CMA-ES
    
    def cmaes_search(x0, sigma0, budget_fraction):
        nonlocal best, best_params
        target_end = start + timedelta(seconds=max_time * budget_fraction)
        
        n = len(x0)
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        
        generation = 0
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95 or datetime.now() >= target_end:
                break
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
            
            # Evaluate
            fitnesses = np.zeros(lam)
            for k in range(lam):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    return
                fitnesses[k] = func(arx[k])
                if fitnesses[k] < best:
                    best = fitnesses[k]
                    best_params = arx[k].copy()
            
            # Sort by fitness
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            # Update evolution paths
            mean_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma
            
            # Update covariance matrix
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B, D from C
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu) / n / 10:
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
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            generation += 1
    
    # Sort initial population by fitness, pick top candidates for CMA-ES restarts
    sorted_idx = np.argsort(init_fitness)
    
    # Determine how much time is left
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining <= 0.1:
        return best
    
    # Run CMA-ES from best solutions with restarts
    n_restarts = max(3, min(10, int(remaining / 2)))
    
    for r in range(n_restarts):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.90:
            break
        
        # Pick starting point
        if r < len(sorted_idx):
            x0 = init_pop[sorted_idx[r]].copy()
        else:
            # Random restart
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        # Adaptive sigma based on restart number
        sigma0 = 0.3 * np.mean(upper - lower) * (0.5 ** (r // 3))
        
        budget_frac = min(elapsed / max_time + (1.0 - elapsed / max_time) / max(1, n_restarts - r), 0.95)
        
        cmaes_search(x0, sigma0, budget_frac)
    
    # --- Phase 3: Local refinement with Nelder-Mead style ---
    elapsed = (datetime.now() - start).total_seconds()
    if best_params is not None and elapsed < max_time * 0.90:
        # Simple coordinate descent refinement
        step = 0.01 * (upper - lower)
        improved = True
        while improved:
            improved = False
            for d in range(dim):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    return best
                
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = func(trial)
                    if f < best:
                        best = f
                        best_params = trial.copy()
                        improved = True
            
            step *= 0.5
            if np.max(step / (upper - lower)) < 1e-10:
                break
    
    return best
