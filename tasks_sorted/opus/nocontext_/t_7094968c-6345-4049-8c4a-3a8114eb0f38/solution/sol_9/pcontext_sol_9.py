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
    
    # --- Phase 2: CMA-ES inspired search from best points ---
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
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        sigma = sigma0
        mean = x0.copy()
        
        run_start = datetime.now()
        run_budget = timedelta(seconds=time_fraction)
        
        generation = 0
        while True:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return
            if (datetime.now() - run_start) >= run_budget:
                return
            
            # Generate offspring
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                z = np.random.randn(n)
                arx[k] = mean + sigma * (B @ (D * z))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = func(arx[k])
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_params = arx[k].copy()
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            artmp = (arx[arindex[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            eigeneval += 1
            if eigeneval >= 1.0 / (c1 + cmu_val) / n / 10 or generation == 0:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    # Reset
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            generation += 1
            
            # Check for convergence
            if sigma * np.max(D) < 1e-12 * np.max(upper - lower):
                return
    
    # Sort initial samples
    sorted_idx = np.argsort(init_fitness)
    
    # Time remaining
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining <= 0.1:
        return best
    
    # Multiple CMA-ES restarts from best initial points
    n_restarts = max(1, min(10, int(remaining / 0.5)))
    time_per_restart = remaining * 0.9 / n_restarts
    
    sigma0 = 0.3 * np.max(upper - lower)
    
    for i in range(n_restarts):
        if (datetime.now() - start).total_seconds() >= max_time * 0.95:
            break
        
        if i < len(sorted_idx):
            x0 = init_samples[sorted_idx[i]].copy()
        else:
            # Random restart near best
            x0 = best_params + np.random.randn(dim) * sigma0 * 0.1 * (0.5 ** (i // len(sorted_idx)))
            x0 = np.clip(x0, lower, upper)
        
        # Decrease sigma for later restarts, increase for diversity
        if i == 0:
            s = sigma0
        elif i < 3:
            s = sigma0 * 0.5
        else:
            s = sigma0 * (0.5 + 0.5 * np.random.rand())
        
        cma_es_run(x0, s, time_per_restart)
    
    # --- Phase 3: Local refinement with Nelder-Mead ---
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.5 and best_params is not None:
        # Simple Nelder-Mead
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex around best
        scale = 0.01 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = best
        
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += scale[i]
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            f_simplex[i + 1] = func(simplex[i + 1])
            if f_simplex[i + 1] < best:
                best = f_simplex[i + 1]
                best_params = simplex[i + 1].copy()
        
        while (datetime.now() - start).total_seconds() < max_time * 0.98:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best:
                best = f_simplex[0]
                best_params = simplex[0].copy()
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            elif fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            else:
                # Contraction
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), lower, upper)
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_params = simplex[i].copy()
            
            # Check convergence
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
