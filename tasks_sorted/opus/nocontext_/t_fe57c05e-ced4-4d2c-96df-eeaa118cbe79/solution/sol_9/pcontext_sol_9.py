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
    pop_size = min(4 + int(3 * np.log(dim)), 100)
    if pop_size % 2 != 0:
        pop_size += 1
    
    # Start with multiple restarts of a Nelder-Mead-like + differential evolution hybrid
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    # --- Initial Latin Hypercube Sampling ---
    n_init = min(max(10 * dim, 50), 500)
    init_samples = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_samples[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * (upper[d] - lower[d])
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            break
        f = func(init_samples[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_params = init_samples[i].copy()
    
    if remaining() < 0.05 * max_time:
        return best
    
    # --- Phase 2: CMA-ES ---
    def run_cmaes(x0, sigma0, budget_fraction):
        nonlocal best, best_params
        
        deadline = elapsed() + remaining() * budget_fraction
        
        n = len(x0)
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
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
        
        while elapsed() < deadline and elapsed() < max_time * 0.95:
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
            
            # Evaluate
            fitness = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= max_time * 0.95:
                    return
                fitness[k] = func(arx[k])
                if fitness[k] < best:
                    best = fitness[k]
                    best_params = arx[k].copy()
            
            # Sort
            idx = np.argsort(fitness)
            arx = arx[idx]
            arz = arz[idx]
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            # Update evolution paths
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ (D * zmean))
            
            # Update covariance matrix
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            generation += 1
            eigeneval += lam
            
            if generation % (1 + int(1 / (10 * n * (c1 + cmu)))) == 0:
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
            
            # Check for too small sigma
            if sigma * np.max(D) < 1e-12 * np.max(upper - lower):
                break
    
    # Sort initial samples and pick top ones for CMA-ES restarts
    sorted_idx = np.argsort(init_fitness)
    
    # Run CMA-ES from best initial point
    if best_params is not None:
        sigma0 = 0.3 * np.max(upper - lower)
        run_cmaes(best_params.copy(), sigma0, 0.5)
    
    # --- Phase 3: Restarts with increasing population or from different starting points ---
    restart_count = 0
    while elapsed() < max_time * 0.95:
        restart_count += 1
        
        # Pick a starting point: mix of best known + random
        if restart_count % 3 == 0 or best_params is None:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        else:
            # Perturb best
            perturbation = np.random.randn(dim) * 0.3 * (upper - lower)
            x0 = clip(best_params + perturbation)
        
        sigma0 = (0.1 + 0.4 * np.random.random()) * np.max(upper - lower)
        
        budget = min(0.3, remaining() / max_time)
        if budget < 0.01:
            break
        run_cmaes(x0, sigma0, budget)
    
    # --- Phase 4: Local refinement with Nelder-Mead ---
    if best_params is not None and remaining() > 0.02 * max_time:
        # Simple Nelder-Mead
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex around best
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        scale = 0.05 * (upper - lower)
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += scale[i] if best_params[i] + scale[i] <= upper[i] else -scale[i]
        simplex = clip(simplex)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if elapsed() >= max_time * 0.95:
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if fr < f_simplex[0]:
                # Expand
                xe = clip(centroid + gamma * (xr - centroid))
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[n] = xe
                    f_simplex[n] = fe
                else:
                    simplex[n] = xr
                    f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr
                f_simplex[n] = fr
            else:
                # Contract
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[n] - centroid))
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                
                if fc < min(fr, f_simplex[n]):
                    simplex[n] = xc
                    f_simplex[n] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        if elapsed() >= max_time * 0.95:
                            return best
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_params = simplex[i].copy()
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
