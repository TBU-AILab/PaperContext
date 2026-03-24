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
        init_samples[:, d] = (perms + np.random.uniform(size=n_init)) / n_init
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
    # Sort and pick top candidates
    sorted_idx = np.argsort(init_fitness)
    
    # CMA-ES parameters
    pop_size = 4 + int(3 * np.log(dim))
    pop_size = max(pop_size, 10)
    mu = pop_size // 2
    
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)
    
    # Run CMA-ES from multiple restarts
    n_restarts = max(3, min(10, len(sorted_idx)))
    
    for restart in range(n_restarts):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.90):
            break
        
        # Initialize from a top candidate
        mean = init_samples[sorted_idx[restart % len(sorted_idx)]].copy()
        
        # Initial sigma
        sigma = 0.3 * np.mean(upper - lower)
        if restart > 0:
            sigma *= (0.5 + 0.5 * np.random.random())
        
        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu_val = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        eigeneval = 0
        counteval = 0
        max_iter_per_restart = 1000 + 500 * dim
        
        for gen in range(max_iter_per_restart):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.90):
                break
            
            # Eigendecomposition
            if counteval - eigeneval > pop_size / (c1 + cmu_val) / dim / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_vals, B = np.linalg.eigh(C)
                    D_vals = np.maximum(D_vals, 1e-20)
                    D = np.sqrt(D_vals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)
                    invsqrtC = np.eye(dim)
            else:
                if gen == 0:
                    D = np.ones(dim)
                    B = np.eye(dim)
                    invsqrtC = np.eye(dim)
            
            # Sample population
            arx = np.zeros((pop_size, dim))
            arz = np.zeros((pop_size, dim))
            arfitness = np.zeros(pop_size)
            
            for k in range(pop_size):
                arz[k] = np.random.randn(dim)
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
                
                f = func(arx[k])
                counteval += 1
                arfitness[k] = f
                if f < best:
                    best = f
                    best_params = arx[k].copy()
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / pop_size)) / chiN < 1.4 + 2.0 / (dim + 1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
            
            artmp = (arx[arindex[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (weights[:, None] * artmp).T @ artmp
            
            # Sigma adaptation
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # Check for convergence
            if sigma < 1e-12:
                break
            if sigma * np.max(D) < 1e-12 * np.max(upper - lower):
                break
    
    # --- Phase 3: Nelder-Mead local search from best found ---
    time_left = max_time - (datetime.now() - start).total_seconds()
    if time_left > 0.5 and best_params is not None:
        # Simple Nelder-Mead
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex around best_params
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        scale = 0.05 * (upper - lower)
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += scale[i]
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        for nm_iter in range(5000):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                break
            
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
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_params = simplex[i].copy()
                        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                            break
            
            # Check convergence
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
