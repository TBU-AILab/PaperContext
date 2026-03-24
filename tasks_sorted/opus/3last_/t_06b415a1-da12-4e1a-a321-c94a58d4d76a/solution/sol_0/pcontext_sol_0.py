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
    # Select top candidates
    n_top = min(max(10, dim), n_init // 2)
    top_indices = np.argsort(init_fitness)[:n_top]
    
    # Run multiple restarts of a simplified CMA-ES
    def cma_es_run(x0, sigma0, budget_fraction):
        nonlocal best, best_params
        
        n = dim
        lam = 4 + int(3 * np.log(n))  # population size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        time_limit = timedelta(seconds=max_time * budget_fraction)
        
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return
            
            # Generate offspring
            try:
                if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                else:
                    try:
                        D, B
                    except:
                        D_vals, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D_vals, 1e-20))
            except:
                return
            
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return
                
                z = np.random.randn(n)
                arx[k] = mean + sigma * (B @ (D * z))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = func(arx[k])
                counteval += 1
                
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_params = arx[k].copy()
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (1.0/D * (B.T @ (mean - old_mean)))) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            # CCA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance matrix adaptation
            artmp = (arx[arindex[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Step size update
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Break conditions
            if sigma < 1e-12:
                return
            if counteval > 10000 * dim:
                return
    
    # Run CMA-ES from top candidates
    n_restarts = max(3, n_top)
    budget_per_restart = 0.8 / n_restarts
    
    for i in range(n_restarts):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.90):
            break
        
        if i < len(top_indices):
            x0 = init_points[top_indices[i]].copy()
        else:
            x0 = np.array([np.random.uniform(l, h) for l, h in bounds])
        
        sigma0 = np.mean(upper - lower) * 0.3
        cma_es_run(x0, sigma0, budget_per_restart)
    
    # --- Phase 3: Nelder-Mead from best found ---
    if best_params is not None:
        passed_time = (datetime.now() - start)
        remaining = max_time - passed_time.total_seconds()
        
        if remaining > 0.5:
            # Simple Nelder-Mead
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            n = dim
            
            # Initialize simplex around best
            simplex = np.zeros((n + 1, n))
            simplex[0] = best_params.copy()
            scale = np.maximum(np.abs(best_params) * 0.05, (upper - lower) * 0.01)
            for i in range(n):
                simplex[i + 1] = best_params.copy()
                simplex[i + 1][i] += scale[i]
                simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            
            f_simplex = np.zeros(n + 1)
            for i in range(n + 1):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                f_simplex[i] = func(simplex[i])
                if f_simplex[i] < best:
                    best = f_simplex[i]
                    best_params = simplex[i].copy()
            
            while True:
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                
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
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = func(xc)
                    if fc < best:
                        best = fc
                        best_params = xc.copy()
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = func(simplex[i])
                            if f_simplex[i] < best:
                                best = f_simplex[i]
                                best_params = simplex[i].copy()
                
                # Convergence check
                if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                    break
    
    return best
