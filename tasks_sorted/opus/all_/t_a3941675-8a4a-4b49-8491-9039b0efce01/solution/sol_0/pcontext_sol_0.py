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
    # Use multiple restarts of a simplified CMA-ES
    
    def cma_es_run(x0, sigma0, budget_fraction):
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
        
        mean = x0.copy()
        sigma = sigma0
        
        deadline = start + timedelta(seconds=max_time * budget_fraction)
        
        generation = 0
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95 or datetime.now() >= deadline:
                break
            
            # Sample offspring
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
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            
            # CSA
            diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / 
                    (np.sqrt(n) * (1 + 1/(4*n)))) < 1.4 + 2/(n + 1)
            hsig = float(hsig)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            # Covariance matrix adaptation
            artmp = np.zeros((mu, n))
            for k in range(mu):
                artmp[k] = (arx[idx[k]] - old_mean) / sigma
            
            C = ((1 - c1 - cmu_val) * C + 
                 c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                 cmu_val * sum(weights[k] * np.outer(artmp[k], artmp[k]) for k in range(mu)))
            
            # Step size update
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            generation += 1
            if generation % (1 / (c1 + cmu_val) / n / 10 + 1) < 1:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            # Check convergence
            if sigma * np.max(D) < 1e-12 * np.max(upper - lower):
                break
    
    # Sort initial population and pick top candidates for CMA-ES restarts
    sorted_idx = np.argsort(init_fitness)
    
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.1:
        # First run: use best found point with moderate sigma
        sigma0 = 0.3 * np.max(upper - lower)
        n_restarts = max(1, min(5, int(remaining / 2)))
        
        for r in range(n_restarts):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95:
                break
            
            remaining = max_time - elapsed
            budget_frac = min(elapsed / max_time + remaining / (max_time * (n_restarts - r + 0.01)), 0.98)
            budget_frac = min(budget_frac, elapsed / max_time + (remaining * 0.8) / (max_time * max(1, n_restarts - r)))
            # Simpler: allocate remaining time equally
            per_restart_time = remaining / max(1, n_restarts - r)
            budget_frac = (elapsed + per_restart_time) / max_time
            
            if r == 0:
                x0 = best_params.copy()
                sig = sigma0
            else:
                # Pick from top candidates or random perturbation
                if r < len(sorted_idx):
                    x0 = init_pop[sorted_idx[r % min(5, len(sorted_idx))]].copy()
                else:
                    x0 = lower + np.random.random(dim) * (upper - lower)
                sig = sigma0 * (0.5 + 0.5 * np.random.random())
                # Also try with larger sigma for diversity
                if r >= 2:
                    sig = sigma0 * (1 + np.random.random())
            
            cma_es_run(x0, sig, budget_frac)
    
    # --- Phase 3: Local Nelder-Mead refinement ---
    elapsed = (datetime.now() - start).total_seconds()
    if elapsed < max_time * 0.90 and best_params is not None:
        # Simple Nelder-Mead
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex around best point
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        scale = 0.05 * (upper - lower)
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += scale[i]
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95:
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95:
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
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    break
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
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    break
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
                        elapsed = (datetime.now() - start).total_seconds()
                        if elapsed >= max_time * 0.95:
                            return best
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_params = simplex[i].copy()
            
            # Check convergence
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
