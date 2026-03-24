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
            best_x = init_pop[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Use multiple restarts with CMA-ES
    
    def cma_es(x0, sigma0, budget_fraction):
        nonlocal best, best_x
        
        n = dim
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
        xmean = x0.copy()
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        generation = 0
        target_end = (datetime.now() - start).total_seconds() + budget_fraction * max_time
        
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= min(target_end, max_time * 0.98):
                break
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = xmean + sigma * (B @ (D * arz[k]))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
            
            # Evaluate
            fitvals = np.zeros(lam)
            for k in range(lam):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.98:
                    return
                fitvals[k] = func(arx[k])
                if fitvals[k] < best:
                    best = fitvals[k]
                    best_x = arx[k].copy()
            
            # Sort
            arindex = np.argsort(fitvals)
            
            # Recombination
            xold = xmean.copy()
            xmean = np.zeros(n)
            for k in range(mu):
                xmean += weights[k] * arx[arindex[k]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
            
            # Covariance matrix adaptation
            artmp = np.zeros((mu, n))
            for k in range(mu):
                artmp[k] = (arx[arindex[k]] - xold) / sigma
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Adapt step size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu) / n / 10 or generation == 0:
                eigeneval = 0
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
            
            generation += 1
            
            # Check convergence
            if sigma * np.max(D) < 1e-12:
                break
            if generation > 0 and fitvals[arindex[0]] == fitvals[arindex[min(lam - 1, int(0.7 * lam))]]:
                sigma *= np.exp(0.2 + cs / damps)
    
    # Sort initial population and pick top candidates for CMA-ES restarts
    sorted_idx = np.argsort(init_fitness)
    
    # Estimate initial sigma
    range_vals = upper - lower
    sigma0 = np.mean(range_vals) / 4.0
    
    # First CMA-ES run from the best point with larger budget
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.1:
        cma_es(init_pop[sorted_idx[0]], sigma0, 0.5 * remaining / max_time)
    
    # Restart CMA-ES from diverse good points
    n_restarts = min(5, len(sorted_idx))
    restart_idx = 0
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time - elapsed
        if remaining < 0.1 or restart_idx >= n_restarts:
            break
        
        # Pick next good candidate, but also try with smaller sigma
        idx = sorted_idx[min(restart_idx, len(sorted_idx) - 1)]
        s = sigma0 * (0.5 ** restart_idx) if restart_idx > 0 else sigma0 * 0.5
        
        budget = remaining * 0.4
        cma_es(init_pop[idx], s, budget / max_time)
        restart_idx += 1
    
    # --- Phase 3: Local refinement with Nelder-Mead ---
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.1 and best_x is not None:
        # Simple Nelder-Mead
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex around best_x
        scale = np.minimum(np.abs(best_x - lower), np.abs(upper - best_x))
        scale = np.maximum(scale, (upper - lower) * 0.001)
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1, i] += scale[i] * 0.05
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.98:
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_x = simplex[i].copy()
        
        for _ in range(10000):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.98:
                break
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best:
                best = f_simplex[0]
                best_x = simplex[0].copy()
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_x = xr.copy()
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            elif fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.98:
                    break
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_x = xe.copy()
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
                if elapsed >= max_time * 0.98:
                    break
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_x = xc.copy()
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        elapsed = (datetime.now() - start).total_seconds()
                        if elapsed >= max_time * 0.98:
                            return best
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_x = simplex[i].copy()
            
            # Check convergence
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
