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
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(init_pop[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_params = init_pop[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Use multiple restarts with CMA-ES
    
    def cma_es(x0, sigma0, budget_fraction):
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
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        
        sigma = sigma0
        xmean = x0.copy()
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        time_limit = timedelta(seconds=max_time * budget_fraction)
        gen = 0
        
        while True:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return
            
            # Generate offspring
            arx = np.zeros((lam, n))
            arfitness = np.full(lam, float('inf'))
            
            for k in range(lam):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    return
                z = np.random.randn(n)
                arx[k] = xmean + sigma * (B @ (D * z))
                # Clip to bounds
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = func(arx[k])
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_params = arx[k].copy()
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arx[arindex[i]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN < 1.4 + 2/(n + 1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
            
            artmp = np.zeros((mu, n))
            for i in range(mu):
                artmp[i] = (arx[arindex[i]] - xold) / sigma
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(artmp[i], artmp[i])
            
            # Sigma update
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            gen += 1
            if gen % (1 / (c1 + cmu) / n / 10 + 1) < 1:
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
            
            # Break conditions
            if sigma * np.max(D) < 1e-12 * np.max(upper - lower):
                return
            if gen > 500 + 200 * n:
                return
    
    # Sort initial population and pick top candidates for restarts
    sorted_idx = np.argsort(init_fitness)
    
    # Determine how much time we have left
    elapsed = (datetime.now() - start).total_seconds()
    remaining = max_time - elapsed
    
    if remaining <= 0.1:
        return best
    
    # Run CMA-ES from the best found point
    n_restarts = max(1, min(10, int(remaining / 2)))
    budget_per_restart = 0.9 / n_restarts
    
    for r in range(n_restarts):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        
        if r < min(n_restarts, len(sorted_idx)):
            x0 = init_pop[sorted_idx[r]].copy()
        else:
            x0 = init_pop[sorted_idx[0]].copy() + 0.1 * (upper - lower) * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
        
        # Initial sigma: fraction of domain size, decreasing with restarts
        if r == 0:
            sigma0 = 0.3 * np.mean(upper - lower)
        else:
            sigma0 = max(0.01, 0.5 / (r + 1)) * np.mean(upper - lower)
        
        cma_es(x0, sigma0, budget_per_restart)
    
    # --- Phase 3: Local Nelder-Mead refinement around best ---
    if best_params is not None and (datetime.now() - start) < timedelta(seconds=max_time * 0.95):
        # Simple Nelder-Mead
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        scale = 0.01 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += scale[i]
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        for _ in range(1000 * n):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return best
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            elif fr < f_simplex[0]:
                # Expand
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
                # Contract
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
                        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                            return best
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
