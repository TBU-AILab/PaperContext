#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **CMA-ES implementation has issues** - the eigendecomposition frequency check is buggy, and the restart strategy could be better
#2. **Time budget allocation is suboptimal** - too much time on initialization, not enough on CMA-ES
#3. **Population size could be adaptive** - use IPOP-CMA-ES (increasing population) restart strategy
#4. **Boundary handling** - use mirroring instead of clipping for better CMA-ES behavior
#
#Let me implement a cleaner, more robust version:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.98):
        return elapsed() < max_time * fraction
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Phase 1: Sobol-like initialization via Latin Hypercube
    n_init = min(max(30 * dim, 200), 800)
    
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.90):
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    # Phase 2: IPOP-CMA-ES with restarts
    def cma_es(x0, sigma0, lam=None):
        nonlocal best, best_params
        n = dim
        if lam is None:
            lam = 4 + int(3 * np.log(n))
        lam = max(lam, 4)
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
        
        sigma = sigma0
        xmean = x0.copy()
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        counteval = 0
        eigeneval = 0
        gen = 0
        stag_count = 0
        prev_best_gen = float('inf')
        
        while time_ok(0.97):
            arx = np.zeros((lam, n))
            arfitness = np.full(lam, float('inf'))
            
            for k in range(lam):
                if not time_ok(0.97):
                    return
                z = np.random.randn(n)
                arx[k] = xmean + sigma * (B @ (D * z))
                # Mirror boundary handling
                for dd in range(n):
                    while arx[k, dd] < lower[dd] or arx[k, dd] > upper[dd]:
                        if arx[k, dd] < lower[dd]:
                            arx[k, dd] = 2 * lower[dd] - arx[k, dd]
                        if arx[k, dd] > upper[dd]:
                            arx[k, dd] = 2 * upper[dd] - arx[k, dd]
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            
            if arfitness[arindex[0]] < prev_best_gen:
                prev_best_gen = arfitness[arindex[0]]
                stag_count = 0
            else:
                stag_count += 1
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (xmean - xold) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - xold) / sigma
            
            C = ((1 - c1 - cmu_val) * C 
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu_val * (weights[:, None] * artmp).T @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            counteval += lam
            
            if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                return
            if stag_count > 10 + 30 * n / lam:
                return
            if np.max(D) / np.min(D) > 1e7:
                return
    
    sorted_idx = np.argsort(init_fitness)
    
    # IPOP: increase population on each restart
    base_lam = 4 + int(3 * np.log(dim))
    restart = 0
    
    while time_ok(0.92):
        if restart < min(5, len(sorted_idx)):
            x0 = init_pop[sorted_idx[restart]].copy()
        else:
            x0 = best_params.copy() + 0.2 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
        
        lam_r = int(base_lam * (2 ** (restart // 2)))
        lam_r = min(lam_r, 256)
        
        sigma0 = 0.3 * np.mean(ranges) if restart < 3 else 0.15 * np.mean(ranges)
        
        cma_es(x0, sigma0, lam_r)
        restart += 1
    
    # Phase 3: Final local search - coordinate descent
    if best_params is not None:
        x = best_params.copy()
        step = 0.001 * ranges
        while time_ok(0.995):
            improved = False
            for d in range(dim):
                if not time_ok(0.995):
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = eval_func(xn)
                    if fn < eval_func(x):
                        x = xn.copy()
                        improved = True
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break
    
    return best
