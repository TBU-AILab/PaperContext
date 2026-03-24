#Looking at the previous algorithm, I identify several issues:
#
#1. **The coordinate descent phase calls `eval_func(x)` redundantly** - comparing against a re-evaluation instead of cached value
#2. **IPOP population doubling is too aggressive** - wastes evaluations on large populations
#3. **Restart strategy is weak** - should mix global exploration with local refinement around best
#4. **CMA-ES stagnation detection could be improved**
#5. **Need better diversity in restarts** - try multiple starting points with different sigma values
#
#Let me implement a more refined version with better time management and a hybrid approach combining CMA-ES with differential evolution-style perturbations:
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
    center = (lower + upper) / 2.0
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.98):
        return elapsed() < max_time * fraction
    
    def eval_func(x):
        nonlocal best, best_params, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Phase 1: Latin Hypercube Sampling initialization
    n_init = min(max(20 * dim, 100), 500)
    
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.85):
            n_init = i
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    sorted_idx = np.argsort(init_fitness[:n_init])
    
    # Keep a pool of good solutions for restarts
    pool_size = min(20, n_init)
    pool = init_pop[sorted_idx[:pool_size]].copy()
    pool_fit = init_fitness[sorted_idx[:pool_size]].copy()
    
    def update_pool(x, f):
        nonlocal pool, pool_fit
        worst_idx = np.argmax(pool_fit)
        if f < pool_fit[worst_idx]:
            pool[worst_idx] = x.copy()
            pool_fit[worst_idx] = f
    
    # Phase 2: CMA-ES with restarts
    def cma_es(x0, sigma0, lam=None, max_gen=None):
        nonlocal best, best_params
        n = dim
        if lam is None:
            lam = 4 + int(3 * np.log(n))
        lam = max(lam, 6)
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
        
        eigeneval = 0
        counteval = 0
        gen = 0
        stag_count = 0
        best_in_run = float('inf')
        flat_count = 0
        
        while time_ok(0.96):
            if max_gen is not None and gen >= max_gen:
                return
            
            arx = np.zeros((lam, n))
            arfitness = np.full(lam, float('inf'))
            
            for k in range(lam):
                if not time_ok(0.96):
                    return
                z = np.random.randn(n)
                arx[k] = xmean + sigma * (B @ (D * z))
                # Bounce boundary handling
                for dd in range(n):
                    lo, hi = lower[dd], upper[dd]
                    val = arx[k, dd]
                    if val < lo or val > hi:
                        rng = hi - lo
                        if rng > 0:
                            val = val - lo
                            val = val % (2 * rng)
                            if val < 0:
                                val += 2 * rng
                            if val > rng:
                                val = 2 * rng - val
                            arx[k, dd] = val + lo
                        else:
                            arx[k, dd] = lo
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            
            # Update pool
            for idx in arindex[:3]:
                update_pool(arx[idx], arfitness[idx])
            
            gen_best = arfitness[arindex[0]]
            if gen_best < best_in_run - 1e-10 * abs(best_in_run):
                best_in_run = gen_best
                stag_count = 0
            else:
                stag_count += 1
            
            # Check flat fitness
            if arfitness[arindex[0]] == arfitness[arindex[min(lam-1, mu)]]:
                flat_count += 1
            else:
                flat_count = 0
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (xmean - xold) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            
            norm_ps = np.linalg.norm(ps)
            hsig = float(norm_ps / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - xold) / sigma
            
            C = ((1 - c1 - cmu_val) * C 
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu_val * (weights[:, None] * artmp).T @ artmp)
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            gen += 1
            
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-20) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return
            
            # Termination criteria
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                return
            if stag_count > 20 + 30 * n / lam:
                return
            if np.max(D) > 1e7 * np.min(D):
                return
            if flat_count > 5:
                return
    
    base_lam = 4 + int(3 * np.log(dim))
    restart = 0
    
    while time_ok(0.90):
        if restart < min(3, len(sorted_idx)):
            x0 = init_pop[sorted_idx[restart]].copy()
            sigma0 = 0.25 * np.mean(ranges)
            lam_r = base_lam
        elif restart < 6:
            # Local restart from best with small sigma
            x0 = best_params.copy() + 0.05 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.1 * np.mean(ranges)
            lam_r = base_lam
        elif restart % 3 == 0:
            # Global restart with larger population
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.4 * np.mean(ranges)
            lam_r = min(base_lam * 2, 128)
        elif restart % 3 == 1:
            # Restart from pool with DE-like perturbation
            idxs = np.random.choice(len(pool), 3, replace=False)
            x0 = pool[idxs[0]] + 0.8 * (pool[idxs[1]] - pool[idxs[2]])
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.15 * np.mean(ranges)
            lam_r = base_lam
        else:
            x0 = best_params.copy() + 0.1 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.05 * np.mean(ranges)
            lam_r = base_lam
        
        cma_es(x0, sigma0, lam_r)
        restart += 1
    
    # Phase 3: Nelder-Mead simplex from best
    if best_params is not None and time_ok(0.92):
        n = dim
        alpha, gamma_nm, rho, shrink = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex_f = np.full(n + 1, float('inf'))
        simplex[0] = best_params.copy()
        simplex_f[0] = best
        
        step_size = 0.01 * ranges
        for i in range(n):
            if not time_ok(0.98):
                break
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] = np.clip(simplex[i+1][i] + step_size[i], lower[i], upper[i])
            simplex_f[i+1] = eval_func(simplex[i+1])
        
        for _ in range(5000):
            if not time_ok(0.995):
                break
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = eval_func(xr)
            
            if fr < simplex_f[0]:
                xe = centroid + gamma_nm * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lower, upper)
                fc = eval_func(xc)
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    for i in range(1, n + 1):
                        if not time_ok(0.995):
                            break
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        simplex_f[i] = eval_func(simplex[i])
            
            vol = np.max(np.abs(simplex - simplex[0]))
            if vol < 1e-15 * np.max(ranges):
                break
    
    return best
