#Looking at the previous algorithms, I can identify several improvements:
#
#1. **Coordinate descent wastes evaluations** by re-evaluating `x` every iteration
#2. **Population doubling is too aggressive** - wastes budget on huge populations
#3. **Need better balance between exploration and exploitation**
#4. **Add Differential Evolution as a complementary strategy**
#5. **Improve local search with pattern search / golden section**
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
    
    # Phase 1: Latin Hypercube Sampling
    n_init = min(max(20 * dim, 100), 600)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    actual_init = 0
    for i in range(n_init):
        if not time_ok(0.80):
            break
        init_fitness[i] = eval_func(init_pop[i])
        actual_init = i + 1
    
    sorted_idx = np.argsort(init_fitness[:actual_init])
    
    # Maintain archive of good solutions
    archive_size = min(30, actual_init)
    archive = init_pop[sorted_idx[:archive_size]].copy()
    archive_fit = init_fitness[sorted_idx[:archive_size]].copy()
    
    def update_archive(x, f):
        nonlocal archive, archive_fit
        worst_idx = np.argmax(archive_fit)
        if f < archive_fit[worst_idx]:
            archive[worst_idx] = x.copy()
            archive_fit[worst_idx] = f
    
    # Phase 2: CMA-ES with BIPOP-like restarts
    def cma_es(x0, sigma0, lam=None, max_evals=None):
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
        hist_best = []
        
        while time_ok(0.96):
            if max_evals is not None and counteval >= max_evals:
                return
            
            arx = np.zeros((lam, n))
            arfitness = np.full(lam, float('inf'))
            
            for k in range(lam):
                if not time_ok(0.96):
                    return
                z = np.random.randn(n)
                arx[k] = xmean + sigma * (B @ (D * z))
                # Mirror boundary
                for dd in range(n):
                    lo, hi = lower[dd], upper[dd]
                    while arx[k, dd] < lo or arx[k, dd] > hi:
                        if arx[k, dd] < lo:
                            arx[k, dd] = 2*lo - arx[k, dd]
                        if arx[k, dd] > hi:
                            arx[k, dd] = 2*hi - arx[k, dd]
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            
            for idx in arindex[:min(3, lam)]:
                update_archive(arx[idx], arfitness[idx])
            
            gen_best = arfitness[arindex[0]]
            if gen_best < best_in_run - 1e-10 * (abs(best_in_run) + 1e-20):
                best_in_run = gen_best
                stag_count = 0
            else:
                stag_count += 1
            
            hist_best.append(gen_best)
            
            if len(hist_best) > 20:
                recent = hist_best[-20:]
                if max(recent) - min(recent) < 1e-12 * (abs(min(recent)) + 1e-20):
                    flat_count += 1
                else:
                    flat_count = 0
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (xmean - xold) / (sigma + 1e-30)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            
            norm_ps = np.linalg.norm(ps)
            hsig = float(norm_ps / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - xold) / (sigma + 1e-30)
            
            C = ((1 - c1 - cmu_val) * C 
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu_val * (weights[:, None] * artmp).T @ artmp)
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            gen += 1
            
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
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
            if stag_count > 15 + 30 * n / lam:
                return
            if np.max(D) > 1e7 * np.min(D):
                return
            if flat_count > 3:
                return
    
    base_lam = 4 + int(3 * np.log(dim))
    restart = 0
    large_restart = 0
    
    while time_ok(0.88):
        if restart < min(5, actual_init):
            x0 = init_pop[sorted_idx[restart]].copy()
            sigma0 = 0.3 * np.mean(ranges)
            lam_r = base_lam
        elif restart % 3 == 0:
            # IPOP: large population global
            large_restart += 1
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.4 * np.mean(ranges)
            lam_r = min(base_lam * (2 ** large_restart), 256)
        elif restart % 3 == 1:
            # Small local from best
            x0 = best_params.copy() + 0.02 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.05 * np.mean(ranges)
            lam_r = base_lam
        else:
            # DE-like from archive
            idxs = np.random.choice(len(archive), min(3, len(archive)), replace=False)
            if len(idxs) == 3:
                x0 = archive[idxs[0]] + 0.8 * (archive[idxs[1]] - archive[idxs[2]])
            else:
                x0 = best_params.copy() + 0.1 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.15 * np.mean(ranges)
            lam_r = base_lam
        
        cma_es(x0, sigma0, lam_r)
        restart += 1
    
    # Phase 3: Nelder-Mead from best
    if best_params is not None and time_ok(0.90):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex_f = np.full(n + 1, float('inf'))
        simplex[0] = best_params.copy()
        simplex_f[0] = best
        
        step = 0.005 * ranges
        for i in range(n):
            if not time_ok(0.97):
                break
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] = np.clip(simplex[i+1][i] + step[i], lower[i], upper[i])
            simplex_f[i+1] = eval_func(simplex[i+1])
        
        alpha_nm, gamma_nm, rho_nm, shrink_nm = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(10000):
            if not time_ok(0.995):
                break
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha_nm * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = eval_func(xr)
            
            if fr < simplex_f[0]:
                xe = centroid + gamma_nm * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                xc = centroid + rho_nm * ((simplex[-1] if fr >= simplex_f[-1] else xr) - centroid)
                xc = np.clip(xc, lower, upper)
                fc = eval_func(xc)
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1], simplex_f[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        if not time_ok(0.995):
                            break
                        simplex[i] = simplex[0] + shrink_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        simplex_f[i] = eval_func(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15 * np.max(ranges):
                break
    
    return best
