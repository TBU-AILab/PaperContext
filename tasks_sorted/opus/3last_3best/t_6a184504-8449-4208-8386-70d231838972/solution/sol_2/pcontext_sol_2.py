#Looking at the results, the best algorithm (35.49) uses IPOP-CMA-ES. I'll improve it by:
#
#1. **Adding a local search (Nelder-Mead simplex)** to refine solutions after CMA-ES converges
#2. **Better restart diversity** - mix BIPOP strategy (small and large populations)
#3. **Differential Evolution phase** for better global exploration
#4. **Tighter sigma adaptation** and better boundary handling (mirror instead of clip)
#5. **More efficient time allocation**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def mirror_bound(x):
        # Mirror boundary handling - better than clipping for CMA-ES
        for i in range(len(x)):
            while x[i] < lower[i] or x[i] > upper[i]:
                if x[i] < lower[i]:
                    x[i] = 2 * lower[i] - x[i]
                if x[i] > upper[i]:
                    x[i] = 2 * upper[i] - x[i]
        return x
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_x, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(20 * dim, 200), 800)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i] + lower[i]
    
    init_fits = []
    for i in range(n_init):
        if remaining() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fits.append((f, i))
    
    init_fits.sort()
    top_k = min(15, len(init_fits))
    top_points = [init_points[init_fits[i][1]].copy() for i in range(top_k)]
    top_fits = [init_fits[i][0] for i in range(top_k)]
    
    # --- Nelder-Mead local search ---
    def nelder_mead(x0, max_iters=None, max_seconds=None):
        nonlocal best, best_x
        nm_start = elapsed()
        n = len(x0)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = eval_func(x0)
        
        scale = np.maximum(np.abs(x0) * 0.05, ranges * 0.02)
        for i in range(n):
            if remaining() <= 0:
                return
            point = x0.copy()
            point[i] += scale[i]
            point = clip(point)
            simplex[i + 1] = point
            f_simplex[i + 1] = eval_func(point)
        
        iters = 0
        while True:
            if remaining() <= 0:
                return
            if max_iters and iters >= max_iters:
                return
            if max_seconds and (elapsed() - nm_start) > max_seconds:
                return
            
            # Sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if remaining() <= 0:
                                return
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = eval_func(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if remaining() <= 0:
                                return
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
            
            iters += 1
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-14 * (1 + abs(f_simplex[0])):
                return
            if np.max(np.std(simplex, axis=0)) < 1e-14:
                return
    
    # --- CMA-ES ---
    def run_cmaes(x0, init_sigma, pop_factor=1):
        nonlocal best, best_x
        
        if remaining() <= 0:
            return
        
        cma_start = elapsed()
        
        sigma = init_sigma
        mean = x0.copy()
        
        base_lam = 4 + int(3 * np.log(dim))
        lam = max(base_lam * pop_factor, 6)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu_p = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        use_sep = dim > 80
        
        if use_sep:
            diagC = np.ones(dim)
        else:
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigeneval = 0
        
        counteval = 0
        gen = 0
        flat_count = 0
        prev_median = float('inf')
        best_gen_fit = float('inf')
        no_improve_count = 0
        
        history_best = []
        
        while True:
            if remaining() <= 0:
                return
            
            if not use_sep:
                if counteval - eigeneval > lam / (c1 + cmu_p + 1e-20) / dim / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        return
            
            arxs = []
            fitnesses = []
            
            for k in range(lam):
                if remaining() <= 0:
                    return
                z = np.random.randn(dim)
                if use_sep:
                    x = mean + sigma * (np.sqrt(np.maximum(diagC, 1e-20)) * z)
                else:
                    x = mean + sigma * (B @ (D * z))
                x = mirror_bound(x)
                x = clip(x)
                f = eval_func(x)
                counteval += 1
                arxs.append(x)
                fitnesses.append(f)
            
            idx = np.argsort(fitnesses)
            sorted_arxs = [arxs[i] for i in idx]
            sorted_fits = [fitnesses[i] for i in idx]
            
            median_fit = sorted_fits[lam // 2]
            
            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * sorted_arxs[i]
            
            diff = mean - old_mean
            
            if use_sep:
                inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(diagC, 1e-20))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_diag * diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff / sigma
            
            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2/(dim + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            artmp = np.zeros((mu, dim))
            for i in range(mu):
                artmp[i] = (sorted_arxs[i] - old_mean) / sigma
            
            if use_sep:
                diagC = (1 - c1 - cmu_p) * diagC + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC)
                for i in range(mu):
                    diagC += cmu_p * weights[i] * artmp[i]**2
                diagC = np.maximum(diagC, 1e-20)
            else:
                C = (1 - c1 - cmu_p) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_p * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 2)
            
            gen += 1
            
            # Track improvement
            if sorted_fits[0] < best_gen_fit - 1e-12 * (abs(best_gen_fit) + 1e-12):
                best_gen_fit = sorted_fits[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if abs(median_fit - prev_median) < 1e-12 * (abs(median_fit) + 1e-12):
                flat_count += 1
            else:
                flat_count = 0
            prev_median = median_fit
            
            history_best.append(sorted_fits[0])
            
            # Stagnation checks
            if not use_sep:
                if len(D) > 0 and D.max() > 1e7 * D.min():
                    return
            
            if flat_count >= 10 + dim // 3:
                return
            if no_improve_count >= 30 + 10 * dim:
                return
            if sigma < 1e-16:
                return
            
            # Check if fitness range in population is too small
            fit_range = sorted_fits[-1] - sorted_fits[0]
            if fit_range < 1e-14 * (abs(sorted_fits[0]) + 1e-14) and gen > 5:
                return
    
    # --- Differential Evolution phase ---
    def run_de(budget_seconds):
        nonlocal best, best_x
        
        if remaining() <= 0:
            return
        
        de_start = elapsed()
        
        pop_size = min(max(10 * dim, 40), 200)
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = np.zeros((pop_size, dim))
        pop_f = np.full(pop_size, float('inf'))
        
        # Use some top points
        for i in range(min(len(top_points), pop_size // 2)):
            pop[i] = top_points[i].copy()
        if best_x is not None:
            pop[0] = best_x.copy()
        
        for i in range(min(len(top_points), pop_size // 2), pop_size):
            pop[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        for i in range(pop_size):
            if remaining() <= 0:
                return
            pop_f[i] = eval_func(pop[i])
        
        while True:
            if remaining() <= 0 or (elapsed() - de_start) > budget_seconds:
                return
            
            for i in range(pop_size):
                if remaining() <= 0 or
