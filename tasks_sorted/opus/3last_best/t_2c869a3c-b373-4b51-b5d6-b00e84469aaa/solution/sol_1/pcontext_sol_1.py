#Looking at the current CMA-ES implementation, I can identify several areas for improvement:
#
#1. **Better restart strategy** with multiple independent restarts (IPOP-CMA-ES style with increasing population)
#2. **Diagonal CMA-ES for high dimensions** to avoid expensive eigendecompositions
#3. **Better initialization** with more diverse sampling
#4. **Local search refinement** (Nelder-Mead style) on the best found solution
#5. **Boundary handling** via mirror/resample instead of clipping
#
#Here's the improved version:
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
    
    def time_left():
        return max_time * 0.95 - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x_clipped = np.clip(x, lower, upper)
        f = func(x_clipped)
        if f < best:
            best = f
            best_x = x_clipped.copy()
        return f
    
    def mirror_bounds(x):
        """Mirror boundary handling"""
        xn = x.copy()
        for i in range(dim):
            while xn[i] < lower[i] or xn[i] > upper[i]:
                if xn[i] < lower[i]:
                    xn[i] = 2 * lower[i] - xn[i]
                if xn[i] > upper[i]:
                    xn[i] = 2 * upper[i] - xn[i]
        return np.clip(xn, lower, upper)
    
    # --- Phase 1: Sobol-like initialization with LHS ---
    n_init = min(max(30 * dim, 200), 1000)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    # Also add center and corners
    center = (lower + upper) / 2.0
    
    init_fitnesses = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_f(init_points[i])
        init_fitnesses.append((f, i))
    
    eval_f(center)
    
    # Sort and pick top candidates for multi-start
    init_fitnesses.sort()
    top_k = min(10, len(init_fitnesses))
    starting_points = [init_points[init_fitnesses[i][1]].copy() for i in range(top_k)]
    
    # --- Phase 2: IPOP-CMA-ES with restarts ---
    base_pop_size = 4 + int(3 * np.log(dim))
    base_pop_size = max(base_pop_size, 10)
    restart_count = 0
    pop_multiplier = 1
    
    while time_left() > 0.5:
        # Choose starting point
        if restart_count < len(starting_points):
            x0 = starting_points[restart_count].copy()
        elif np.random.random() < 0.3:
            x0 = best_x.copy() if best_x is not None else center.copy()
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        # Increase population on restarts (IPOP)
        if restart_count > len(starting_points):
            pop_multiplier = min(pop_multiplier * 2, 16)
        
        pop_size = min(base_pop_size * pop_multiplier, 256)
        mu = pop_size // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        use_sep = dim > 100  # Separable CMA for high dim
        
        mean = x0.copy()
        sigma = np.mean(ranges) / 4.0
        if restart_count > 0 and np.random.random() < 0.5:
            sigma = np.mean(ranges) / (2.0 + restart_count)
        
        if use_sep:
            diag_C = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
        else:
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            eigen_update_freq = max(1, int(1 / (c1 + c_mu_val) / dim / 10))
            eigenvalues = np.ones(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
        
        generation = 0
        stagnation_count = 0
        prev_best_local = float('inf')
        best_local = float('inf')
        flat_count = 0
        
        max_gen_this_restart = max(100, 100 + 50 * dim)
        
        while time_left() > 0.2 and generation < max_gen_this_restart:
            if stagnation_count > 30 + 10 * dim:
                break
            if sigma < 1e-14:
                break
            
            if use_sep:
                sqrt_diag = np.sqrt(np.maximum(diag_C, 1e-20))
                solutions = []
                fitnesses = []
                for k in range(pop_size):
                    if time_left() <= 0.1:
                        return best
                    z = np.random.randn(dim)
                    x = mean + sigma * sqrt_diag * z
                    x = mirror_bounds(x)
                    f = eval_f(x)
                    solutions.append(x)
                    fitnesses.append(f)
                    if f < best_local:
                        best_local = f
                
                idx = np.argsort(fitnesses)
                old_mean = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += weights[i] * solutions[idx[i]]
                
                mean_diff = mean - old_mean
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * mean_diff / (sigma * sqrt_diag)
                
                norm_ps = np.linalg.norm(p_sigma)
                h_sigma = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff / sigma
                
                diag_C = (1 - c1 - c_mu_val) * diag_C + c1 * (p_c ** 2 + (1 - h_sigma) * c_c * (2 - c_c) * diag_C)
                for i in range(mu):
                    diag_C += c_mu_val * weights[i] * ((solutions[idx[i]] - old_mean) / sigma) ** 2
                diag_C = np.maximum(diag_C, 1e-20)
                
                sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
                sigma = min(sigma, np.mean(ranges) * 2)
            else:
                need_eigen = (generation % eigen_update_freq == 0)
                if need_eigen:
                    try:
                        C = (C + C.T) / 2
                        eigenvalues, B = np.linalg.eigh(C)
                        eigenvalues = np.maximum(eigenvalues, 1e-20)
                        D = np.sqrt(eigenvalues)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        C = np.eye(dim)
                        B = np.eye(dim)
                        D = np.ones(dim)
                        invsqrtC = np.eye(dim)
                
                solutions = []
                fitnesses = []
                for k in range(pop_size):
                    if time_left() <= 0.1:
                        return best
                    z = np.random.randn(dim)
                    x = mean + sigma * (B @ (D * z))
                    x = mirror_bounds(x)
                    f = eval_f(x)
                    solutions.append(x)
                    fitnesses.append(f)
                    if f < best_local:
                        best_local = f
                
                idx = np.argsort(fitnesses)
                old_mean = mean.copy()
                mean = np.sum(weights[:, None] * np.array([solutions[idx[i]] for i in range(mu)]), axis=0)
                
                mean_diff = mean - old_mean
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * invsqrtC @ mean_diff / sigma
                
                norm_ps = np.linalg.norm(p_sigma)
                h_sigma = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff / sigma
                
                artmp = np.array([(solutions[idx[i]] - old_mean) / sigma for i in range(mu)]).T
                C = (1 - c1 - c_mu_val) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu_val * (artmp * weights) @ artmp.T
                
                sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
                sigma = min(sigma, np.mean(ranges) * 2)
            
            generation += 1
            
            if best_local < prev_best_local - 1e-10:
                stagnation_count = 0
                prev_best_local = best_local
            else:
                stagnation_count += 1
            
            # Detect flat fitness
            if len(set(fitnesses)) <= 1:
                flat_count += 1
                if flat_count > 5:
                    break
            else:
                flat_count = 0
        
        restart_count += 1
    
    # --- Phase 3: Final local refinement with Nelder-Mead ---
    if best_x is not None and time_left() > 0.5:
        n = dim
        alpha, gamma, rho_nm, shrink = 1.0, 2.0, 0.5, 0.5
        step = np.minimum(ranges * 0.01, np.abs(best_x) * 0.05 + 1e-4)
        
        simplex = [best_x.copy()]
        simplex_f = [best]
        for i in range(n):
            if time_left() <= 0.2:
                return best
            xi = best_x.copy()
            xi[i] += step[i] if xi[i] + step[i] <= upper[i] else -step[i]
            fi = eval_f(xi)
            simplex.append(xi)
            simplex_f.append(fi)
        
        while time_left() > 0.1:
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = mirror_bounds(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < simplex_f[0]:
                xe = mirror_bounds(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                xc = mirror_bounds(centroid + rho_nm * (simplex[-1] - centroid))
                fc = eval_f(xc)
                if fc < simplex_f[-1]:
                    simplex[-1], simplex_f[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        if time_left() <= 0.1:
                            return best
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        simplex_f[i] = eval_f(simplex[i])
            
            if np.max(np.std(simplex, axis=0)) < 1e-12:
                break
    
    return best
