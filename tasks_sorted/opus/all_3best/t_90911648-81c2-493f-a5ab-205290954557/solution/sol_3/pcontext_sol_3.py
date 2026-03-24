#Looking at the results, the best algorithm (112.07) uses IPOP-CMA-ES. I'll improve it with:
#
#1. **Less time wasted on initialization** - use fewer init samples
#2. **BIPOP strategy** - alternate between small local and large global populations
#3. **Active CMA** - use negative weights for worst solutions to repel from bad regions
#4. **Better local search** - Nelder-Mead simplex around best after CMA runs
#5. **Smarter eigendecomposition caching**
#6. **More efficient restart initialization using archive diversity**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Phase 1: Compact initialization
    n_init = min(max(15 * dim, 100), 400)
    archive = []
    
    for i in range(n_init):
        if remaining() <= 0:
            return best
        params = lower + ranges * np.random.rand(dim)
        fitness = eval_func(params)
        archive.append((fitness, params.copy()))
    
    archive.sort(key=lambda x: x[0])
    archive = archive[:100]
    
    # Nelder-Mead style local search
    def nelder_mead_search(x0, initial_step, max_evals):
        n = dim
        # Create initial simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = eval_func(x0)
        evals_used = 1
        
        for i in range(n):
            if remaining() <= 0.3:
                return
            point = x0.copy()
            point[i] = np.clip(point[i] + initial_step * ranges[i], lower[i], upper[i])
            simplex[i + 1] = point
            f_simplex[i + 1] = eval_func(point)
            evals_used += 1
            if evals_used >= max_evals:
                return
        
        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(max_evals - evals_used):
            if remaining() <= 0.3:
                return
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                # Expand
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
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
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                else:
                    # Inside contraction
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), lower, upper)
                fc = eval_func(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        if remaining() <= 0.3:
                            return
                        simplex[i] = np.clip(simplex[0] + shrink * (simplex[i] - simplex[0]), lower, upper)
                        f_simplex[i] = eval_func(simplex[i])
    
    # Quick local search on best
    if remaining() > 1.0 and best_params is not None:
        nelder_mead_search(best_params.copy(), 0.05, min(dim * 3, 150))
    
    # Phase 2: BIPOP-CMA-ES
    default_pop = 4 + int(3 * np.log(dim))
    large_budget_used = 0
    small_budget_used = 0
    large_pop_factor = 1
    run_number = 0
    
    while remaining() > 0.5:
        run_number += 1
        
        # BIPOP: alternate between large-pop global and small-pop local
        use_large = (run_number % 3 == 0) or (small_budget_used > large_budget_used * 1.5)
        
        if use_large:
            large_pop_factor = min(large_pop_factor * 2, 16)
            pop_size = min(default_pop * large_pop_factor, 256)
            mean = lower + ranges * np.random.rand(dim)
            sigma = 0.4 * np.mean(ranges)
        else:
            pop_size = max(default_pop, 10)
            r = np.random.rand()
            if r < 0.5 and best_params is not None:
                mean = best_params.copy() + 0.03 * ranges * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
                sigma = 0.1 * np.mean(ranges)
            elif r < 0.8 and len(archive) > 1:
                idx = np.random.randint(0, min(10, len(archive)))
                mean = archive[idx][1].copy() + 0.02 * ranges * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
                sigma = 0.15 * np.mean(ranges)
            else:
                mean = lower + ranges * np.random.rand(dim)
                sigma = 0.3 * np.mean(ranges)
        
        mu = pop_size // 2
        
        # Weights with active CMA (negative weights for worst)
        raw_w = np.log(pop_size / 2.0 + 0.5) - np.log(np.arange(1, pop_size + 1))
        w_pos = raw_w[:mu].copy()
        w_pos /= np.sum(w_pos)
        mu_eff = 1.0 / np.sum(w_pos ** 2)
        
        # Negative weights
        w_neg = raw_w[mu:]
        if np.sum(np.abs(w_neg)) > 0:
            w_neg = w_neg / np.sum(np.abs(w_neg))
        mu_eff_neg = 1.0 / (np.sum(w_neg ** 2) + 1e-20)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Active CMA scaling
        alpha_mu_neg = 1 + c1 / (c_mu_val + 1e-20)
        alpha_mu_eff = 1 + 2 * mu_eff_neg / (mu_eff + 2)
        alpha_pos_def = (1 - c1 - c_mu_val) / (dim * c_mu_val + 1e-20)
        alpha_min = min(alpha_mu_neg, alpha_mu_eff, alpha_pos_def)
        
        weights_full = np.concatenate([w_pos, alpha_min * w_neg])
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        use_full = dim <= 70
        if use_full:
            C = np.eye(dim)
            eigvals = np.ones(dim)
            eigvecs = np.eye(dim)
            eigen_counter = 0
        else:
            C_diag = np.ones(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        stag = 0
        best_run = float('inf')
        gen = 0
        gen_evals = 0
        
        while remaining() > 0.3:
            if use_full and eigen_counter >= max(1, int(0.5/(c1+c_mu_val)/dim/5)):
                try:
                    ev, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(ev, 1e-20)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim); eigvals = np.ones(dim); eigvecs = np.eye(dim)
                eigen_counter = 0
            
            if use_full:
                sq = np.sqrt(eigvals)
                isq = 1.0 / sq
            
            pop = []; fits = []
            for _ in range(pop_size):
                if remaining() <= 0.2: return best
                z = np.random.randn(dim)
                y = (eigvecs @ (sq * z)) if use_full else (np.sqrt(np.maximum(C_diag, 1e-20)) * z)
                x = np.clip(mean + sigma * y, lower, upper)
                f = eval_func(x)
                pop.append((x, y)); fits.append(f)
                gen_evals += 1
            
            idx = np.argsort(fits)
            if fits[idx[0]] < best_run: best_run = fits[idx[0]]; stag = 0
            else: stag += 1
            
            mean_new = np.zeros(dim); y_w = np.zeros(dim)
            for i in range(mu):
                mean_new += w_pos[i] * pop[idx[i]][0]
                y_w += w_pos[i] * pop[idx[i]][1]
            mean = np.clip(mean_new, lower, upper)
            
            if use_full:
                p_sigma = (1-c_sigma)*p_sigma + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*(eigvecs@(isq*(eigvecs.T@y_w)))
            else:
                p_sigma = (1-c_sigma)*p_sigma + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*(y_w/np.sqrt(np.maximum(C_diag,1e-20)))
            
            ps_n = np.linalg.norm(p_sigma)
            h_s = 1 if ps_n/np.sqrt(1-(1-c_sigma)**(2*(gen+1))) < (1.4+2/(dim+1))*chi_n else 0
            p_c = (1-c_c)*p_c + h_s*np.sqrt(c_c*(2-c_c)*mu_eff)*y_w
            
            if use_full:
                # Active CMA update
                art_all = np.column_stack([pop[idx[i]][1] for i in range(pop_size)])
                C_new = (1-c1-c_mu_val*np.sum(np.abs(weights_full))+(1-h_s)*c1*c_c*(2-c_c))*C + c1*np.outer(p_c,p_c)
                for i in range(pop_size):
                    wi = weights_full[i] if i < len(weights_full) else 0
                    if wi != 0:
                        yi = pop[idx[i]][1]
                        C_new += c_mu_val * wi * np.outer(yi, yi)
                C = (C_new + C_new.T) / 2
                eigen_counter += 1
            else:
                C_diag = (1-c1-c_mu_val+(1-h_s)*c1*c_c*(2-c_c))*C_diag + c1*p_c**2
                for i in range(mu): C_diag += c_mu_val*w_pos[i]*pop[idx[i]][1]**2
                C_diag = np.maximum(C_diag, 1e-20)
            
            sigma *= np.exp((c_sigma/d_sigma)*(ps_n/chi_n - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.mean(ranges))
            gen += 1
            
            if stag > 10 + 3*dim or sigma < 1e-14: break
        
        if use_large: large_budget_used += gen_evals
        else: small_budget_used += gen_evals
        
        # Archive update
        if best_params is not None:
            archive.append((best, best_params.copy()))
            archive.sort(key=lambda x: x[0])
            archive = archive[:50]
        
        # Local refinement after each run
        if remaining() > 1.5 and best_params is not None:
            nelder_mead_search(best_params.copy(), max(sigma/np.mean(ranges), 0.005), min(dim * 2, 100))
    
    return best
