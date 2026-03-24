#Looking at the best algorithm (score 38.03), I'll make several targeted improvements:
#
#1. **Faster mirror bounds** using modular arithmetic instead of while loops
#2. **BIPOP restart strategy** - alternating between small local and large global restarts
#3. **Lower sep-CMA threshold** (dim > 60) for better scaling
#4. **Active CMA-ES** with negative weights
#5. **Better sigma/restart management** - more restarts from best with small sigma
#6. **Reduced init budget** to spend more time on CMA-ES
#7. **Nelder-Mead only when dimension is small enough to be useful**
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
    center = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.96 - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        if f < best:
            best = f
            best_x = x_c.copy()
        return f
    
    def mirror_bounds(x):
        x = x.copy()
        for d in range(dim):
            r = ranges[d]
            if r <= 0:
                x[d] = lower[d]
                continue
            x[d] -= lower[d]
            period = 2 * r
            x[d] = x[d] % period
            if x[d] > r:
                x[d] = period - x[d]
            x[d] += lower[d]
        return np.clip(x, lower, upper)
    
    # Phase 1: LHS initialization - reduced budget
    n_init = min(max(20 * dim, 150), 800)
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fitnesses = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_f(init_points[i])
        init_fitnesses.append((f, i))
    
    eval_f(center)
    
    init_fitnesses.sort()
    top_k = min(10, len(init_fitnesses))
    starting_points = [init_points[init_fitnesses[i][1]].copy() for i in range(top_k)]
    
    # Phase 2: BIPOP-CMA-ES
    base_pop_size = max(4 + int(3 * np.log(dim)), 10)
    restart_count = 0
    large_restarts = 0
    small_restarts = 0
    
    def run_cmaes(x0, pop_size, sigma0):
        nonlocal best, best_x
        
        mu = pop_size // 2
        weights_pos = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_pos /= np.sum(weights_pos)
        mu_eff = 1.0 / np.sum(weights_pos ** 2)
        
        # Active CMA negative weights
        n_neg = pop_size - mu
        weights_neg_raw = np.log(mu + 0.5) - np.log(np.arange(mu + 1, pop_size + 1))
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        alpha_mu_neg = 1 + c1 / c_mu_val if c_mu_val > 0 else 1
        alpha_mueff_neg = 1 + 2 * mu_eff / (mu_eff + 2)
        alpha_pos_def = (1 - c1 - c_mu_val) / (dim * c_mu_val) if c_mu_val > 0 else 1
        min_alpha = min(alpha_mu_neg, alpha_mueff_neg, alpha_pos_def)
        
        if n_neg > 0 and np.sum(np.abs(weights_neg_raw)) > 0:
            weights_neg = -min_alpha * mu_eff * weights_neg_raw / np.sum(np.abs(weights_neg_raw))
        else:
            weights_neg = np.zeros(max(n_neg, 0))
        
        use_sep = dim > 50
        
        mean = x0.copy()
        sigma = sigma0
        
        if use_sep:
            diag_C = np.ones(dim)
            p_s = np.zeros(dim)
            p_c = np.zeros(dim)
        else:
            C = np.eye(dim)
            p_s = np.zeros(dim)
            p_c = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_countdown = 0
        
        gen = 0
        stag = 0
        best_l = float('inf')
        prev_best_l = float('inf')
        flat_count = 0
        max_gen = max(100, 300 + 80 * dim // pop_size)
        local_evals = 0
        hist_best = []
        
        while time_left() > 0.2 and gen < max_gen:
            if stag > 25 + 15 * dim // pop_size:
                break
            if sigma < 1e-16:
                break
            # Condition number check for sep
            if use_sep:
                cond = np.max(diag_C) / max(np.min(diag_C), 1e-20)
                if cond > 1e14:
                    break
            
            if use_sep:
                sq = np.sqrt(np.maximum(diag_C, 1e-20))
                Z = np.random.randn(pop_size, dim)
                X = mean[None, :] + sigma * sq[None, :] * Z
                sols = []
                fits = []
                for k in range(pop_size):
                    if time_left() <= 0.1:
                        return
                    xk = mirror_bounds(X[k])
                    f = eval_f(xk)
                    local_evals += 1
                    sols.append(xk)
                    fits.append(f)
                    if f < best_l:
                        best_l = f
                
                idx = np.argsort(fits)
                old_mean = mean.copy()
                sel = np.array([sols[idx[i]] for i in range(mu)])
                mean = weights_pos @ sel
                md = mean - old_mean
                p_s = (1-c_sigma)*p_s + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*md/(sigma*sq)
                nps = np.linalg.norm(p_s)
                hs = 1 if nps/np.sqrt(1-(1-c_sigma)**(2*(gen+1))) < (1.4+2/(dim+1))*chi_n else 0
                p_c = (1-c_c)*p_c + hs*np.sqrt(c_c*(2-c_c)*mu_eff)*md/sigma
                
                new_diag = (1-c1-c_mu_val)*diag_C + c1*(p_c**2+(1-hs)*c_c*(2-c_c)*diag_C)
                for i in range(mu):
                    new_diag += c_mu_val*weights_pos[i]*((sols[idx[i]]-old_mean)/sigma)**2
                if n_neg > 0:
                    for i in range(n_neg):
                        j = idx[mu + i]
                        diff = (sols[j]-old_mean)/sigma
                        # Normalize by diag for neg weights
                        new_diag += c_mu_val*weights_neg[i]*diff**2
                diag_C = np.maximum(new_diag, 1e-20)
                sigma *= np.exp((c_sigma/d_sigma)*(nps/chi_n-1))
            else:
                if eigen_countdown <= 0:
                    try:
                        C = (C+C.T)/2
                        ev, B = np.linalg.eigh(C)
                        ev = np.maximum(ev, 1e-20)
                        if np.max(ev)/np.min(ev) > 1e14:
                            ev = np.clip(ev, np.max(ev)*1e-14, None)
                        D = np.sqrt(ev)
                        invsqrtC = B @ np.diag(1/D) @ B.T
                    except:
                        C = np.eye(dim); B = np.eye(dim); D = np.ones(dim); invsqrtC = np.eye(dim)
                    eigen_countdown = max(1, int(1/(c1+c_mu_val)/dim/10))
                eigen_countdown -= 1
                
                Z = np.random.randn(pop_size, dim)
                BD = B * D[None, :]
                X = mean[None,:] + sigma * (Z @ BD.T)
                sols = []; fits = []
                for k in range(pop_size):
                    if time_left() <= 0.1:
                        return
                    xk = mirror_bounds(X[k])
                    f = eval_f(xk)
                    local_evals += 1
                    sols.append(xk)
                    fits.append(f)
                    if f < best_l:
                        best_l = f
                
                idx = np.argsort(fits)
                old_mean = mean.copy()
                sel = np.array([sols[idx[i]] for i in range(mu)])
                mean = weights_pos @ sel
                md = mean - old_mean
                p_s = (1-c_sigma)*p_s + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*invsqrtC@md/sigma
                nps = np.linalg.norm(p_s)
                hs = 1 if nps/np.sqrt(1-(1-c_sigma)**(2*(gen+1))) < (1.4+2/(dim+1))*chi_n else 0
                p_c = (1-c_c)*p_c + hs*np.sqrt(c_c*(2-c_c)*mu_eff)*md/sigma
                
                artmp_pos = np.array([(sols[idx[i]]-old_mean)/sigma for i in range(mu)]).T
                C_new = (1-c1-c_mu_val)*C + c1*(np.outer(p_c,p_c)+(1-hs)*c_c*(2-c_c)*C) + c_mu_val*(artmp_pos*weights_pos)@artmp_pos.T
                
                if n_neg > 0:
                    artmp_neg = np.array([(sols[idx[mu+i]]-old_mean)/sigma for i in range(n_neg)]).T
                    C_new += c_mu_val * (artmp_neg * weights_neg) @ artmp_neg.T
                
                C = C_new
                sigma *= np.exp((c_sigma/d_sigma)*(nps/chi_n-1))
            
            sigma = min(sigma, np.mean(ranges)*2)
            sigma = max(sigma, 1e-20)
            gen += 1
            
            if best_l < prev_best_l - 1e-10:
                stag = 0
                prev_best_l = best_l
            else:
                stag += 1
            
            if len(set(fits)) <= 1:
                flat_count += 1
                if flat_count > 5:
                    break
            else:
                flat_count = 0
            
            hist_best.append(best_l)
            if len(hist_best) > 50:
                if abs(hist_best[-1] - hist_best[-50]) < 1e-12 * (1 + abs(hist_best[-1])):
                    break
    
    while time_left() > 0.5:
        if restart_count < len(starting_points):
            x0 = starting_points[restart_count].copy()
            pop_size = base_pop_size
            sigma0 = np.mean(ranges) / 4.0
        else:
            # BIPOP: alternate small/large
            use_large = (large_restarts == 0) or (small_restarts / max(large_restarts, 1) > 3)
            if use_large:
                large_restarts += 1
                x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
                pop_size = min(base_pop_size * (2 ** min(large_restarts, 5)), 512)
                sigma0 = np.mean(ranges) / 3.0
            else:
                small_restarts += 1
                if best_x is not None and np.random.random() < 0.7:
                    x0 = best_x.copy() + np.random.randn(dim) * ranges * 0.02
                    x0 = np.clip(x0, lower, upper)
                    sigma0 = np.mean(ranges) / (6.0 + small_restarts)
                else:
                    x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
                    sigma0 = np.mean(ranges) / 5.0
                pop_size = max(base_pop_size, int(base_pop_size * (0.5 * np.random.uniform(0.5, 1.5))))
        
        run_cmaes(x0, pop_size, sigma0)
        restart_count += 1
    
    # Phase 3: Nelder-Mead refinement for low dim
    if best_x is not None and time_left() > 0.3 and dim <= 40:
        n = dim
        step = np.minimum(ranges * 0.005, np.abs(best_x) * 0.02 + 1e-5)
        simplex = [best_x.copy()]
        simplex_f = [best]
        for i in range(n):
            if time_left() <= 0.15:
                return best
            xi = best_x.copy()
            xi[i
