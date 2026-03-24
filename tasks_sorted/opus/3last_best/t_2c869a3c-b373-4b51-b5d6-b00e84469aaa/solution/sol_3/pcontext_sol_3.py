#Looking at the best algorithm (score 38.03), I can identify several key improvements:
#
#1. **Vectorized mirror bounds** - the per-element while loop is slow
#2. **Better BIPOP restart strategy** - alternating small/large populations
#3. **Reduced initialization overhead** - spend more budget on CMA-ES
#4. **Threshold for sep-CMA lowered** to handle moderate dimensions better
#5. **Smarter sigma selection per restart**
#6. **Active CMA-ES** with negative weights for worst solutions
#7. **Better stagnation criteria and time management**
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
    
    evals = 0
    def eval_f(x):
        nonlocal best, best_x, evals
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        evals += 1
        if f < best:
            best = f
            best_x = x_c.copy()
        return f
    
    def mirror_bounds_vec(x):
        x = x.copy()
        for d in range(dim):
            r = ranges[d]
            if r <= 0:
                x[d] = lower[d]
                continue
            # Normalize to [0, r]
            x[d] -= lower[d]
            # Use modular arithmetic for mirroring
            period = 2 * r
            x[d] = x[d] % period
            if x[d] > r:
                x[d] = period - x[d]
            x[d] += lower[d]
        return np.clip(x, lower, upper)
    
    # Phase 1: LHS initialization
    n_init = min(max(15 * dim, 100), 600)
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
    top_k = min(8, len(init_fitnesses))
    starting_points = [init_points[init_fitnesses[i][1]].copy() for i in range(top_k)]
    
    # Phase 2: BIPOP-CMA-ES
    base_pop_size = 4 + int(3 * np.log(dim))
    base_pop_size = max(base_pop_size, 10)
    restart_count = 0
    large_budget = 0
    small_budget = 0
    
    def run_cmaes(x0, pop_size, sigma0, max_evals_run=None):
        nonlocal best, best_x
        
        mu = pop_size // 2
        weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_pos = weights_raw / np.sum(weights_raw)
        mu_eff = 1.0 / np.sum(weights_pos ** 2)
        
        # Active CMA: negative weights
        n_neg = pop_size - mu
        weights_neg_raw = np.log(mu + 0.5) - np.log(np.arange(mu + 1, pop_size + 1))
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Active CMA negative weight factor
        alpha_mu_neg = 1 + c1 / c_mu_val
        alpha_mueff_neg = 1 + 2 * mu_eff / (mu_eff + 2)
        alpha_pos_def = (1 - c1 - c_mu_val) / (dim * c_mu_val)
        min_alpha = min(alpha_mu_neg, alpha_mueff_neg, alpha_pos_def)
        
        if n_neg > 0 and np.sum(np.abs(weights_neg_raw)) > 0:
            weights_neg = weights_neg_raw / np.sum(np.abs(weights_neg_raw)) * min_alpha * mu_eff * (-1)
        else:
            weights_neg = np.zeros(n_neg)
        
        # Full weight vector
        weights_full = np.concatenate([weights_pos, weights_neg]) if n_neg > 0 else weights_pos.copy()
        
        use_sep = dim > 60
        
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
        max_gen = max(100, 200 + 50 * dim // pop_size)
        local_evals = 0
        
        while time_left() > 0.15 and gen < max_gen:
            if max_evals_run and local_evals >= max_evals_run:
                break
            if stag > 25 + 10 * dim // pop_size:
                break
            if sigma < 1e-15:
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
                    xk = mirror_bounds_vec(X[k])
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
                # Active: subtract negative weights contribution
                if n_neg > 0:
                    for i in range(n_neg):
                        j = idx[mu + i]
                        new_diag += c_mu_val*weights_neg[i]*((sols[j]-old_mean)/sigma)**2
                diag_C = np.maximum(new_diag, 1e-20)
                sigma *= np.exp((c_sigma/d_sigma)*(nps/chi_n-1))
            else:
                if eigen_countdown <= 0:
                    try:
                        C = (C+C.T)/2
                        ev, B = np.linalg.eigh(C)
                        ev = np.maximum(ev, 1e-20)
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
                    xk = mirror_bounds_vec(X[k])
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
                
                # Active CMA negative update
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
    
    while time_left() > 0.3:
        if restart_count < len(starting_points):
            x0 = starting_points[restart_count].copy()
            pop_size = base_pop_size
            sigma0 = np.mean(ranges) / 4.0
        elif restart_count % 2 == 0 or (best_x is not None and np.random.random() < 0.5):
            x0 = best_x.copy() if best_x is not None else center.copy()
            x0 = x0 + np.random.randn(dim) * ranges * 0.02
            x0 = np.clip(x0, lower, upper)
            pop_size = base_pop_size
            sigma0 = np.mean(ranges) / 8.0
            small_budget += pop_size * 100
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            large_budget += 1
            pop_size = min(base_pop_size * (2 ** min(large_budget, 4)), 256)
            sigma0 = np.mean(ranges) / 3.0
        
        run_cmaes(x0, pop_size, sigma0)
        restart_count += 1
    
    return best
