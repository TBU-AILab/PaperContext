#Looking at the previous results, the best algorithm (score 112.07) uses IPOP-CMA-ES. I'll improve it with several key enhancements:
#
#1. **Better time allocation** - less time on random init, more on optimization
#2. **Hybrid approach** - combine CMA-ES with Differential Evolution for diversity
#3. **Improved local search** - coordinate-wise search (Powell-like) around best
#4. **Smarter restarts** - alternating between small/large population (BIPOP)
#5. **Active CMA** - use negative weights for worst solutions
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
    
    evals = [0]
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Phase 1: Initialization with opposition-based learning
    n_init = min(max(20 * dim, 100), 500)
    archive = []
    
    for i in range(n_init):
        if remaining() <= 0:
            return best
        params = lower + ranges * np.random.rand(dim)
        fitness = eval_func(params)
        archive.append((fitness, params.copy()))
        # Opposition-based candidate
        if remaining() > 0:
            opp = lower + upper - params
            opp = np.clip(opp, lower, upper)
            f_opp = eval_func(opp)
            archive.append((f_opp, opp.copy()))
    
    archive.sort(key=lambda x: x[0])
    archive = archive[:200]  # Keep top solutions
    
    # Phase 2: Coordinate-wise local search on best
    def coord_search(x0, step_init, max_iters):
        x = x0.copy()
        fx = eval_func(x)
        step = step_init
        for _ in range(max_iters):
            improved = False
            for d in range(dim):
                if remaining() <= 0.3:
                    return x, fx
                trial = x.copy()
                trial[d] = np.clip(x[d] + step * ranges[d], lower[d], upper[d])
                ft = eval_func(trial)
                if ft < fx:
                    x = trial
                    fx = ft
                    improved = True
                    continue
                trial[d] = np.clip(x[d] - step * ranges[d], lower[d], upper[d])
                ft = eval_func(trial)
                if ft < fx:
                    x = trial
                    fx = ft
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-12:
                    break
        return x, fx
    
    if remaining() > 1.0:
        coord_search(best_params.copy(), 0.1, 3)
    
    # Phase 3: BIPOP-CMA-ES
    default_pop = 4 + int(3 * np.log(dim))
    large_pop_factor = 1
    use_large = False
    
    while remaining() > 0.5:
        if use_large:
            large_pop_factor *= 2
            pop_size = min(default_pop * large_pop_factor, 512)
            mean = lower + ranges * np.random.rand(dim)
            sigma = 0.4 * np.mean(ranges)
        else:
            pop_size = max(default_pop, 10)
            if best_params is not None and np.random.rand() < 0.6:
                mean = best_params.copy() + 0.05 * ranges * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
                sigma = 0.15 * np.mean(ranges)
            else:
                idx = np.random.randint(0, min(5, len(archive)))
                mean = archive[idx][1].copy()
                sigma = 0.25 * np.mean(ranges)
        
        use_large = not use_large
        mu = pop_size // 2
        
        weights_pos = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_pos = weights_pos / np.sum(weights_pos)
        mu_eff = 1.0 / np.sum(weights_pos ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        use_full = dim <= 60
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
        
        while remaining() > 0.3:
            if use_full and eigen_counter >= max(1, int(0.5/(c1+c_mu)/dim/5)):
                try:
                    ev, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(ev, 1e-20)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim); eigvals = np.ones(dim); eigvecs = np.eye(dim)
                eigen_counter = 0
            
            if use_full:
                sq = np.sqrt(eigvals)
                isq = 1.0/sq
            
            pop = []; fits = []
            for _ in range(pop_size):
                if remaining() <= 0.2: return best
                z = np.random.randn(dim)
                y = (eigvecs @ (sq * z)) if use_full else (np.sqrt(np.maximum(C_diag, 1e-20)) * z)
                x = np.clip(mean + sigma * y, lower, upper)
                f = eval_func(x)
                pop.append((x, y)); fits.append(f)
            
            idx = np.argsort(fits)
            if fits[idx[0]] < best_run: best_run = fits[idx[0]]; stag = 0
            else: stag += 1
            
            old_mean = mean.copy()
            mean = np.zeros(dim); y_w = np.zeros(dim)
            for i in range(mu):
                mean += weights_pos[i] * pop[idx[i]][0]
                y_w += weights_pos[i] * pop[idx[i]][1]
            mean = np.clip(mean, lower, upper)
            
            if use_full:
                p_sigma = (1-c_sigma)*p_sigma + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*(eigvecs@(isq*(eigvecs.T@y_w)))
            else:
                p_sigma = (1-c_sigma)*p_sigma + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*(y_w/np.sqrt(np.maximum(C_diag,1e-20)))
            
            ps_n = np.linalg.norm(p_sigma)
            h_s = 1 if ps_n/np.sqrt(1-(1-c_sigma)**(2*(gen+1))) < (1.4+2/(dim+1))*chi_n else 0
            p_c = (1-c_c)*p_c + h_s*np.sqrt(c_c*(2-c_c)*mu_eff)*y_w
            
            if use_full:
                art = np.column_stack([pop[idx[i]][1] for i in range(mu)])
                C = (1-c1-c_mu+(1-h_s)*c1*c_c*(2-c_c))*C + c1*np.outer(p_c,p_c) + c_mu*(art*weights_pos[:mu])@art.T
                C = (C+C.T)/2; eigen_counter += 1
            else:
                C_diag = (1-c1-c_mu+(1-h_s)*c1*c_c*(2-c_c))*C_diag + c1*p_c**2
                for i in range(mu): C_diag += c_mu*weights_pos[i]*pop[idx[i]][1]**2
                C_diag = np.maximum(C_diag, 1e-20)
            
            sigma *= np.exp((c_sigma/d_sigma)*(ps_n/chi_n - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.mean(ranges))
            gen += 1
            
            if stag > 15 + 3*dim or sigma < 1e-14: break
        
        # Local refinement after each run
        if remaining() > 1.0 and best_params is not None:
            coord_search(best_params.copy(), max(sigma/np.mean(ranges), 0.001), 2)
    
    return best
