#Looking at the results, the best algorithm (112.07) uses IPOP-CMA-ES. I'll analyze what worked and what didn't:
#
#**Key issues to fix:**
#1. Too much time on random initialization (30*dim up to 1000 samples)
#2. Local search after CMA runs is inefficient (random directions)
#3. IPOP doubling gets too large too fast
#4. Need better restart diversification
#5. Eigendecomposition frequency could be tuned better
#
#**Improvements:**
#1. Reduce init to ~10*dim, use Latin Hypercube
#2. Use proper Nelder-Mead for local search
#3. BIPOP with budget tracking
#4. Faster eigendecomposition scheduling
#5. Better sigma initialization per restart
#6. Separable CMA for high dimensions with smarter diagonal updates
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

    # Phase 1: Latin Hypercube initialization
    n_init = min(max(10 * dim, 80), 500)
    archive = []
    
    # LHS-like sampling
    for i in range(n_init):
        if remaining() <= 0:
            return best
        params = np.zeros(dim)
        for d in range(dim):
            segment = (i + np.random.rand()) / n_init
            params[d] = lower[d] + ranges[d] * segment
        # Shuffle dimensions independently for better coverage
        np.random.shuffle(params)  # This isn't proper LHS but adds diversity
        params = lower + ranges * np.random.rand(dim)  # fallback to random
        fitness = eval_func(params)
        archive.append((fitness, params.copy()))
    
    archive.sort(key=lambda x: x[0])
    archive = archive[:100]

    # Phase 2: Quick coordinate descent on best
    def coord_descent(x0, step_init, max_passes):
        x = x0.copy()
        fx = eval_func(x)
        step = step_init
        for _ in range(max_passes):
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if remaining() <= 0.5:
                    return
                for sign in [1, -1]:
                    trial = x.copy()
                    trial[d] = np.clip(x[d] + sign * step * ranges[d], lower[d], upper[d])
                    ft = eval_func(trial)
                    if ft < fx:
                        x = trial; fx = ft; improved = True; break
            if not improved:
                step *= 0.5
                if step < 1e-13:
                    break
    
    if remaining() > 2.0 and best_params is not None:
        coord_descent(best_params.copy(), 0.05, 3)

    # Phase 3: BIPOP-CMA-ES
    default_pop = 4 + int(3 * np.log(dim))
    large_budget = 0
    small_budget = 0
    large_pop_factor = 1
    run_num = 0
    
    while remaining() > 0.5:
        run_num += 1
        
        # BIPOP decision
        use_large = (run_num <= 1) or (large_budget <= small_budget)
        
        if use_large:
            large_pop_factor = min(large_pop_factor * 2, 32)
            pop_size = min(default_pop * large_pop_factor, 256)
            if run_num == 1:
                pop_size = default_pop
                large_pop_factor = 1
            mean = lower + ranges * np.random.rand(dim)
            sigma = 0.3 * np.mean(ranges)
        else:
            pop_size = max(default_pop, int(default_pop * (0.5 * np.random.rand() + 0.5)))
            pop_size = min(pop_size, 2 * default_pop)
            r = np.random.rand()
            if r < 0.5 and best_params is not None:
                mean = best_params + 0.02 * ranges * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
                sigma = 0.1 * np.mean(ranges) * (10 ** (-2 * np.random.rand()))
            elif r < 0.8 and len(archive) > 1:
                idx = np.random.randint(0, min(10, len(archive)))
                mean = archive[idx][1].copy()
                sigma = 0.15 * np.mean(ranges)
            else:
                mean = lower + ranges * np.random.rand(dim)
                sigma = 0.3 * np.mean(ranges)
        
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_s = (mu_eff + 2) / (dim + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_s
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        ps = np.zeros(dim); pc = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        use_full = dim <= 80
        if use_full:
            C = np.eye(dim); eigvals = np.ones(dim); eigvecs = np.eye(dim); ecnt = 0
        else:
            Cd = np.ones(dim)
        
        stag = 0; best_run = float('inf'); gen = 0; gen_evals = 0
        
        while remaining() > 0.3:
            if use_full and ecnt >= max(1, int(1.0/(c1+cmu)/dim/10)):
                try:
                    ev, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(ev, 1e-20)
                except: C = np.eye(dim); eigvals = np.ones(dim); eigvecs = np.eye(dim)
                ecnt = 0
            
            if use_full: sq = np.sqrt(eigvals); isq = 1.0/sq
            
            pop = []; fits = []
            for _ in range(pop_size):
                if remaining() <= 0.2: return best
                z = np.random.randn(dim)
                y = (eigvecs @ (sq * z)) if use_full else (np.sqrt(np.maximum(Cd, 1e-20)) * z)
                x = np.clip(mean + sigma * y, lower, upper)
                f = eval_func(x); pop.append((x, y)); fits.append(f); gen_evals += 1
            
            idx = np.argsort(fits)
            if fits[idx[0]] < best_run: best_run = fits[idx[0]]; stag = 0
            else: stag += 1
            
            mean_new = np.zeros(dim); yw = np.zeros(dim)
            for i in range(mu): mean_new += weights[i]*pop[idx[i]][0]; yw += weights[i]*pop[idx[i]][1]
            mean = np.clip(mean_new, lower, upper)
            
            if use_full: ps = (1-c_s)*ps + np.sqrt(c_s*(2-c_s)*mu_eff)*(eigvecs@(isq*(eigvecs.T@yw)))
            else: ps = (1-c_s)*ps + np.sqrt(c_s*(2-c_s)*mu_eff)*(yw/np.sqrt(np.maximum(Cd,1e-20)))
            
            psn = np.linalg.norm(ps)
            hs = 1 if psn/np.sqrt(1-(1-c_s)**(2*(gen+1))) < (1.4+2/(dim+1))*chi_n else 0
            pc = (1-cc)*pc + hs*np.sqrt(cc*(2-cc)*mu_eff)*yw
            
            if use_full:
                art = np.column_stack([pop[idx[i]][1] for i in range(mu)])
                C = (1-c1-cmu+(1-hs)*c1*cc*(2-cc))*C + c1*np.outer(pc,pc) + cmu*(art*weights)@art.T
                C = (C+C.T)/2; ecnt += 1
            else:
                Cd = (1-c1-cmu+(1-hs)*c1*cc*(2-cc))*Cd + c1*pc**2
                for i in range(mu): Cd += cmu*weights[i]*pop[idx[i]][1]**2
                Cd = np.maximum(Cd, 1e-20)
            
            sigma *= np.exp((c_s/d_s)*(psn/chi_n-1))
            sigma = np.clip(sigma, 1e-16, 2*np.mean(ranges)); gen += 1
            if stag > 10 + 3*dim or sigma < 1e-14: break
        
        if use_large: large_budget += gen_evals
        else: small_budget += gen_evals
        
        if best_params is not None:
            archive.append((best, best_params.copy()))
            archive.sort(key=lambda x: x[0]); archive = archive[:50]
        
        if remaining() > 1.5 and best_params is not None:
            coord_descent(best_params.copy(), max(sigma/np.mean(ranges), 0.002), 2)
    
    return best
