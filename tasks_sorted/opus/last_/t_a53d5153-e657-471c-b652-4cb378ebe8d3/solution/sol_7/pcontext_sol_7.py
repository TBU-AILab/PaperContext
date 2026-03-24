#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **DE phase takes too much time** with complex SHADE mechanics but may not explore well enough
#2. **CMA-ES restart strategy** could be better tuned with smarter budget allocation
#3. **Need better initial exploration** with opposition-based learning
#4. **Bayesian-like exploitation** of good regions found so far
#5. **Powell's method** as an additional local search
#
#Let me redesign with cleaner phases and better time allocation:
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
    mid = (lower + upper) / 2
    
    evals = 0
    all_points = []
    all_fits = []
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max(0, max_time - elapsed())
    
    def eval_f(x):
        nonlocal best, best_params, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        all_points.append(x.copy())
        all_fits.append(f)
        return f

    # === Phase 1: Sobol-like LHS + opposition-based sampling ===
    n_init = min(max(15 * dim, 100), 500)
    population = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        population[:, d] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    fitness = np.empty(n_init)
    for i in range(n_init):
        if elapsed() >= max_time * 0.05:
            n_init = i
            break
        fitness[i] = eval_f(population[i])
    
    if n_init == 0:
        return best
    
    population = population[:n_init]
    fitness = fitness[:n_init]
    
    # Opposition-based learning
    n_opp = min(n_init, 100)
    top_idx = np.argsort(fitness)[:n_opp]
    for i in range(n_opp):
        if elapsed() >= max_time * 0.08:
            break
        opp = lower + upper - population[top_idx[i]]
        eval_f(opp)
    
    # === Phase 2: CMA-ES with smart restarts ===
    def run_cmaes(init_mean, init_sigma, deadline, lam_override=None):
        nonlocal best, best_params
        n = dim
        
        lam = lam_override if lam_override else max(4 + int(3 * np.log(n)), 8)
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n+1.3)**2 + mu_eff)
        cmu_val = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + mu_eff))
        damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = np.clip(init_mean.copy(), lower, upper)
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 80)
        
        if use_full:
            C = np.eye(n)
            eig_vals_sq = np.ones(n)
            B = np.eye(n)
            invsqrtC = np.eye(n)
            eigen_counter = 0
            eigen_interval = max(1, int(1 / (10*n*(c1 + cmu_val) + 1e-30)))
        else:
            diag_C = np.ones(n)
        
        stale = 0
        best_gen_f = float('inf')
        f_hist = []
        gen = 0
        
        while elapsed() < deadline:
            gen += 1
            offspring = np.empty((lam, n))
            f_off = np.empty(lam)
            
            for i in range(lam):
                if elapsed() >= deadline:
                    return
                z = np.random.randn(n)
                if use_full:
                    offspring[i] = mean + sigma * (B @ (eig_vals_sq * z))
                else:
                    offspring[i] = mean + sigma * np.sqrt(np.maximum(diag_C, 1e-20)) * z
                offspring[i] = np.clip(offspring[i], lower, upper)
                f_off[i] = eval_f(offspring[i])
            
            order = np.argsort(f_off)
            selected = offspring[order[:mu]]
            
            old_mean = mean.copy()
            mean = np.dot(weights, selected)
            mean = np.clip(mean, lower, upper)
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ diff)
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * diff / np.sqrt(np.maximum(diag_C, 1e-20))
            
            ps_norm = np.linalg.norm(ps)
            gen_thresh = max(1 - (1-cs)**(2*gen), 1e-30)
            hs = 1 if ps_norm / np.sqrt(gen_thresh) < (1.4 + 2/(n+1)) * chi_n else 0
            
            pc = (1-cc)*pc + hs * np.sqrt(cc*(2-cc)*mu_eff) * diff
            
            if use_full:
                artmp = ((selected - old_mean) / max(sigma, 1e-30)).T
                C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * C + \
                    c1 * np.outer(pc, pc) + cmu_val * (artmp @ np.diag(weights) @ artmp.T)
                C = np.triu(C) + np.triu(C, 1).T
                
                eigen_counter += 1
                if eigen_counter >= eigen_interval:
                    eigen_counter = 0
                    try:
                        ev, B = np.linalg.eigh(C)
                        ev = np.maximum(ev, 1e-20)
                        eig_vals_sq = np.sqrt(ev)
                        invsqrtC = B @ np.diag(1.0/eig_vals_sq) @ B.T
                    except:
                        C = np.eye(n); eig_vals_sq = np.ones(n)
                        B = np.eye(n); invsqrtC = np.eye(n)
            else:
                artmp = (selected - old_mean) / max(sigma, 1e-30)
                diag_C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * diag_C + \
                         c1 * pc**2 + cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp(np.clip((cs/damps) * (ps_norm/chi_n - 1), -0.5, 0.5))
            sigma = np.clip(sigma, 1e-17 * np.mean(ranges), 2.0 * np.mean(ranges))
            
            cur_best = f_off[order[0]]
            f_hist.append(cur_best)
            
            if cur_best < best_gen_f - 1e-15:
                best_gen_f = cur_best
                stale = 0
            else:
                stale += 1
            
            if use_full:
                max_std = sigma * np.max(eig_vals_sq)
            else:
                max_std = sigma * np.max(np.sqrt(diag_C))
            
            if max_std < 1e-14 * np.mean(ranges):
                return
            if stale > 10 + 30 * n // lam:
                return
            if len(f_hist) > 40:
                recent = f_hist[-40:]
                if max(recent) - min(recent) < 1e-14 * (abs(best) + 1e-30):
                    return

    # Gather top points for restart seeds
    all_fits_arr = np.array(all_fits)
    all_points_arr = np.array(all_points)
    top_k = min(10, len(all_fits_arr))
    top_indices = np.argsort(all_fits_arr)[:top_k]
    
    # Run CMA-ES restarts with IPOP
    lam_mult = 1.0
    restart = 0
    base_lam = max(4 + int(3 * np.log(dim)), 8)
    
    while elapsed() < max_time * 0.85:
        if remaining() < 0.5:
            break
        
        if restart == 0:
            sp = best_params.copy()
            sig = 0.2 * np.mean(ranges)
            lam = base_lam
        elif restart == 1:
            sp = best_params.copy()
            sig = 0.02 * np.mean(ranges)
            lam = base_lam
        elif restart % 5 == 0:
            # IPOP: large population random restart
            lam_mult = min(lam_mult * 2, 12)
            sp = lower + np.random.rand(dim) * ranges
            sig = 0.3 * np.mean(ranges)
            lam = int(base_lam * lam_mult)
        elif restart % 5 == 1:
            # Restart from a random top point
            idx = top_indices[np.random.randint(min(5, len(top_indices)))]
            sp = all_points_arr[idx] + 0.05 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.08 * np.mean(ranges)
            lam = base_lam
        elif restart % 5 == 2:
            # Random restart medium sigma
            sp = lower + np.random.rand(dim) * ranges
            sig = 0.15 * np.mean(ranges)
            lam = base_lam
        elif restart % 5 == 3:
            # Near-best with small sigma
            sp = best_params + 0.01 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.01 * np.mean(ranges)
            lam = base_lam
        else:
            # Midpoint between two good solutions
            if len(top_indices) >= 2:
                i1, i2 = np.random.choice(top_indices[:min(5, len(top_indices))], 2, replace=False)
                sp = 0.5 * (all_points_arr[i1] + all_points_arr[i2]) + 0.02 * ranges * np.random.randn(dim)
                sp = np.clip(sp, lower, upper)
            else:
                sp = best_params + 0.05 * ranges * np.random.randn(dim)
                sp = np.clip(sp, lower, upper)
            sig = 0.05 * np.mean(ranges)
            lam = base_lam
        
        time_frac = min(0.3, max(0.08, remaining() * 0.3))
        deadline = elapsed() + time_frac
        deadline = min(deadline, max_time * 0.85)
        
        run_cmaes(sp, sig, deadline, lam)
        
        # Update top points
        if len(all_fits) > len(all_fits_arr):
            all_fits_arr = np.array(all_fits)
            all_points_arr = np.array(all_points)
            top_k = min(10, len(all_fits_arr))
            top_indices = np.argsort(all_fits_arr)[:top_k]
        
        restart += 1

    # === Phase 3: Powell-like coordinate descent ===
    if best_params is not None and elapsed() < max_time * 0.93:
        x = best_params.copy()
        fx = best
        step_sizes = 0.001 * ranges.copy()
        
        while elapsed() < max_time * 0.93:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.93:
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step_sizes[d], lower[d], upper[d])
                    fn = eval_f(xn)
                    if fn < fx:
                        # Accelerated step
                        while elapsed() < max_time * 0.93:
                            step_sizes[d] *= 1.5
                            xn2 = x.copy()
                            xn2[d] = np.clip(xn[d] + sign * step_sizes[d], lower[d], upper[d])
                            fn2 = eval_f(xn2)
                            if fn2 < fn:
                                xn, fn = xn2, fn2
                            else:
                                break
                        x, fx = xn, fn
                        improved = True
                        break
            if not improved:
                step_sizes *= 0.5
                if np.max(step_sizes / ranges) < 1e-15:
                    break

    # === Phase 4: Nelder-Mead refinement ===
    if best_params is not None and elapsed() < max_time * 0.98:
        n = dim
        scale = 0.0005 * ranges
        simplex = np.empty((n+1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i]
            simplex[i+1] = np.clip(simplex[i+1], lower, upper)
        
        f_simplex = np.empty(n+1)
        for i in range(n+1):
            if elapsed() >= max_time * 0.98:
                break
            f_simplex[i] = eval_f(simplex[i])
        else:
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            while elapsed() < max_time * 0.997:
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
                fr = eval_f(xr)
                
                if fr < f_simplex[0]:
                    xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1], f_simplex[-1] = xe, fe
                    else:
                        simplex[-1], f_simplex[-1] = xr, fr
                elif fr < f_simplex[-2]:
                    simplex[-1], f_simplex[-1] = xr, fr
                else:
                    if fr < f_simplex[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                        fc = eval_f(xc)
                        if fc <= fr:
                            simplex[-1], f_simplex[-1] = xc, fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                                if elapsed() >= max_time * 0.997:
                                    break
                    else:
                        xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                        fc = eval_f(xc)
                        if fc < f_simplex[-1]:
                            simplex[-1], f_simplex[-1] = xc, fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                                if elapsed() >= max_time * 0.997:
                                    break

    # === Phase 5: Final micro-perturbations ===
    while elapsed() < max_time * 0.999:
        sc = np.random.choice([1e-4, 1e-5, 1e-6, 1e-7])
        candidate = best_params + sc * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best
#
#Key improvements:
#1. **Removed DE phase** - replaced with more CMA-ES restarts and coordinate descent, which are more efficient
#2. **Added opposition-based learning** in initialization for better initial coverage
#3. **Added Powell-like coordinate descent** with acceleration - very effective for separable/partially separable functions
#4. **Better CMA-ES restart strategy** with 5 different restart types including midpoint crossover of good solutions
#5. **Smarter time budgeting** - more time for CMA-ES restarts, added coordinate descent phase
#6. **Track all evaluated points** to seed restarts from diverse good solutions
#7. **Dynamic restart time allocation** based on remaining time
