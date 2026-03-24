#Looking at the previous algorithm, I can see several areas for improvement:
#
#1. **Better global search** - The SHADE implementation could be more efficient
#2. **Smarter restart strategy** - Need better diversity in CMA-ES restarts
#3. **Bayesian-like surrogate elements** - Use collected evaluations to guide search
#4. **More aggressive local search** - Powell-like method for final refinement
#5. **Better time management** - Adaptive phase allocation
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
    
    all_x = []
    all_f = []
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        all_x.append(x.copy())
        all_f.append(f)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Sobol-like quasi-random initialization ---
    def halton_sequence(n_points, n_dim):
        def halton_single(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f /= base
            return result
        
        primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
                   73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,
                   157,163,167,173,179,181,191,193,197,199,211,223,227,229]
        
        points = np.zeros((n_points, n_dim))
        skip = 20  # skip initial correlation
        for j in range(n_dim):
            base = primes[j % len(primes)]
            for i in range(n_points):
                points[i, j] = halton_single(i + skip, base)
        return points
    
    n_init = min(max(15 * dim, 80), 400)
    
    halton_pts = halton_sequence(n_init, dim)
    init_pop = lower + halton_pts * ranges
    
    # Add some random points too
    n_rand = n_init // 5
    rand_pts = np.random.uniform(lower, upper, (n_rand, dim))
    init_pop = np.vstack([init_pop, rand_pts])
    n_total_init = len(init_pop)
    
    init_fitness = np.full(n_total_init, float('inf'))
    for i in range(n_total_init):
        if time_left() < max_time * 0.15:
            n_total_init = i
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    init_pop = init_pop[:n_total_init]
    init_fitness = init_fitness[:n_total_init]
    
    if n_total_init == 0:
        return best
    
    sorted_idx = np.argsort(init_fitness)

    # --- CMA-ES with restarts ---
    def cmaes_search(start_point, initial_sigma, budget_seconds, lam_mult=1.0):
        nonlocal best, best_params
        
        cma_start = elapsed()
        n = dim
        
        lam = max(int((4 + int(3 * np.log(n))) * lam_mult), 6)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        mean = start_point.copy()
        sigma = initial_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = n <= 80
        
        if use_full:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0
        else:
            diag_C = np.ones(n)
        
        gen = 0
        stag = 0
        best_local = float('inf')
        best_local_gen = 0
        
        while True:
            t_used = elapsed() - cma_start
            if t_used >= budget_seconds:
                return best_local
            
            if use_full and eigen_countdown <= 0:
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)
                    B = eigenvectors
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
                eigen_countdown = max(1, int(1.0 / (c1 + cmu_val) / n / 10))
            
            if use_full:
                eigen_countdown -= 1
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                if use_full:
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
                else:
                    arx[k] = mean + sigma * np.sqrt(diag_C) * arz[k]
                arx[k] = np.clip(arx[k], lower, upper)
            
            fitness = np.full(lam, float('inf'))
            for k in range(lam):
                if elapsed() - cma_start >= budget_seconds:
                    return best_local
                fitness[k] = eval_func(arx[k])
            
            order = np.argsort(fitness)
            arx = arx[order]
            arz = arz[order]
            fitness = fitness[order]
            
            if fitness[0] < best_local:
                best_local = fitness[0]
                best_local_gen = gen
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            diff = (mean - old_mean) / sigma
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (diff / np.sqrt(np.maximum(diag_C, 1e-20)))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            if use_full:
                artmp = (arx[:mu] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (weights[:, None] * artmp).T @ artmp
                C = np.triu(C) + np.triu(C, 1).T
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                diag_C = (1 - c1 - cmu_val) * diag_C + \
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C) + \
                         cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges) * 2)
            
            gen += 1
            
            cond = np.max(D) / (np.min(D) + 1e-30) if use_full else np.sqrt(np.max(diag_C) / (np.min(diag_C) + 1e-30))
            
            stag_limit = 20 + 10 * n
            if gen - best_local_gen > stag_limit or sigma < 1e-16 or cond > 1e14:
                return best_local
        
        return best_local

    # --- L-SHADE ---
    def lshade_search(budget_seconds, pop_size=None):
        nonlocal best, best_params
        
        de_start = elapsed()
        if pop_size is None:
            pop_size = min(max(8 * dim, 40), 200)
        
        N_init = pop_size
        N_min = max(4, dim // 2)
        H = 100
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        pop = np.random.uniform(lower, upper, (pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        n_seed = min(pop_size // 2, n_total_init)
        for i in range(n_seed):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        
        if best_params is not None:
            pop[0] = best_params.copy()
            fit[0] = best
        
        for i in range(n_seed, pop_size):
            if elapsed() - de_start >= budget_seconds:
                return
            fit[i] = eval_func(pop[i])
        
        archive = []
        max_archive = pop_size
        
        nfe = pop_size
        max_nfe_estimate = int(budget_seconds * pop_size * 50)  # rough estimate
        
        gen = 0
        while True:
            if elapsed() - de_start >= budget_seconds:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            p_min = max(2, int(0.05 * pop_size))
            p_max = max(p_min, int(0.2 * pop_size))
            
            sort_idx = np.argsort(fit)
            
            trial_pop = np.empty_like(pop)
            trial_fit = np.full(pop_size, float('inf'))
            Fi_arr = np.zeros(pop_size)
            CRi_arr = np.zeros(pop_size)
            
            for i in range(pop_size):
                ri = np.random.randint(H)
                
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi >= 1.0:
                        Fi = 1.0
                        break
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                Fi_arr[i] = Fi
                CRi_arr[i] = CRi
                
                p = np.random.randint(p_min, p_max + 1)
                pbest_idx = sort_idx[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                combined_pop_arr = np.vstack([pop] + ([np.array(archive)] if archive else []))  if archive else pop
                
                r2_candidates = [j for j in range(len(combined_pop_arr)) if j != i and j != r1]
                if not r2_candidates:
                    r2_candidates = [j for j in range(pop_size) if j != i]
                r2 = np.random.choice(r2_candidates)
                xr2 = combined_pop_arr[r2]
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - xr2)
                
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                
                trial_pop[i] = np.clip(trial, lower, upper)
            
            for i in range(pop_size):
                if elapsed() - de_start >= budget_seconds:
                    return
                trial_fit[i] = eval_func(trial_pop[i])
                nfe += 1
            
            for i in range(pop_size):
                if trial_fit[i] < fit[i]:
                    delta = fit[i] - trial_fit[i]
                    S_F.append(Fi_arr[i])
                    S_CR.append(CRi_arr[i])
                    S_delta.append(delta)
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial_pop[i]
                    fit[i] = trial_fit[i]
                elif trial_fit[i] == fit[i]:
                    pop[i] = trial_pop[i]
                    fit[i] = trial_fit[i]
            
            if len(S_F) > 0:
                S_delta_arr = np.array(S_delta)
                w = S_delta_arr / (np.sum(S_delta_arr) + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)
                M_F[k] = np.sum(w * S_F_arr ** 2) / (np.sum(w * S_F_arr) + 1e-30)
                M_CR[k] = np.sum(w * S_CR_arr)
                k = (k + 1) % H
            
            # Linear population size reduction
            ratio = nfe / max(max_nfe_estimate, 1)
            new_pop_size = max(N_min, int(np.round(N_init - (N_init - N_min) * ratio)))
            
            if new_pop_size < pop_size:
                keep_idx = np.argsort(fit)[:new_pop_size]
                pop = pop[keep_idx]
                fit = fit[keep_idx]
                pop_size = new_pop_size
                max_archive = pop_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
            
            gen += 1

    # --- Powell-like coordinate descent with quadratic interpolation ---
    def powell_search(start_point, budget_seconds):
        nonlocal best, best_params
        
        pw_start = elapsed()
        x = start_point.copy()
        fx = eval_func(x)
        
        directions = np.eye(dim)
        
        for outer in range(100):
            if elapsed() - pw_start >= budget_seconds:
                return
            
            improved_any = False
            deltas = np.zeros(dim)
            
            for i in range(dim):
                if elapsed() - pw_start >= budget_seconds:
                    return
                
                d = directions[i]
                
                # Golden section in this direction
                best_alpha = 0
                best_f = fx
                
                # Try a few step sizes
                for step in [ranges[i % dim] * 0.1, ranges[i % dim] * 0.01, ranges[i % dim] * 0.001]:
                    for sign in [1, -1]:
                        x_try = np.clip(x + sign * step * d, lower, upper)
                        f_try = eval_func(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_alpha = sign * step
                
                if best_alpha != 0:
                    # Quadratic interpolation
                    a0 = fx
                    alpha1 = best_alpha
                    f1 = best_f
                    alpha2 = best_alpha * 2
                    x2 = np.clip(x + alpha2 * d, lower, upper)
                    f2 = eval_func(x2)
                    
                    # Fit quadratic through (0, a0), (alpha1, f1), (alpha2, f2)
                    denom = 2 * ((alpha2 - alpha1) * (f1 - a0) - (alpha1) * (f2 - a0))
                    if abs(denom) > 1e-30:
                        alpha_star = ((alpha2**2 - alpha1**2) * (f1 - a0) - (alpha1**2) * (f2 - a0)) / denom
                        # Hmm, simpler: just try midpoints
                    
                    # Refinement: try half step and double step  
                    for mult in [0.5, 1.5, 0.25]:
                        alpha_t = best_alpha * mult
                        x_try = np.clip(x + alpha_t * d, lower, upper)
                        f_try = eval_func(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_alpha = alpha_t
                    
                    if best_f < fx:
                        deltas[i] = fx - best_f
                        x = np.clip(x + best_alpha * d, lower, upper)
                        fx = best_f
                        improved_any = True
            
            if not improved_any:
                # Try random directions
                for _ in range(dim):
                    if elapsed() - pw_start >= budget_seconds:
                        return
                    rd = np.random.randn(dim)
                    rd /= (np.linalg.norm(rd) + 1e-30)
                    
                    for step in [np.mean(ranges) * 0.05, np.mean(ranges) * 0.005]:
                        for sign in [1, -1]:
                            x_try = np.clip(x + sign * step * rd, lower, upper)
                            f_try = eval_func(x_try)
                            if f_try < fx:
                                x = x_try
                                fx = f_try
                                improved_any = True
                                break
                        if improved_any:
                            break
                
                if not improved_any:
                    return
            
            # Update directions (simplified Powell update)
            if np.max(deltas) > 0:
                worst_dir = np.argmax(deltas)
                new_dir = x - start_point
                norm = np.linalg.norm(new_dir)
                if norm > 1e-15:
                    new_dir /= norm
                    directions[worst_dir] = new_dir

    # --- Nelder-Mead ---
    def nelder_mead(start_point, budget_seconds, scale_factor=0.05):
        nonlocal best, best_params
        
        nm_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = start_point.copy()
        f_values = np.full(n + 1, float('inf'))
        f_values[0] = eval_func(simplex[0])
        
        scale = ranges * scale_factor
        for i in range(n):
            simplex[i + 1] = start_point.copy()
            simplex[i + 1][i] += scale[i] if scale[i] > 1e-15 else 0.01
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            if elapsed() - nm_start >= budget_seconds:
                return
            f_values[i + 1] = eval_func(simplex[i + 1])
        
        no_improve = 0
        for _ in range(500000):
            if elapsed() - nm_start >= budget_seconds:
                return
            
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            best_before = f_values[0]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if f_values[0] <= fr < f_values[-2]:
                simplex[-1] = xr; f_values[-1] = fr
            elif fr < f_values[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe; f_values[-1] = fe
                else:
                    simplex[-1] = xr; f_values[-1] = fr
            else:
                if fr < f_values[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc; f_values[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = eval_func(simplex[i])
                else:
                    xcc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fcc = eval_func(xcc)
                    if fcc < f_values[-1]:
                        simplex[-1] = xcc; f_values[-1] = fcc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = eval_func(simplex[i])
            
            if f_values[0] < best_before - 1e-14:
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 30 * n:
                return
            
            diam = np.max(np.abs(simplex[-1] - simplex[0]))
            if diam < 1e-16:
                return

    # === Main orchestration ===
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Phase 1: L-SHADE (30% of time)
    remaining = time_left()
    if remaining > 0.5:
        lshade_search(remaining * 0.30)
    
    # Phase 2: IPOP-CMA-ES (35% of time)
    remaining = time_left()
    if remaining > 0.5:
        cma_budget = remaining * 0.45
        cma_start_time = elapsed()
        restart_count = 0
        lam_mult = 1.0
        
        while elapsed() - cma_start_time < cma_budget:
            budget_this = cma_budget - (elapsed() - cma_start_time)
            if budget_this < 0.3:
                break
            budget_this = min(budget_this, max(cma_budget / 4, 1.0))
            
            if restart_count == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.15
            elif restart_count == 1 and best_params is not None:
                sp = best_params + np.random.randn(dim) * ranges * 0.2
                sp = np.clip(sp, lower, upper)
                sig = np.mean(ranges) * 0.25
            else:
                if best_params is not None and np.random.random() < 0.4:
                    sp = best_params + np.random.randn(dim) * ranges * np.random.uniform(0.1, 0.5)
                    sp = np.clip(sp, lower, upper)
                    sig = np.mean(ranges) * np.random.uniform(0.1, 0.4)
                else:
                    sp = np.random.uniform(lower, upper)
                    sig = np.mean(ranges) * 0.4
            
            cmaes_search(sp, sig, budget_this, lam_mult)
            restart_count += 1
            lam_mult = min(lam_mult * 1.5, 5)
    
    # Phase 3: Powell-like search from best
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.35)
    
    # Phase 4: Nelder-Mead refinement
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.4, scale_factor=0.02)
    
    # Phase 5: Fine CMA-ES
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        cmaes_search(best_params.copy(), np.mean(ranges) * 0.01, remaining * 0.4)
    
    # Phase 6: Very fine Nelder-Mead
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.5, scale_factor=0.001)
    
    # Phase 7: Final Powell refinement
    remaining = time_left()
    if remaining > 0.1 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.9)
    
    return best
