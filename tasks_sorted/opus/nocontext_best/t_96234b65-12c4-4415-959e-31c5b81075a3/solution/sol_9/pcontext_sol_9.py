#Looking at the current algorithm, I need to analyze what's working and what isn't. The score of 27.38 suggests the algorithm is getting stuck in local optima or not exploring efficiently enough. Let me redesign with better strategies:
#
#1. **Better global search** with multi-strategy DE and restart mechanisms
#2. **Improved CMA-ES** with better parameter adaptation
#3. **More aggressive local search refinement**
#4. **Smarter time allocation based on dimension**
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
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Sobol-like quasi-random initialization ---
    def quasi_random_init(n):
        pop = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n * ranges[d]
        return pop

    # --- CMA-ES with restarts ---
    def cmaes(start_point, initial_sigma, budget_seconds, lam_mult=1.0):
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
        
        best_local = float('inf')
        best_local_x = start_point.copy()
        stag_count = 0
        gen = 0
        
        while True:
            if elapsed() - cma_start >= budget_seconds or time_left() <= 0.1:
                return best_local, best_local_x
            
            if use_full and eigen_countdown <= 0:
                try:
                    C = (C + C.T) / 2
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)
                    B = eigenvectors
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
                eigen_countdown = max(1, int(1.0 / (c1 + cmu_val + 1e-30) / n / 10))
            
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
                if elapsed() - cma_start >= budget_seconds or time_left() <= 0.05:
                    return best_local, best_local_x
                fitness[k] = eval_func(arx[k])
            
            order = np.argsort(fitness)
            arx = arx[order]
            arz = arz[order]
            fitness = fitness[order]
            
            if fitness[0] < best_local:
                best_local = fitness[0]
                best_local_x = arx[0].copy()
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (diff / np.sqrt(np.maximum(diag_C, 1e-20)))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            if use_full:
                artmp = (arx[:mu] - old_mean) / (sigma + 1e-30)
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (weights[:, None] * artmp).T @ artmp
            else:
                artmp = (arx[:mu] - old_mean) / (sigma + 1e-30)
                diag_C = (1 - c1 - cmu_val) * diag_C + \
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C) + \
                         cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges) * 2)
            
            gen += 1
            
            cond = np.max(D) / (np.min(D) + 1e-30) if use_full else np.sqrt(np.max(diag_C) / (np.min(diag_C) + 1e-30))
            if sigma < 1e-16 or stag_count > 20 + 10 * n or cond > 1e14:
                return best_local, best_local_x
        
        return best_local, best_local_x

    # --- L-SHADE ---
    def lshade(budget_seconds, pop_size=None):
        nonlocal best, best_params
        de_start = elapsed()
        
        N_init = pop_size if pop_size else min(max(8 * dim, 40), 200)
        N_min = max(4, dim)
        H = 50
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        pop = quasi_random_init(N_init)
        fit = np.full(N_init, float('inf'))
        
        # Seed with best known points
        if best_params is not None:
            pop[0] = best_params.copy()
        
        for i in range(N_init):
            if elapsed() - de_start >= budget_seconds or time_left() <= 0.05:
                return
            fit[i] = eval_func(pop[i])
        
        archive = []
        max_archive = N_init
        max_evals_estimate = int(budget_seconds * 1000)  # rough estimate
        evals_used = N_init
        
        gen = 0
        while True:
            if elapsed() - de_start >= budget_seconds or time_left() <= 0.05:
                return
            
            N_current = len(pop)
            S_F, S_CR, S_delta = [], [], []
            
            sort_idx = np.argsort(fit)
            
            trial_pop = np.empty_like(pop)
            trial_fit = np.full(N_current, float('inf'))
            
            for i in range(N_current):
                if elapsed() - de_start >= budget_seconds or time_left() <= 0.05:
                    return
                
                ri = np.random.randint(H)
                
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi >= 1.0:
                        Fi = 1.0
                        break
                
                CRi = M_CR[ri]
                if CRi < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.normal(CRi, 0.1), 0, 1)
                
                p = max(2, int(np.random.uniform(0.05, 0.2) * N_current))
                pbest_idx = sort_idx[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(N_current))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                all_pool = list(range(N_current)) + list(range(N_current, N_current + len(archive)))
                all_pool = [x for x in all_pool if x != i and x != r1]
                if not all_pool:
                    all_pool = [j for j in range(N_current) if j != i]
                r2 = np.random.choice(all_pool)
                if r2 < N_current:
                    xr2 = pop[r2]
                else:
                    xr2 = archive[r2 - N_current]
                
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
                trial = np.clip(trial, lower, upper)
                
                f_trial = eval_func(trial)
                evals_used += 1
                
                if f_trial < fit[i]:
                    delta = fit[i] - f_trial
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial
                    fit[i] = f_trial
                elif f_trial == fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
            
            if len(S_F) > 0:
                S_delta = np.array(S_delta)
                w = S_delta / (np.sum(S_delta) + 1e-30)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                mean_F = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                mean_CR = np.sum(w * S_CR)
                M_F[k] = mean_F
                M_CR[k] = mean_CR
                k = (k + 1) % H
            
            # Linear population size reduction
            ratio = min(evals_used / max(max_evals_estimate, 1), 1.0)
            new_N = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            if new_N < N_current:
                worst_idx = np.argsort(fit)[-int(N_current - new_N):]
                keep = np.ones(N_current, dtype=bool)
                keep[worst_idx] = False
                pop = pop[keep]
                fit = fit[keep]
            
            gen += 1

    # --- Nelder-Mead ---
    def nelder_mead(start_point, budget_seconds, initial_scale=0.05):
        nonlocal best, best_params
        nm_start = elapsed()
        n = dim
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = start_point.copy()
        f_values = np.full(n + 1, float('inf'))
        f_values[0] = eval_func(simplex[0])
        
        scale = ranges * initial_scale
        for i in range(n):
            if elapsed() - nm_start >= budget_seconds or time_left() <= 0.05:
                return
            simplex[i + 1] = start_point.copy()
            delta = scale[i] if abs(scale[i]) > 1e-15 else 0.01
            simplex[i + 1][i] += delta
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            f_values[i + 1] = eval_func(simplex[i + 1])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        no_improve = 0
        
        for _ in range(500000):
            if elapsed() - nm_start >= budget_seconds or time_left() <= 0.05:
                return
            
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            best_before = f_values[0]
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
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
            
            if f_values[0] < best_before - 1e-14 * (abs(best_before) + 1):
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 30 * n:
                return
            
            diam = np.max(np.abs(simplex[-1] - simplex[0]))
            if diam < 1e-15:
                return

    # --- Powell's conjugate direction method ---
    def powell_search(start_point, budget_seconds):
        nonlocal best, best_params
        ps_start = elapsed()
        n = dim
        x = start_point.copy()
        fx = eval_func(x)
        
        directions = np.eye(n)
        
        for outer in range(100):
            if elapsed() - ps_start >= budget_seconds or time_left() <= 0.05:
                return
            
            x_start = x.copy()
            fx_start = fx
            biggest_decrease = 0
            biggest_idx = 0
            
            for i in range(n):
                if elapsed() - ps_start >= budget_seconds or time_left() <= 0.05:
                    return
                
                d = directions[i]
                fx_before = fx
                
                # Line search along direction d using golden section
                x, fx = golden_line_search(x, fx, d, ps_start, budget_seconds)
                
                decrease = fx_before - fx
                if decrease > biggest_decrease:
                    biggest_decrease = decrease
                    biggest_idx = i
            
            improvement = fx_start - fx
            if improvement < 1e-14 * (abs(fx_start) + 1):
                return
            
            # Update directions
            new_dir = x - x_start
            norm = np.linalg.norm(new_dir)
            if norm > 1e-20:
                new_dir /= norm
                directions = np.delete(directions, biggest_idx, axis=0)
                directions = np.vstack([directions, new_dir])
    
    def golden_line_search(x, fx, d, search_start, budget):
        gr = (np.sqrt(5) + 1) / 2
        
        # Find bracket
        step = np.mean(ranges) * 0.01
        a, b = 0, step
        
        xa = x
        fa = fx
        xb = np.clip(x + b * d, lower, upper)
        
        if elapsed() - search_start >= budget or time_left() <= 0.05:
            return x, fx
        fb = eval_func(xb)
        
        if fb < fa:
            # Expand
            for _ in range(20):
                b *= 2
                xb = np.clip(x + b * d, lower, upper)
                if elapsed() - search_start >= budget or time_left() <= 0.05:
                    break
                fb_new = eval_func(xb)
                if fb_new >= fb:
                    break
                fb = fb_new
        else:
            # Try negative
            b = -step
            xb = np.clip(x + b * d, lower, upper)
            if elapsed() - search_start >= budget or time_left() <= 0.05:
                return x, fx
            fb = eval_func(xb)
            if fb < fa:
                for _ in range(20):
                    b *= 2
                    xb = np.clip(x + b * d, lower, upper)
                    if elapsed() - search_start >= budget or time_left() <= 0.05:
                        break
                    fb_new = eval_func(xb)
                    if fb_new >= fb:
                        break
                    fb = fb_new
            else:
                return x, fx
        
        lo, hi = min(0, b), max(0, b)
        
        # Golden section
        for _ in range(30):
            if elapsed() - search_start >= budget or time_left() <= 0.05:
                break
            if abs(hi - lo) < 1e-15:
                break
            
            c = hi - (hi - lo) / gr
            dd = lo + (hi - lo) / gr
            
            xc = np.clip(x + c * d, lower, upper)
            fc = eval_func(xc)
            xd = np.clip(x + dd * d, lower, upper)
            fd = eval_func(xd)
            
            if fc < fd:
                hi = dd
            else:
                lo = c
        
        best_t = (lo + hi) / 2
        x_best = np.clip(x + best_t * d, lower, upper)
        f_best = eval_func(x_best)
        
        if f_best < fx:
            return x_best, f_best
        return x, fx

    # ===================== MAIN ORCHESTRATION =====================
    
    # Phase 0: Quick initial sampling
    remaining = time_left()
    n_init = min(max(15 * dim, 80), 400)
    init_pop = quasi_random_init(n_init)
    init_fit = np.full(n_init, float('inf'))
    
    for i in range(n_init):
        if time_left() < max_time * 0.85:
            n_init = i
            break
        init_fit[i] = eval_func(init_pop[i])
    
    # Phase 1: L-SHADE (40% of time)
    remaining = time_left()
    if remaining > 1.0:
        lshade(remaining * 0.40)
    
    # Phase 2: IPOP-CMA-ES (30% of time)
    remaining = time_left()
    if remaining > 0.5:
        cma_budget = remaining * 0.35
        cma_phase_start = elapsed()
        restart = 0
        pop_mult = 1.0
        
        while elapsed() - cma_phase_start < cma_budget and time_left() > 0.2:
            this_budget = min(cma_budget - (elapsed() - cma_phase_start), cma_budget / max(2, 4 - restart))
            if this_budget < 0.15:
                break
            
            if restart == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.15
            elif restart == 1 and best_params is not None:
                sp = best_params + np.random.randn(dim) * ranges * 0.2
                sp = np.clip(sp, lower, upper)
                sig = np.mean(ranges) * 0.25
            else:
                if best_params is not None and np.random.random() < 0.4:
                    sp = best_params + np.random.randn(dim) * ranges * 0.4
                    sp = np.clip(sp, lower, upper)
                    sig = np.mean(ranges) * 0.3
                else:
                    sp = np.random.uniform(lower, upper)
                    sig = np.mean(ranges) * 0.4
            
            cmaes(sp, sig, this_budget, pop_mult)
            restart += 1
            pop_mult = min(pop_mult * 1.5, 5)
    
    # Phase 3: Powell's method
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.35)
    
    # Phase 4: Nelder-Mead refinement
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.4, initial_scale=0.02)
    
    # Phase 5: Fine CMA-ES
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        cmaes(best_params.copy(), np.mean(ranges) * 0.01, remaining * 0.4, 1.0)
    
    # Phase 6: Final Nelder-Mead with tiny simplex
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.5, initial_scale=0.001)
    
    # Phase 7: Final Powell
    remaining = time_left()
    if remaining > 0.1 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.9)
    
    return best
#
#Key improvements:
#1. **L-SHADE with linear population reduction** - more efficient population management over time
#2. **Powell's conjugate direction method** - very effective for local optimization, systematically searches along conjugate directions
#3. **Golden section line search** - precise 1D optimization within Powell's method
#4. **Better IPOP restart strategy** - progressive population increase with diverse starting points
#5. **More refinement phases** - fine CMA-ES + multiple NM + Powell for thorough local exploitation
#6. **Removed directional search** in favor of the more principled Powell's method
#7. **Better time allocation** - 40% global (L-SHADE), 35% semi-global (CMA-ES), 25% local refinement
