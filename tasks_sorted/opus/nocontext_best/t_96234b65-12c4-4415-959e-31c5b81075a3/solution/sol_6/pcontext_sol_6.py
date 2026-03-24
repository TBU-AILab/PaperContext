#Looking at the current algorithm, I see several areas for improvement:
#
#1. **SHADE is slow per iteration** - the inner loop with archive management is costly
#2. **CMA-ES restarts could be smarter** - using information from previous runs
#3. **Time allocation could be more adaptive** - spending time where progress is made
#4. **Missing a good surrogate or model-based component**
#5. **The directional search is basic** - could use Powell's method instead
#
#Let me redesign with a tighter, more efficient approach:
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
    
    # --- LHS initialization ---
    n_init = min(max(15 * dim, 80), 400)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.1:
            n_init = i
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    init_pop = init_pop[:n_init]
    init_fitness = init_fitness[:n_init]
    if n_init == 0:
        return best
    
    sorted_idx = np.argsort(init_fitness)
    
    # Collect all evaluated points for surrogate use
    all_X = list(init_pop)
    all_F = list(init_fitness)

    # --- CMA-ES with restarts ---
    def cmaes_search(start_point, initial_sigma, budget_seconds, lam_mult=1.0):
        nonlocal best, best_params
        
        cma_start = datetime.now()
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
        
        stag = 0
        best_local = float('inf')
        prev_best = float('inf')
        gen = 0
        
        while True:
            t_left = budget_seconds - (datetime.now() - cma_start).total_seconds()
            if t_left <= 0:
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
                if (datetime.now() - cma_start).total_seconds() >= budget_seconds:
                    return best_local
                fitness[k] = eval_func(arx[k])
                all_X.append(arx[k].copy())
                all_F.append(fitness[k])
            
            order = np.argsort(fitness)
            arx = arx[order]; arz = arz[order]; fitness = fitness[order]
            
            if fitness[0] < best_local:
                best_local = fitness[0]
            
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
                C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (weights[:, None] * artmp).T @ artmp
                C = np.triu(C) + np.triu(C, 1).T
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                diag_C = (1 - c1 - cmu_val) * diag_C + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C) + cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges) * 2)
            
            gen += 1
            
            if best_local < prev_best - 1e-13:
                stag = 0
                prev_best = best_local
            else:
                stag += 1
            
            cond = np.max(D) / (np.min(D) + 1e-30) if use_full else np.sqrt(np.max(diag_C) / (np.min(diag_C) + 1e-30))
            
            if sigma < 1e-15 or stag > 20 + 10 * n or cond > 1e14:
                return best_local
        
        return best_local

    # --- SHADE with linear pop reduction (L-SHADE) ---
    def lshade_search(budget_seconds):
        nonlocal best, best_params
        
        de_start = datetime.now()
        pop_size_init = min(max(8 * dim, 40), 200)
        pop_size_min = max(4, dim // 2)
        pop_size = pop_size_init
        H = 50
        max_nfe_est = int(budget_seconds * 800)  # rough estimate
        nfe = 0
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        pop = np.random.uniform(lower, upper, (pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        n_seed = min(pop_size // 2, n_init)
        for i in range(n_seed):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        
        if best_params is not None:
            pop[0] = best_params.copy()
            fit[0] = best
        
        for i in range(n_seed, pop_size):
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            fit[i] = eval_func(pop[i])
            nfe += 1
        
        archive = []
        max_archive = pop_size_init
        
        gen = 0
        while True:
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            
            S_F = []; S_CR = []; S_delta = []
            
            sort_idx = np.argsort(fit[:pop_size])
            
            trial_pop = np.empty((pop_size, dim))
            trial_fit = np.full(pop_size, float('inf'))
            trial_params = []  # (i, Fi, CRi)
            
            for i in range(pop_size):
                if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                    return
                
                ri = np.random.randint(H)
                
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi >= 1.0:
                        Fi = 1.0; break
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                p = max(2, int(np.random.uniform(0.05, 0.2) * pop_size))
                pbest_idx = sort_idx[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                combined_pop_arr = np.vstack([pop[:pop_size]] + ([np.array(archive)] if archive else []))
                
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
                trial = np.clip(trial, lower, upper)
                
                f_trial = eval_func(trial)
                nfe += 1
                all_X.append(trial.copy())
                all_F.append(f_trial)
                
                if f_trial < fit[i]:
                    delta = fit[i] - f_trial
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial; fit[i] = f_trial
                elif f_trial == fit[i]:
                    pop[i] = trial; fit[i] = f_trial
            
            if S_F:
                S_delta = np.array(S_delta)
                w = S_delta / (np.sum(S_delta) + 1e-30)
                S_F = np.array(S_F); S_CR = np.array(S_CR)
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
            # Linear population reduction
            ratio = nfe / max(max_nfe_est, nfe + 1)
            new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
            if new_pop_size < pop_size:
                order = np.argsort(fit[:pop_size])
                pop_tmp = pop[order[:new_pop_size]].copy()
                fit_tmp = fit[order[:new_pop_size]].copy()
                pop = np.empty((pop_size_init, dim))  # keep array big enough
                fit = np.full(pop_size_init, float('inf'))
                pop[:new_pop_size] = pop_tmp
                fit[:new_pop_size] = fit_tmp
                pop_size = new_pop_size
            
            gen += 1

    # --- Powell's conjugate direction method ---
    def powell_search(start_point, budget_seconds):
        nonlocal best, best_params
        pw_start = datetime.now()
        n = dim
        x = start_point.copy()
        fx = eval_func(x)
        
        directions = np.eye(n)
        
        for outer in range(1000):
            if (datetime.now() - pw_start).total_seconds() >= budget_seconds:
                return
            
            x_start = x.copy()
            fx_start = fx
            biggest_decrease = 0
            biggest_idx = 0
            
            for i in range(n):
                if (datetime.now() - pw_start).total_seconds() >= budget_seconds:
                    return
                
                d = directions[i]
                # Line search along d
                x, fx, dec = _line_search(x, fx, d, pw_start, budget_seconds)
                if dec > biggest_decrease:
                    biggest_decrease = dec
                    biggest_idx = i
            
            improvement = fx_start - fx
            if improvement < 1e-14 * (abs(fx) + 1e-30):
                return
            
            # Update directions
            new_dir = x - x_start
            norm = np.linalg.norm(new_dir)
            if norm > 1e-20:
                new_dir /= norm
                directions = np.delete(directions, biggest_idx, axis=0)
                directions = np.vstack([directions, new_dir])
                # One more line search along new direction
                x, fx, _ = _line_search(x, fx, new_dir, pw_start, budget_seconds)
    
    def _line_search(x, fx, d, ref_start, budget_seconds):
        """Golden section line search"""
        # Find bracket
        step = np.mean(ranges) * 0.01
        decrease = 0
        
        # First try to find improvement direction
        x_pos = np.clip(x + step * d, lower, upper)
        if (datetime.now() - ref_start).total_seconds() >= budget_seconds:
            return x, fx, 0
        f_pos = eval_func(x_pos)
        
        x_neg = np.clip(x - step * d, lower, upper)
        if (datetime.now() - ref_start).total_seconds() >= budget_seconds:
            if f_pos < fx:
                return x_pos, f_pos, fx - f_pos
            return x, fx, 0
        f_neg = eval_func(x_neg)
        
        if f_pos >= fx and f_neg >= fx:
            # Try smaller step
            step *= 0.1
            x_pos = np.clip(x + step * d, lower, upper)
            f_pos = eval_func(x_pos)
            x_neg = np.clip(x - step * d, lower, upper)
            f_neg = eval_func(x_neg)
            if f_pos >= fx and f_neg >= fx:
                return x, fx, 0
        
        if f_neg < f_pos:
            d = -d
            f_pos = f_neg
        
        # Expanding search
        best_x = x.copy()
        best_f = fx
        
        if f_pos < best_f:
            best_x = np.clip(x + step * d, lower, upper)
            best_f = f_pos
        
        for _ in range(25):
            if (datetime.now() - ref_start).total_seconds() >= budget_seconds:
                break
            step *= 2.0
            x_try = np.clip(x + step * d, lower, upper)
            f_try = eval_func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
            else:
                break
        
        # Golden section refinement in [0, step]
        a = 0.0
        b = step
        gr = (np.sqrt(5) + 1) / 2
        
        for _ in range(20):
            if (datetime.now() - ref_start).total_seconds() >= budget_seconds:
                break
            if b - a < 1e-15:
                break
            c = b - (b - a) / gr
            dd = a + (b - a) / gr
            
            xc = np.clip(x + c * d, lower, upper)
            xd = np.clip(x + dd * d, lower, upper)
            fc = eval_func(xc)
            fd = eval_func(xd)
            
            if fc < best_f:
                best_f = fc; best_x = xc.copy()
            if fd < best_f:
                best_f = fd; best_x = xd.copy()
            
            if fc < fd:
                b = dd
            else:
                a = c
        
        decrease = fx - best_f
        if decrease > 0:
            return best_x, best_f, decrease
        return x, fx, 0

    # --- Nelder-Mead ---
    def nelder_mead(start_point, budget_seconds, scale_factor=0.05):
        nonlocal best, best_params
        nm_start = datetime.now()
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
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            f_values[i + 1] = eval_func(simplex[i + 1])
        
        no_improve = 0
        for _ in range(500000):
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            
            order = np.argsort(f_values)
            simplex = simplex[order]; f_values = f_values[order]
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
            
            if no_improve > 50 * n:
                return
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                return

    # === Main orchestration ===
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Phase 1: L-SHADE (30% of time)
    remaining = time_left()
    if remaining > 0.5:
        lshade_search(remaining * 0.30)
    
    # Phase 2: CMA-ES with IPOP restarts (35% of time)
    remaining = time_left()
    if remaining > 0.5:
        cma_budget = remaining * 0.40
        cma_start_time = elapsed()
        restart_count = 0
        pop_mult = 1.0
        while elapsed() - cma_start_time < cma_budget:
            budget_this = cma_budget - (elapsed() - cma_start_time)
            if budget_this < 0.2:
                break
            budget_this = min(budget_this, max(cma_budget / 3, 1.0))
            
            if restart_count == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.15
            elif restart_count == 1 and best_params is not None:
                sp = best_params + np.random.randn(dim) * ranges * 0.25
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
            
            cmaes_search(sp, sig, budget_this, pop_mult)
            restart_count += 1
            pop_mult = min(pop_mult * 1.5, 5)
    
    # Phase 3: Powell's method from best
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.35)
    
    # Phase 4: Nelder-Mead refinement
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.4, scale_factor=0.02)
    
    # Phase 5: Another CMA-ES from best with small sigma
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        cmaes_search(best_params.copy(), np.mean(ranges) * 0.05, remaining * 0.4)
    
    # Phase 6: Final Powell refinement  
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.5)
    
    # Phase 7: Final NM with tiny simplex
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.8, scale_factor=0.001)
    
    return best
#
#Key improvements:
#1. **L-SHADE** (linear population size reduction) replaces basic SHADE - better convergence
#2. **Powell's conjugate direction method** replaces basic directional search - much more efficient for local optimization
#3. **Golden section line search** within Powell for precise 1D optimization
#4. **Better phase structure** - alternates between global (CMA-ES) and local (Powell, NM) methods
#5. **Late-stage CMA-ES** with small sigma for fine-tuning around best solution
#6. **More restarts with diverse strategies** in CMA-ES IPOP
#7. **Collecting all evaluations** for potential future surrogate use
