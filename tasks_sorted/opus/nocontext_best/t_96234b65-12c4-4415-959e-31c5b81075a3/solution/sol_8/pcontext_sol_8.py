#Looking at the previous algorithm, I can see several areas for improvement:
#
#1. **Time allocation** could be better balanced
#2. **CMA-ES restarts** need better diversity
#3. **SHADE** could benefit from linear population size reduction (L-SHADE)
#4. **Missing surrogate-assisted local search** for expensive functions
#5. **Better coordination between phases**
#
#Here's my improved version:
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
    
    eval_count = 0
    all_X = []
    all_F = []
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params, eval_count
        x = np.clip(x, lower, upper)
        eval_count += 1
        f = func(x)
        if len(all_X) < 50000:
            all_X.append(x.copy())
            all_F.append(f)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def normalize(x):
        return (x - lower) / (ranges + 1e-30)
    
    def denormalize(x_norm):
        return lower + x_norm * ranges

    # --- Sobol-like quasi-random initialization ---
    def quasi_random_init(n_points):
        points = np.zeros((n_points, dim))
        for i in range(n_points):
            for d in range(dim):
                # Use a simple additive recurrence (low-discrepancy)
                golden = (1 + np.sqrt(5)) / 2
                alpha_d = 1.0 / (golden ** (d + 1))
                points[i, d] = lower[d] + ((0.5 + i * alpha_d) % 1.0) * ranges[d]
        return points

    # --- LHS initialization ---
    n_init = min(max(15 * dim, 80), 400)
    
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    # Add some quasi-random points
    n_quasi = min(n_init // 3, 100)
    quasi_pts = quasi_random_init(n_quasi)
    init_pop[:n_quasi] = quasi_pts
    
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

    # --- CMA-ES with sep-CMA for high dim ---
    def cmaes_search(start_point, initial_sigma, budget_seconds, pop_mult=1.0):
        nonlocal best, best_params
        
        cma_start = datetime.now()
        n = dim
        
        lam = max(int((4 + int(3 * np.log(n))) * pop_mult), 6)
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
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
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
            
            # Stagnation
            if gen - best_local_gen > 20 + 10 * n:
                return best_local
            
            cond = np.max(D) / (np.min(D) + 1e-30) if use_full else np.sqrt(np.max(diag_C) / (np.min(diag_C) + 1e-30))
            
            if sigma < 1e-15 or cond > 1e14:
                return best_local
        
        return best_local
    
    # --- L-SHADE ---
    def lshade_search(budget_seconds):
        nonlocal best, best_params
        
        de_start = datetime.now()
        pop_init = min(max(8 * dim, 40), 200)
        pop_min = max(4, dim // 2)
        pop_size = pop_init
        H = 100
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        pop = np.random.uniform(lower, upper, (pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        n_seed = min(pop_size // 2, n_init)
        for i in range(n_seed):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        
        if best_params is not None and n_seed > 0:
            pop[0] = best_params.copy()
            fit[0] = best
        
        for i in range(n_seed, pop_size):
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            fit[i] = eval_func(pop[i])
        
        archive = []
        max_archive = pop_size
        max_evals_estimate = int(budget_seconds * eval_count / max(elapsed(), 0.01)) if elapsed() > 0.01 else 100000
        evals_used = pop_size
        
        gen = 0
        while True:
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            p_best_rate = max(2.0 / pop_size, 0.05 + 0.15 * (1 - evals_used / max(max_evals_estimate, 1)))
            p = max(2, int(p_best_rate * pop_size))
            
            sort_idx = np.argsort(fit)
            
            trial_pop = pop.copy()
            trial_fit = np.full(pop_size, float('inf'))
            Fs = np.zeros(pop_size)
            CRs = np.zeros(pop_size)
            
            for i in range(pop_size):
                ri = np.random.randint(H)
                
                Fi = -1
                attempts = 0
                while Fi <= 0 and attempts < 20:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    attempts += 1
                Fi = np.clip(Fi, 0.01, 1.0)
                
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                Fs[i] = Fi
                CRs[i] = CRi
                
                pbest_idx = sort_idx[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                combined_pop_arr = np.vstack([pop] + ([np.array(archive)] if archive else [])) if archive else pop
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
                if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                    return
                trial_fit[i] = eval_func(trial_pop[i])
                evals_used += 1
            
            for i in range(pop_size):
                if trial_fit[i] < fit[i]:
                    delta = fit[i] - trial_fit[i]
                    S_F.append(Fs[i])
                    S_CR.append(CRs[i])
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
                if np.max(S_CR_arr) == 0:
                    M_CR[k] = -1
                else:
                    M_CR[k] = np.sum(w * S_CR_arr)
                k = (k + 1) % H
            
            # Linear population size reduction
            progress = min(evals_used / max(max_evals_estimate, 1), 1.0)
            new_pop_size = max(pop_min, int(round(pop_init + (pop_min - pop_init) * progress)))
            if new_pop_size < pop_size:
                worst_idx = np.argsort(fit)[-1:-(pop_size - new_pop_size + 1):-1]
                keep = np.setdiff1d(np.arange(pop_size), worst_idx)
                pop = pop[keep]
                fit = fit[keep]
                pop_size = len(pop)
                max_archive = pop_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
            
            gen += 1
    
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
            
            if no_improve > 40 * n:
                return
            
            diam = np.max(np.abs(simplex[-1] - simplex[0]))
            if diam < 1e-15:
                return

    # --- Powell-like coordinate descent with quadratic interpolation ---
    def powell_search(start_point, budget_seconds):
        nonlocal best, best_params
        ps_start = datetime.now()
        
        x = start_point.copy()
        fx = eval_func(x)
        
        directions = np.eye(dim)
        
        for outer in range(100):
            if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                return
            
            x_start = x.copy()
            fx_start = fx
            best_delta = 0
            best_dir_idx = 0
            
            for di in range(dim):
                if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                    return
                
                direction = directions[di]
                fx_before = fx
                
                # Golden section line search
                a_min = -np.max(ranges)
                a_max = np.max(ranges)
                
                # First find a bracket
                step = np.mean(ranges) * 0.01
                best_alpha = 0
                best_f_line = fx
                
                for sign in [1, -1]:
                    alpha = step * sign
                    for _ in range(30):
                        x_try = np.clip(x + alpha * direction, lower, upper)
                        f_try = eval_func(x_try)
                        if f_try < best_f_line:
                            best_f_line = f_try
                            best_alpha = alpha
                            alpha *= 2.0
                        else:
                            break
                        if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                            if best_f_line < fx:
                                x = np.clip(x + best_alpha * direction, lower, upper)
                                fx = best_f_line
                            return
                
                if best_f_line < fx:
                    x = np.clip(x + best_alpha * direction, lower, upper)
                    fx = best_f_line
                
                delta = fx_before - fx
                if delta > best_delta:
                    best_delta = delta
                    best_dir_idx = di
            
            improvement = fx_start - fx
            if improvement < 1e-14 * (abs(fx) + 1e-30):
                # Try random rotation of directions
                Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                directions = Q
                if outer > 3:
                    return
            else:
                # Update directions: add extrapolated direction
                new_dir = x - x_start
                norm = np.linalg.norm(new_dir)
                if norm > 1e-15:
                    new_dir /= norm
                    directions[best_dir_idx] = new_dir

    # --- Adaptive local search with momentum ---
    def momentum_search(start_point, budget_seconds):
        nonlocal best, best_params
        ms_start = datetime.now()
        
        x = start_point.copy()
        fx = eval_func(x)
        velocity = np.zeros(dim)
        lr = np.mean(ranges) * 0.01
        beta = 0.8
        
        no_improve = 0
        for _ in range(100000):
            if (datetime.now() - ms_start).total_seconds() >= budget_seconds:
                return
            
            # Estimate gradient with finite differences (2-point)
            grad = np.zeros(dim)
            h = lr * 0.01
            for d in range(dim):
                if (datetime.now() - ms_start).total_seconds() >= budget_seconds:
                    return
                e = np.zeros(dim)
                e[d] = h
                fp = eval_func(np.clip(x + e, lower, upper))
                fm = eval_func(np.clip(x - e, lower, upper))
                grad[d] = (fp - fm) / (2 * h)
            
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-30:
                grad /= gnorm
            
            velocity = beta * velocity - lr * grad
            x_new = np.clip(x + velocity, lower, upper)
            fx_new = eval_func(x_new)
            
            if fx_new < fx:
                x = x_new
                fx = fx_new
                no_improve = 0
                lr *= 1.1
            else:
                no_improve += 1
                lr *= 0.5
                velocity *= 0.5
            
            if lr < 1e-15 or no_improve > 10:
                return

    # === Main orchestration ===
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Phase 1: L-SHADE (30% of remaining time)
    remaining = time_left()
    if remaining > 0.5:
        lshade_search(remaining * 0.30)
    
    # Phase 2: IPOP-CMA-ES restarts (35% of remaining time)
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
                    sp = best_params + np.random.randn(dim) * ranges * (0.1 + 0.3 * np.random.random())
                    sp = np.clip(sp, lower, upper)
                    sig = np.mean(ranges) * (0.1 + 0.3 * np.random.random())
                else:
                    sp = np.random.uniform(lower, upper)
                    sig = np.mean(ranges) * 0.4
            
            cmaes_search(sp, sig, budget_this, pop_mult)
            restart_count += 1
            pop_mult = min(pop_mult * 1.5, 5)
    
    # Phase 3: Powell search
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.3)
    
    # Phase 4: Nelder-Mead from best
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.35, scale_factor=0.03)
    
    # Phase 5: Momentum-based gradient descent
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        momentum_search(best_params.copy(), remaining * 0.4)
    
    # Phase 6: Fine Nelder-Mead
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.5, scale_factor=0.002)
    
    # Phase 7: Very fine Powell
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.8)
    
    # Phase 8: Final NM polish
    remaining = time_left()
    if remaining > 0.1 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.9, scale_factor=0.0005)
    
    return best
#
#Key improvements:
#1. **L-SHADE** with linear population size reduction for better convergence
#2. **Powell's conjugate direction method** with adaptive direction updates and golden section line search
#3. **Quasi-random initialization** using additive recurrence for better space coverage
#4. **Momentum-based gradient descent** for smooth landscapes
#5. **Better time allocation** with more refinement phases
#6. **Improved CMA-ES restart strategy** with varied sigma and starting points
#7. **Multiple polishing phases** with decreasing scale factors
