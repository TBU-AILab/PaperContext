#Looking at the current algorithm, I can identify several areas for improvement:
#
#1. **CMA-ES eigendecomposition is too infrequent** and has bugs in the update schedule
#2. **SHADE population is too large** for many problems, wasting evaluations
#3. **Time allocation is suboptimal** - too much time on SHADE, not enough on CMA-ES
#4. **Missing multi-start diversity** - need better exploration
#5. **No population size reduction** (L-SHADE style)
#6. **Nelder-Mead simplex size** should adapt based on current progress
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
    center = (lower + upper) / 2.0
    
    n_evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params, n_evals
        x = np.clip(x, lower, upper)
        f = func(x)
        n_evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Latin Hypercube Sampling initialization
    n_init = min(max(20 * dim, 100), 500)
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
    
    # ---- CMA-ES ----
    def cmaes_search(start_point, initial_sigma, budget_seconds, pop_mult=1):
        nonlocal best, best_params
        
        cma_start = datetime.now()
        n = dim
        
        lam = int((4 + int(3 * np.log(n))) * pop_mult)
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
        
        use_full = n <= 100
        
        if use_full:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0
        else:
            diag_C = np.ones(n)
        
        gen = 0
        stagnation = 0
        prev_median = float('inf')
        flat_count = 0
        best_in_run = float('inf')
        
        while True:
            t_left = budget_seconds - (datetime.now() - cma_start).total_seconds()
            if t_left <= 0:
                return
            
            if use_full and eigen_countdown <= 0:
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)
                    B = eigenvectors
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except np.linalg.LinAlgError:
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
                    return
                fitness[k] = eval_func(arx[k])
            
            order = np.argsort(fitness)
            arx = arx[order]
            arz = arz[order]
            fitness = fitness[order]
            
            if fitness[0] < best_in_run:
                best_in_run = fitness[0]
                stagnation = 0
            else:
                stagnation += 1
            
            median_f = fitness[lam // 2]
            if abs(median_f - prev_median) < 1e-14 * (1 + abs(prev_median)):
                flat_count += 1
            else:
                flat_count = 0
            prev_median = median_f
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            diff = (mean - old_mean) / sigma
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (diff / np.sqrt(diag_C + 1e-30))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            if use_full:
                artmp = (arx[:mu] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (weights[:, None] * artmp).T @ artmp
                C = (C + C.T) / 2
                np.fill_diagonal(C, np.maximum(np.diag(C), 1e-20))
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                diag_C = (1 - c1 - cmu_val) * diag_C + \
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C) + \
                         cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2 * np.max(ranges))
            
            gen += 1
            
            if sigma < 1e-14 or stagnation > 30 + 10 * n or flat_count > 20:
                return
            
            if use_full:
                cond = np.max(D) / (np.min(D) + 1e-30)
                if cond > 1e14:
                    return
    
    # ---- L-SHADE ----
    def lshade_search(budget_seconds):
        nonlocal best, best_params
        
        de_start = datetime.now()
        N_init = min(max(10 * dim, 50), 300)
        N_min = max(4, dim)
        pop_size = N_init
        H = 100
        max_gen_est = max(100, 10 * dim)
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        pop = np.random.uniform(lower, upper, (pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        n_seed = min(pop_size // 2, n_init)
        for i in range(n_seed):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        
        for i in range(n_seed, pop_size):
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            fit[i] = eval_func(pop[i])
        
        archive = []
        max_archive = pop_size
        
        gen = 0
        nfe = pop_size - n_seed
        max_nfe = N_init * max_gen_est
        
        while True:
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            survivors = []
            survivor_fit = []
            
            p_min = max(2 / pop_size, 0.05)
            
            for i in range(pop_size):
                if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                    return
                
                ri = np.random.randint(H)
                
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi >= 1.0:
                        Fi = 1.0
                        break
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                pi = np.random.uniform(p_min, 0.2)
                p_count = max(2, int(pi * pop_size))
                pbest_idx = np.argsort(fit)[:p_count]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(idxs)
                
                all_candidates = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                r2_candidates = [j for j in all_candidates if j != i and j != r1]
                if not r2_candidates:
                    r2_candidates = [j for j in range(pop_size) if j != i]
                r2 = np.random.choice(r2_candidates)
                if r2 >= pop_size:
                    xr2 = archive[r2 - pop_size]
                else:
                    xr2 = pop[r2]
                
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
                
                if f_trial < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(abs(fit[i] - f_trial))
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    survivors.append(trial)
                    survivor_fit.append(f_trial)
                elif f_trial == fit[i]:
                    survivors.append(trial)
                    survivor_fit.append(f_trial)
                else:
                    survivors.append(pop[i].copy())
                    survivor_fit.append(fit[i])
            
            if len(S_F) > 0:
                S_delta = np.array(S_delta)
                w = S_delta / (np.sum(S_delta) + 1e-30)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
            pop = np.array(survivors)
            fit = np.array(survivor_fit)
            
            # Linear population size reduction
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * nfe / max_nfe)))
            if new_pop_size < pop_size:
                best_indices = np.argsort(fit)[:new_pop_size]
                pop = pop[best_indices]
                fit = fit[best_indices]
                pop_size = new_pop_size
                max_archive = pop_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
            
            gen += 1
    
    # ---- Nelder-Mead with adaptive restart ----
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
            delta = scale[i] if scale[i] > 1e-15 else 0.1
            simplex[i + 1][i] += delta
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            f_values[i + 1] = eval_func(simplex[i + 1])
        
        for iteration in range(200000):
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Check convergence
            spread = np.max(np.abs(simplex[-1] - simplex[0]))
            f_spread = abs(f_values[-1] - f_values[0])
            if spread < 1e-15 and f_spread < 1e-15:
                return
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if f_values[0] <= fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                if fr < f_values[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = eval_func(simplex[i])
                else:
                    xcc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fcc = eval_func(xcc)
                    if fcc < f_values[-1]:
                        simplex[-1] = xcc
                        f_values[-1] = fcc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = eval_func(simplex[i])
    
    # ---- Golden section line search along random directions ----
    def directional_search(start_point, budget_seconds):
        nonlocal best, best_params
        ds_start = datetime.now()
        
        x = start_point.copy()
        fx = eval_func(x)
        
        while True:
            if (datetime.now() - ds_start).total_seconds() >= budget_seconds:
                return
            
            # Try each coordinate direction
            improved = False
            for d in range(dim):
                if (datetime.now() - ds_start).total_seconds() >= budget_seconds:
                    return
                
                # Golden section search along dimension d
                lo = lower[d]
                hi = upper[d]
                gr = (np.sqrt(5) + 1) / 2
                
                a, b = lo, hi
                c = b - (b - a) / gr
                d_val = a + (b - a) / gr
                
                xc = x.copy(); xc[d] = c
                xd = x.copy(); xd[d] = d_val
                fc = eval_func(xc)
                fd = eval_func(xd)
                
                for _ in range(20):
                    if (datetime.now() - ds_start).total_seconds() >= budget_seconds:
                        return
                    if abs(b - a) < 1e-12 * ranges[d]:
                        break
                    if fc < fd:
                        b = d_val
                        d_val = c
                        fd = fc
                        c = b - (b - a) / gr
                        xc = x.copy(); xc[d] = c
                        fc = eval_func(xc)
                    else:
                        a = c
                        c = d_val
                        fc = fd
                        d_val = a + (b - a) / gr
                        xd = x.copy(); xd[d] = d_val
                        fd = eval_func(xd)
                
                best_d = (a + b) / 2
                x_new = x.copy()
                x_new[d] = best_d
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    improved = True
            
            if not improved:
                return
    
    # ---- Pattern search with adaptive step ----
    def pattern_search(start_point, budget_seconds):
        nonlocal best, best_params
        ps_start = datetime.now()
        
        x = start_point.copy()
        fx = eval_func(x)
        step = ranges * 0.1
        
        while True:
            if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                return
            
            improved = False
            for d in range(dim):
                if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                    return
                
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lower[d], upper[d])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new; fx = f_new; improved = True
                    continue
                
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lower[d], upper[d])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new; fx = f_new; improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-14:
                    return
    
    # ==== MAIN ORCHESTRATION ====
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Phase 2: L-SHADE (30% of remaining)
    remaining = time_left()
    if remaining > 0.5:
        lshade_search(remaining * 0.30)
    
    # Phase 3: CMA-ES with IPOP restarts (40% of remaining)
    remaining = time_left()
    if remaining > 0.5:
        cma_budget = remaining * 0.50
        cma_start_time = elapsed()
        restart_count = 0
        pop_mult = 1.0
        
        while elapsed() - cma_start_time < cma_budget:
            budget_this = cma_budget - (elapsed() - cma_start_time)
            if budget_this < 0.3:
                break
            budget_this = min(budget_this, max(budget_this / 3, 2.0))
            
            if restart_count == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.2
            elif best_params is not None and np.random.random() < 0.4:
                perturbation = np.random.randn(dim) * ranges * 0.3
                sp = np.clip(best_params + perturbation, lower, upper)
                sig = np.mean(ranges) * 0.3
            else:
                sp = np.random.uniform(lower, upper)
                sig = np.mean(ranges) * 0.4
            
            cmaes_search(sp, sig, budget_this, pop_mult)
            restart_count += 1
            pop_mult = min(pop_mult * 1.5, 8)  # IPOP
    
    # Phase 4: Directional search
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        directional_search(best_params.copy(), remaining * 0.3)
    
    # Phase 5: Nelder-Mead refinement
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.4, scale_factor=0.02)
    
    # Phase 6: Pattern search
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        pattern_search(best_params.copy(), remaining * 0.5)
    
    # Phase 7: Final ultra-fine Nelder-Mead
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.9, scale_factor=0.005)
    
    return best
#
#Key improvements:
#1. **L-SHADE** with linear population size reduction for better convergence
#2. **IPOP-CMA-ES** with proper increasing population multiplier on restarts
#3. **Golden section line search** along coordinate directions for precise 1D optimization
#4. **Better eigendecomposition scheduling** in CMA-ES with countdown timer
#5. **Condition number monitoring** to detect CMA-ES degeneration early
#6. **Multiple refinement phases** with decreasing simplex scales
#7. **Adaptive p-best in L-SHADE** with per-individual random p values
#8. **Better time budget allocation** with more time for CMA-ES restarts
