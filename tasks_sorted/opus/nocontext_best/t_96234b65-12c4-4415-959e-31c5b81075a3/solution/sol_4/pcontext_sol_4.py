#Looking at the current algorithm, I can identify several issues and potential improvements:
#
#1. **Time allocation is suboptimal** - too much splitting between methods
#2. **CMA-ES eigendecomposition frequency** could be better tuned
#3. **SHADE population management** could be improved
#4. **Missing multi-start strategies** for escaping local optima
#5. **No adaptive switching** between methods based on progress
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
    
    n_evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params, n_evals
        x = np.clip(x, lower, upper)
        n_evals += 1
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def mirror_bound(x, lb, ub):
        """Mirror boundary handling"""
        for i in range(len(x)):
            while x[i] < lb[i] or x[i] > ub[i]:
                if x[i] < lb[i]:
                    x[i] = 2*lb[i] - x[i]
                if x[i] > ub[i]:
                    x[i] = 2*ub[i] - x[i]
        return x

    # --- LHS initialization ---
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

    # --- CMA-ES ---
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
        stag = 0
        prev_median = float('inf')
        flat_count = 0
        best_local = float('inf')
        
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
            sigma = np.clip(sigma, 1e-16, np.max(ranges))
            
            gen += 1
            
            # Stagnation detection
            med = np.median(fitness)
            if abs(med - prev_median) < 1e-12 * (abs(prev_median) + 1e-30):
                flat_count += 1
            else:
                flat_count = 0
            prev_median = med
            
            if fitness[0] >= best_local - 1e-13:
                stag += 1
            else:
                stag = 0
            
            cond = np.max(D) / (np.min(D) + 1e-30) if use_full else np.sqrt(np.max(diag_C) / (np.min(diag_C) + 1e-30))
            
            if sigma < 1e-15 or stag > 30 + 15*n or flat_count > 20 or cond > 1e14:
                return best_local
        
        return best_local
    
    # --- SHADE ---
    def shade_search(budget_seconds, pop_size=None):
        nonlocal best, best_params
        
        de_start = datetime.now()
        if pop_size is None:
            pop_size = min(max(6 * dim, 30), 150)
        H = 50
        
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
        
        gen = 0
        while True:
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            # Adaptive p
            p_min = max(2, int(0.05 * pop_size))
            p_max = max(2, int(0.25 * pop_size))
            
            sort_idx = np.argsort(fit)
            
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
                
                p = np.random.randint(p_min, p_max + 1)
                pbest_idx = sort_idx[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined_pop = list(pop) + archive
                r2_candidates = list(range(len(combined_pop)))
                if i in r2_candidates:
                    r2_candidates.remove(i)
                if r1 in r2_candidates:
                    r2_candidates.remove(r1)
                if not r2_candidates:
                    r2_candidates = [j for j in range(pop_size) if j != i]
                r2 = np.random.choice(r2_candidates)
                xr2 = combined_pop[r2]
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - xr2)
                
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Bounce-back
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                trial = np.clip(trial, lower, upper)
                
                f_trial = eval_func(trial)
                
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
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
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
        for _ in range(200000):
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
            
            if no_improve > 50 * n:
                return
            
            diam = np.max(np.abs(simplex[-1] - simplex[0]))
            if diam < 1e-15:
                return

    # --- Golden section line search along random directions ---
    def directional_search(start_point, budget_seconds):
        nonlocal best, best_params
        ds_start = datetime.now()
        
        x = start_point.copy()
        fx = eval_func(x)
        
        while True:
            if (datetime.now() - ds_start).total_seconds() >= budget_seconds:
                return
            
            improved = False
            # Try coordinate directions and random directions
            directions = []
            for d in range(dim):
                e = np.zeros(dim)
                e[d] = 1.0
                directions.append(e)
            for _ in range(dim):
                rd = np.random.randn(dim)
                rd /= (np.linalg.norm(rd) + 1e-30)
                directions.append(rd)
            
            for direction in directions:
                if (datetime.now() - ds_start).total_seconds() >= budget_seconds:
                    return
                
                # Bracket search
                step = np.mean(ranges) * 0.01
                for _ in range(20):
                    x_new = np.clip(x + step * direction, lower, upper)
                    f_new = eval_func(x_new)
                    if f_new < fx:
                        x = x_new; fx = f_new; improved = True
                        step *= 1.5
                    else:
                        x_new = np.clip(x - step * direction, lower, upper)
                        f_new = eval_func(x_new)
                        if f_new < fx:
                            x = x_new; fx = f_new; improved = True
                            step *= 1.5
                        else:
                            step *= 0.5
                    if step < 1e-15:
                        break
            
            if not improved:
                return

    # --- Main orchestration ---
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Phase 1: SHADE (35% of remaining time)
    remaining = time_left()
    if remaining > 0.5:
        shade_search(remaining * 0.35)
    
    # Phase 2: CMA-ES with IPOP restarts (35% of remaining time)
    remaining = time_left()
    if remaining > 0.5:
        cma_budget = remaining * 0.45
        cma_start_time = elapsed()
        restart_count = 0
        pop_mult = 1.0
        while elapsed() - cma_start_time < cma_budget:
            budget_this = cma_budget - (elapsed() - cma_start_time)
            if budget_this < 0.2:
                break
            budget_this = min(budget_this, cma_budget / 3)
            
            if restart_count == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.2
            elif restart_count == 1 and best_params is not None:
                # Perturbed best
                sp = best_params + np.random.randn(dim) * ranges * 0.3
                sp = np.clip(sp, lower, upper)
                sig = np.mean(ranges) * 0.3
            else:
                if best_params is not None and np.random.random() < 0.3:
                    sp = best_params + np.random.randn(dim) * ranges * 0.5
                    sp = np.clip(sp, lower, upper)
                else:
                    sp = np.random.uniform(lower, upper)
                sig = np.mean(ranges) * 0.4
            
            cmaes_search(sp, sig, budget_this, pop_mult)
            restart_count += 1
            pop_mult = min(pop_mult * 1.5, 4)
    
    # Phase 3: Nelder-Mead from best
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.4, scale_factor=0.05)
    
    # Phase 4: Directional refinement
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        directional_search(best_params.copy(), remaining * 0.5)
    
    # Phase 5: Final NM with tiny simplex
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.5, scale_factor=0.001)
    
    # Phase 6: Very fine directional
    remaining = time_left()
    if remaining > 0.1 and best_params is not None:
        directional_search(best_params.copy(), remaining * 0.9)
    
    return best
#
#Key improvements over the previous version:
#1. **Removed evaluation cache** - the hashing overhead often exceeds re-evaluation cost and can miss nearby points
#2. **Better IPOP-CMA-ES restarts** with increasing population multiplier
#3. **Improved stagnation detection** in CMA-ES with condition number monitoring
#4. **Directional search** combining coordinate and random directions with adaptive step sizes
#5. **Adaptive p-range in SHADE** for better exploitation/exploration balance
#6. **Multiple NM phases** with decreasing simplex scale (0.05 → 0.001)
#7. **Better time allocation** with more phases for refinement
