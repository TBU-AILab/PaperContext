#Looking at the current algorithm, I need to analyze what might be causing the score of 27.38. The algorithm has good components but may have issues with:
#
#1. **Time allocation** - too many phases with fixed ratios
#2. **CMA-ES restarts** - budget splitting per restart too aggressive
#3. **SHADE** - could benefit from linear population size reduction (L-SHADE)
#4. **Missing exploitation** - need better local search near the end
#
#Let me create an improved version:
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
            
            order = np.argsort(fitness)
            arx = arx[order]; arz = arz[order]; fitness = fitness[order]
            
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
            
            if sigma < 1e-15 or (gen - best_local_gen) > 20 + 10*n or cond > 1e14:
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
        
        total_evals_estimate = pop_size * int(budget_seconds * 500)  # rough
        evals_used = pop_size
        max_evals = max(total_evals_estimate, pop_size * 50)
        
        gen = 0
        while True:
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            sort_idx = np.argsort(fit)
            
            trial_pop = np.empty_like(pop)
            trial_fit = np.full(pop_size, float('inf'))
            Fs = np.empty(pop_size)
            CRs = np.empty(pop_size)
            
            for i in range(pop_size):
                ri = np.random.randint(H)
                
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi >= 1.0:
                        Fi = 1.0
                        break
                Fs[i] = Fi
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                CRs[i] = CRi
                
                p = max(2, int(np.random.uniform(0.05, 0.2) * pop_size))
                pbest_idx = sort_idx[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                combined = [c for c in combined if c != i and c != r1]
                if not combined:
                    combined = [j for j in range(pop_size) if j != i]
                r2 = np.random.choice(combined)
                if r2 < pop_size:
                    xr2 = pop[r2]
                else:
                    xr2 = archive[r2 - pop_size]
                
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
                S_delta = np.array(S_delta)
                w = S_delta / (np.sum(S_delta) + 1e-30)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
            # Linear population size reduction
            new_pop_size = max(pop_min, int(round(pop_init - (pop_init - pop_min) * evals_used / max_evals)))
            if new_pop_size < pop_size:
                worst = np.argsort(fit)
                keep = worst[:new_pop_size]
                pop = pop[keep]
                fit = fit[keep]
                pop_size = new_pop_size
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
            delta = scale[i] if scale[i] > 1e-15 else 0.01
            simplex[i + 1][i] += delta
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
            
            if no_improve > 30 * n:
                return
            
            diam = np.max(np.abs(simplex[-1] - simplex[0]))
            if diam < 1e-16:
                return

    # --- Powell-like coordinate descent with quadratic fitting ---
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
            max_delta = 0
            max_delta_idx = 0
            
            for i in range(dim):
                if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                    return
                
                d = directions[i]
                f_before = fx
                
                # Golden section line search
                a_lo, a_hi = -np.max(ranges), np.max(ranges)
                # Find bracket first
                step = np.mean(ranges) * 0.01
                best_alpha = 0
                best_f = fx
                
                for alpha in [step, -step, step*5, -step*5, step*20, -step*20]:
                    x_try = np.clip(x + alpha * d, lower, upper)
                    f_try = eval_func(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_alpha = alpha
                
                if best_alpha != 0:
                    # Refine with golden section
                    gr = (np.sqrt(5) + 1) / 2
                    a = best_alpha - abs(best_alpha) * 0.5
                    b = best_alpha + abs(best_alpha) * 0.5
                    
                    for _ in range(15):
                        if (datetime.now() - ps_start).total_seconds() >= budget_seconds:
                            return
                        c = b - (b - a) / gr
                        dd = a + (b - a) / gr
                        fc = eval_func(np.clip(x + c * d, lower, upper))
                        fd = eval_func(np.clip(x + dd * d, lower, upper))
                        if fc < fd:
                            b = dd
                        else:
                            a = c
                    
                    mid = (a + b) / 2
                    x_new = np.clip(x + mid * d, lower, upper)
                    f_new = eval_func(x_new)
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                    elif best_f < fx:
                        x = np.clip(x + best_alpha * d, lower, upper) 
                        fx = best_f
                
                delta = f_before - fx
                if delta > max_delta:
                    max_delta = delta
                    max_delta_idx = i
            
            improvement = fx_start - fx
            if improvement < 1e-14:
                return
            
            # Update directions (Powell's method)
            new_dir = x - x_start
            norm = np.linalg.norm(new_dir)
            if norm > 1e-16:
                new_dir /= norm
                directions = np.delete(directions, max_delta_idx, axis=0)
                directions = np.vstack([directions, new_dir])

    # --- Main orchestration ---
    remaining = time_left()
    if remaining <= 0:
        return best

    # Phase 1: L-SHADE (40% of time)
    remaining = time_left()
    if remaining > 0.5:
        lshade_search(remaining * 0.40)
    
    # Phase 2: CMA-ES with IPOP restarts (30% of time)
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
                    sp = best_params + np.random.randn(dim) * ranges * np.random.uniform(0.1, 0.5)
                    sp = np.clip(sp, lower, upper)
                    sig = np.mean(ranges) * 0.3
                else:
                    sp = np.random.uniform(lower, upper)
                    sig = np.mean(ranges) * 0.4
            
            cmaes_search(sp, sig, budget_this, pop_mult)
            restart_count += 1
            pop_mult = min(pop_mult * 1.5, 5)
    
    # Phase 3: Powell search from best
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.4)
    
    # Phase 4: Nelder-Mead refinement
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.5, scale_factor=0.02)
    
    # Phase 5: Fine NM
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        nelder_mead(best_params.copy(), remaining * 0.5, scale_factor=0.001)
    
    # Phase 6: Final Powell refinement
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        powell_search(best_params.copy(), remaining * 0.9)
    
    return best
#
#Key improvements:
#1. **L-SHADE** with linear population size reduction for better convergence
#2. **Powell's conjugate direction method** replacing the simple directional search - builds up better search directions over time
#3. **Better time allocation** - more time to global search (L-SHADE + CMA-ES = 80%), then systematic local refinement
#4. **Improved CMA-ES restart strategy** with more gradual perturbation scaling
#5. **Golden section line search** within Powell for more precise 1D optimization
#6. **Better stagnation detection** in CMA-ES using generation tracking instead of median comparison
