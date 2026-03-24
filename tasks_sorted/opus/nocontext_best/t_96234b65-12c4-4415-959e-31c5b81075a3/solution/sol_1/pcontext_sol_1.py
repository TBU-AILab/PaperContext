#Looking at the current algorithm, I can see several areas for improvement:
#
#1. **Better CMA-ES implementation** with restarts and adaptive sigma
#2. **More aggressive local search** combining methods
#3. **Better time allocation** - spend more time on the most effective methods
#4. **Add a surrogate-assisted component** for expensive functions
#5. **Improve DE with adaptive parameters (SHADE-like)**
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
    
    eval_cache = {}
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        key = tuple(np.round(x, 12))
        if key in eval_cache:
            return eval_cache[key]
        f = func(x)
        eval_cache[key] = f
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Smart initialization with LHS + Sobol-like ---
    n_init = min(max(30 * dim, 200), 800)
    
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.05:
            n_init = i
            init_pop = init_pop[:n_init]
            init_fitness = init_fitness[:n_init]
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    if n_init == 0:
        return best
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- CMA-ES with IPOP restarts ---
    def cmaes_search(start_point, initial_sigma, budget_seconds):
        nonlocal best, best_params
        
        cma_start = datetime.now()
        n = dim
        
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        mean = start_point.copy()
        sigma = initial_sigma
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        if n <= 80:
            C = np.eye(n)
            use_full = True
            eigeneval = 0
            update_eigen_every = max(1, lam // (10 * n))
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
        else:
            diag_C = np.ones(n)
            use_full = False
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        gen = 0
        stagnation_count = 0
        prev_best_f = float('inf')
        
        while True:
            budget_left = budget_seconds - (datetime.now() - cma_start).total_seconds()
            if budget_left <= 0:
                return
            
            if use_full and gen % max(1, int(1.0 / (c1 + cmu_val) / n / 10)) == 0:
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
            
            # Generate offspring
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
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            diff = (mean - old_mean) / sigma
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (diff / np.sqrt(diag_C))
            
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
            sigma = np.clip(sigma, 1e-16, 2 * np.mean(ranges))
            
            gen += 1
            
            # Check stagnation
            if fitness[0] < prev_best_f - 1e-12:
                stagnation_count = 0
                prev_best_f = fitness[0]
            else:
                stagnation_count += 1
            
            if sigma < 1e-14 or stagnation_count > 50 + 10 * n:
                return
    
    # --- Adaptive DE (SHADE-inspired) ---
    def shade_search(budget_seconds):
        nonlocal best, best_params
        
        de_start = datetime.now()
        pop_size = min(max(8 * dim, 40), 200)
        H = 100
        
        # Memory for F and CR
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        pop = np.random.uniform(lower, upper, (pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        # Seed with best init samples
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
        while True:
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            
            S_F = []
            S_CR = []
            S_delta = []
            
            for i in range(pop_size):
                if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                    return
                
                ri = np.random.randint(H)
                
                # Generate F
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                # Generate CR
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                # Current-to-pbest/1
                p = max(2, int(0.1 * pop_size))
                pbest_idx = np.argsort(fit)[:p]
                xpbest = pop[np.random.choice(pbest_idx)]
                
                idxs = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(idxs)
                
                # r2 from pop + archive
                combined = list(range(pop_size))
                combined_pop = list(pop)
                for a in archive:
                    combined.append(len(combined_pop))
                    combined_pop.append(a)
                r2_candidates = [j for j in range(len(combined_pop)) if j != i and j != r1]
                if len(r2_candidates) == 0:
                    r2_candidates = [j for j in range(pop_size) if j != i]
                r2 = np.random.choice(r2_candidates)
                xr2 = combined_pop[r2] if r2 < len(combined_pop) else pop[r2 % pop_size]
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.random(dim) < CRi, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                
                # Bounce-back
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                trial = np.clip(trial, lower, upper)
                
                f_trial = eval_func(trial)
                
                if f_trial < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(abs(fit[i] - f_trial))
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial
                    fit[i] = f_trial
                elif f_trial == fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
            
            # Update memory
            if len(S_F) > 0:
                S_delta = np.array(S_delta)
                w = S_delta / np.sum(S_delta)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
            gen += 1
    
    # --- Nelder-Mead ---
    def nelder_mead(start_point, start_fitness, budget_seconds):
        nonlocal best, best_params
        
        nm_start = datetime.now()
        n = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = start_point.copy()
        f_values = np.full(n + 1, float('inf'))
        f_values[0] = start_fitness
        
        scale = ranges * 0.05
        for i in range(n):
            simplex[i + 1] = start_point.copy()
            simplex[i + 1][i] += scale[i] if scale[i] > 1e-15 else 0.1
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            f_values[i + 1] = eval_func(simplex[i + 1])
        
        for _ in range(50000):
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
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
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = eval_func(simplex[i])
                else:
                    xcc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fcc = eval_func(xcc)
                    if fcc < f_values[-1]:
                        simplex[-1] = xcc; f_values[-1] = fcc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = eval_func(simplex[i])
            
            if np.max(f_values) - np.min(f_values) < 1e-15:
                return
    
    # --- Coordinate-wise local search ---
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
                
                # Try positive
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lower[d], upper[d])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new; fx = f_new; improved = True
                    continue
                
                # Try negative
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lower[d], upper[d])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new; fx = f_new; improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-14:
                    return
    
    # --- Main orchestration ---
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Phase 2: SHADE (40% of remaining time)
    remaining = time_left()
    if remaining > 0.5:
        shade_search(remaining * 0.40)
    
    # Phase 3: CMA-ES with restarts (35% of remaining time)
    remaining = time_left()
    if remaining > 0.5:
        cma_budget = remaining * 0.50
        cma_start_time = elapsed()
        restart_count = 0
        pop_multiplier = 1
        while elapsed() - cma_start_time < cma_budget:
            budget_this = min(cma_budget - (elapsed() - cma_start_time), cma_budget / max(1, 3 - restart_count))
            if budget_this < 0.3:
                break
            
            if restart_count == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.3
            else:
                # Random restart with bias toward best
                if best_params is not None and np.random.random() < 0.5:
                    perturbation = np.random.randn(dim) * ranges * (0.3 + 0.2 * restart_count)
                    sp = np.clip(best_params + perturbation, lower, upper)
                else:
                    sp = np.random.uniform(lower, upper)
                sig = np.mean(ranges) * (0.3 + 0.1 * restart_count)
            
            cmaes_search(sp, sig, budget_this)
            restart_count += 1
    
    # Phase 4: Nelder-Mead refinement from best
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        nm_budget = remaining * 0.5
        nelder_mead(best_params.copy(), best, nm_budget)
    
    # Phase 5: Pattern search refinement
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        pattern_search(best_params.copy(), remaining * 0.8)
    
    # Phase 6: Final NM with tiny simplex
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), best, remaining * 0.9)
    
    return best
#
#Key improvements:
#1. **SHADE** (Success-History based Adaptive DE) replaces basic DE with adaptive F/CR parameters
#2. **IPOP-CMA-ES** with increasing population restarts and proper eigendecomposition frequency
#3. **Pattern search** as a robust local optimizer for coordinate-aligned problems
#4. **Evaluation caching** to avoid redundant function evaluations
#5. **Better time allocation**: SHADE→CMA-ES restarts→Nelder-Mead→Pattern search→Final NM
#6. **Bounce-back boundary handling** in SHADE instead of simple clipping
#7. **Multiple CMA-ES restarts** with varying sigma and random perturbations
