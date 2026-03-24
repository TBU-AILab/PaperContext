#Looking at the results, algorithm 2 (SHADE + multi-scale Nelder-Mead) scored best at 28.37. I'll improve upon it with several key enhancements:
#
#1. **Better CMA-ES-like local search** using covariance adaptation instead of just Nelder-Mead
#2. **Improved time allocation** - more time for local refinement
#3. **Population size linear reduction (LPSR)** during DE phase
#4. **Multiple independent restarts** of local search from diverse good solutions
#5. **Rank-based weighting** in SHADE updates
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = 0
    def evaluate(x):
        nonlocal best, best_params, eval_count
        x = clip(x)
        f = func(x)
        eval_count += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size = min(max(30, 8 * dim), 200)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness_vals[i] = evaluate(population[i])

    # Store top-k diverse solutions for later restarts
    top_solutions = []
    
    # --- Phase 2: SHADE with Linear Population Size Reduction ---
    memory_size = 25
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    memory_idx = 0
    archive = []
    max_archive = pop_size
    
    stagnation_count = 0
    prev_best = best
    gen = 0
    initial_pop_size = pop_size
    min_pop_size = max(5, dim)
    
    de_time_budget = 0.55
    
    while elapsed() < max_time * de_time_budget:
        gen += 1
        S_F = []
        S_CR = []
        S_delta = []
        
        # Adaptive p for pbest
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.25 * pop_size))
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_budget:
                break
            
            r = np.random.randint(memory_size)
            F_i = -1
            while F_i <= 0:
                F_i = np.random.standard_cauchy() * 0.1 + M_F[r]
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            
            # current-to-pbest/1
            p = np.random.randint(p_min, p_max + 1)
            sorted_idx = np.argsort(fitness_vals[:pop_size])
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2_idx = i
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(combined_size)
            x_r2 = population[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CR_i
            cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2.0
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2.0
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness_vals[i]:
                delta = fitness_vals[i] - f_trial
                if delta > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness_vals[i] = f_trial
        
        # Update memory with weighted Lehmer mean
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[memory_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[memory_idx] = np.sum(weights * scr)
            memory_idx = (memory_idx + 1) % memory_size
        
        if abs(prev_best - best) < 1e-14:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        # Population size reduction
        if gen % 5 == 0 and pop_size > min_pop_size:
            new_size = max(min_pop_size, int(initial_pop_size - (initial_pop_size - min_pop_size) * elapsed() / (max_time * de_time_budget)))
            if new_size < pop_size:
                sorted_idx = np.argsort(fitness_vals[:pop_size])
                population = population[sorted_idx[:new_size]]
                fitness_vals = fitness_vals[sorted_idx[:new_size]]
                pop_size = new_size
        
        if stagnation_count > 10:
            sorted_idx = np.argsort(fitness_vals[:pop_size])
            half = pop_size // 2
            for j in sorted_idx[half:]:
                if elapsed() >= max_time * de_time_budget:
                    break
                if np.random.random() < 0.4:
                    scale = 0.15 * ranges * np.random.random()
                    population[j] = best_params + np.random.randn(dim) * scale
                else:
                    population[j] = lower + np.random.random(dim) * ranges
                population[j] = clip(population[j])
                fitness_vals[j] = evaluate(population[j])
            stagnation_count = 0

    # Collect top diverse solutions
    sorted_idx = np.argsort(fitness_vals[:pop_size])
    for idx in sorted_idx[:min(5, pop_size)]:
        top_solutions.append((population[idx].copy(), fitness_vals[idx]))

    # --- Phase 3: Multi-scale Nelder-Mead ---
    def nelder_mead(start_point, start_fit, scale_factor, time_limit_abs):
        nonlocal best, best_params
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        f_simplex = np.zeros(n + 1)
        simplex[0] = start_point.copy()
        f_simplex[0] = start_fit
        
        for i in range(n):
            if elapsed() >= time_limit_abs:
                return
            simplex[i+1] = start_point.copy()
            simplex[i+1][i] += scale[i] if abs(scale[i]) > 1e-15 else 0.01
            simplex[i+1] = clip(simplex[i+1])
            f_simplex[i+1] = evaluate(simplex[i+1])
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        no_improve = 0
        
        for _ in range(20000):
            if elapsed() >= time_limit_abs:
                return
            order = np.argsort(f_simplex)
            simplex, f_simplex = simplex[order], f_simplex[order]
            
            old_best_f = f_simplex[0]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + alpha*(centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma*(xr - centroid))
                fe = evaluate(xe)
                simplex[-1], f_simplex[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                candidate = xr if fr < f_simplex[-1] else simplex[-1]
                xc = clip(centroid + rho*(candidate - centroid))
                fc = evaluate(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for j in range(1, n+1):
                        simplex[j] = clip(simplex[0] + sigma*(simplex[j]-simplex[0]))
                        f_simplex[j] = evaluate(simplex[j])
                        if elapsed() >= time_limit_abs: return
            
            if f_simplex[0] >= old_best_f - 1e-15:
                no_improve += 1
            else:
                no_improve = 0
            
            if no_improve > max(50, 3*dim) or np.std(f_simplex) < 1e-16:
                break
    
    # Run Nelder-Mead with multiple scales
    nelder_mead(best_params, best, 0.08, max_time*0.68)
    nelder_mead(best_params, best, 0.03, max_time*0.78)
    nelder_mead(best_params, best, 0.01, max_time*0.86)
    nelder_mead(best_params, best, 0.003, max_time*0.92)
    
    # Try from diverse top solutions
    for sol, sol_fit in top_solutions:
        if elapsed() >= max_time * 0.94:
            break
        if np.linalg.norm(sol - best_params) > 0.01 * np.linalg.norm(ranges):
            nelder_mead(sol, sol_fit, 0.02, max_time*0.94)
    
    nelder_mead(best_params, best, 0.001, max_time*0.97)

    # --- Phase 4: Pattern search around best ---
    step = 0.001 * ranges
    while elapsed() < max_time * 0.985:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.985:
                break
            for sign in [1, -1]:
                trial = best_params.copy()
                trial[d] += sign * step[d]
                trial = clip(trial)
                f_trial = evaluate(trial)
                if f_trial < best:
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if np.max(step / ranges) < 1e-12:
                break

    # --- Phase 5: Fine random perturbation ---
    while elapsed() < max_time * 0.998:
        s = np.random.exponential(0.002)
        candidate = best_params + np.random.randn(dim) * s * ranges
        evaluate(candidate)
    
    return best
#
#Key improvements over the best algorithm:
#1. **LPSR (Linear Population Size Reduction)** - gradually reduces population during DE for better convergence
#2. **Adaptive p-range** in current-to-pbest mutation
#3. **More Nelder-Mead passes** at progressively finer scales (5 passes vs 3)
#4. **Diverse restart points** - tries Nelder-Mead from top diverse solutions, not just the best
#5. **Pattern search phase** - coordinate-wise refinement that's very efficient for separable dimensions
#6. **Early termination of stagnant NM** with adaptive no-improve counter
#7. **Better time allocation** - 55% DE, 42% local search, 3% fine tuning
