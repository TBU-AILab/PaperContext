#Looking at the current algorithm, I can see it's a SHADE-like DE with local search. Let me improve it with several enhancements:
#
#1. **Better SHADE implementation** with multiple memory cells
#2. **Linear population size reduction** (L-SHADE)
#3. **CMA-ES-inspired local search** for the refinement phase
#4. **Multiple restarts** of local search on top candidates
#5. **Vectorized operations** where possible for speed
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip_to_bounds(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: LHS initialization ---
    init_pop_size = min(max(30, 12 * dim), 300)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = eval_func(population[i])
    
    # --- Phase 2: L-SHADE ---
    H = 6  # memory size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0  # memory index
    archive = []
    max_archive = pop_size
    
    max_nfe = None  # we use time-based
    nfe = pop_size
    
    generation = 0
    no_improve_count = 0
    prev_best = best
    
    de_time_budget = max_time * 0.85
    
    while elapsed() < de_time_budget:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.2 * pop_size))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            # Pick random memory index
            ri = np.random.randint(0, H)
            
            # Generate F
            while True:
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F > 0:
                    break
            F = min(F, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CR = 0.0
            else:
                CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # pbest
            pi = max(p_min, np.random.randint(p_min, p_max + 1))
            p_best_idx = sorted_idx[np.random.randint(0, pi)]
            
            # r1
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            
            # r2 from pop + archive
            combined_size = pop_size + len(archive)
            r2 = i
            attempts = 0
            while (r2 == i or r2 == r1) and attempts < 25:
                r2 = np.random.randint(0, combined_size)
                attempts += 1
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # Mutation
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - x_r2)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back bounds
            out_low = trial < lower
            out_high = trial > upper
            if np.any(out_low):
                trial[out_low] = lower[out_low] + np.random.random(np.sum(out_low)) * (population[i][out_low] - lower[out_low])
            if np.any(out_high):
                trial[out_high] = upper[out_high] - np.random.random(np.sum(out_high)) * (upper[out_high] - population[i][out_high])
            trial = clip_to_bounds(trial)
            
            trial_f = eval_func(trial)
            nfe += 1
            
            if trial_f <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_f
                if trial_f < fitness[i]:
                    delta = fitness[i] - trial_f
                    S_F.append(F)
                    S_CR.append(CR)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / np.sum(weights)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            
            # Lehmer mean for F
            M_F[k] = np.sum(weights * sf ** 2) / (np.sum(weights * sf) + 1e-30)
            # Weighted mean for CR
            M_CR[k] = np.sum(weights * scr)
            k = (k + 1) % H
        
        # Check improvement
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # L-SHADE population reduction
        fraction_elapsed = elapsed() / de_time_budget
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * fraction_elapsed)))
        
        if new_pop_size < pop_size:
            sidx = np.argsort(fitness)
            keep = sidx[:new_pop_size]
            population = population[keep]
            fitness = fitness[keep]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation restart
        if no_improve_count > 30 and pop_size > min_pop_size + 4:
            no_improve_count = 0
            sidx = np.argsort(fitness)
            n_restart = pop_size // 3
            for kk in range(n_restart):
                idx = sidx[pop_size - 1 - kk]
                # Restart near best with some exploration
                if np.random.random() < 0.5:
                    population[idx] = best_x + np.random.randn(dim) * ranges * 0.1
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = eval_func(population[idx])
    
    # --- Phase 3: Multi-start local search on top candidates ---
    sidx = np.argsort(fitness)
    candidates = [population[sidx[i]].copy() for i in range(min(3, pop_size))]
    if best_x is not None and not any(np.allclose(best_x, c) for c in candidates):
        candidates.insert(0, best_x.copy())
    
    for cand_x in candidates:
        if time_left() < max_time * 0.01:
            break
        
        x = cand_x.copy()
        fx = eval_func(x)
        step = 0.05 * ranges.copy()
        
        while time_left() > max_time * 0.005:
            improved = False
            for j in range(dim):
                if time_left() < max_time * 0.005:
                    return best
                
                # Try positive direction
                trial = x.copy()
                trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
                ft = eval_func(trial)
                if ft < fx:
                    fx = ft
                    x = trial
                    step[j] *= 1.2
                    improved = True
                    continue
                
                # Try negative direction
                trial = x.copy()
                trial[j] = np.clip(trial[j] - step[j], lower[j], upper[j])
                ft = eval_func(trial)
                if ft < fx:
                    fx = ft
                    x = trial
                    step[j] *= 1.2
                    improved = True
                    continue
                
                step[j] *= 0.5
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-14:
                    break
        
        # Also try a Nelder-Mead simplex from this point
        if time_left() > max_time * 0.02:
            n = dim
            alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
            simplex = np.empty((n + 1, n))
            simplex[0] = x.copy()
            simplex_f = np.empty(n + 1)
            simplex_f[0] = fx
            for j in range(n):
                p = x.copy()
                delta = max(0.005 * ranges[j], 1e-8)
                p[j] += delta
                p = clip_to_bounds(p)
                simplex[j + 1] = p
                simplex_f[j + 1] = eval_func(p)
            
            while time_left() > max_time * 0.005:
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                if np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30)) < 1e-14:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                xr = clip_to_bounds(centroid + alpha_nm * (centroid - simplex[-1]))
                fr = eval_func(xr)
                
                if fr < simplex_f[0]:
                    # Expand
                    xe = clip_to_bounds(centroid + gamma_nm * (xr - centroid))
                    fe = eval_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = clip_to_bounds(centroid + rho_nm * (xr - centroid))
                        fc = eval_func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, n + 1):
                                simplex[j] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[j] - simplex[0]))
                                simplex_f[j] = eval_func(simplex[j])
                    else:
                        # Inside contraction
                        xc = clip_to_bounds(centroid - rho_nm * (centroid - simplex[-1]))
                        fc = eval_func(xc)
                        if fc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, n + 1):
                                simplex[j] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[j] - simplex[0]))
                                simplex_f[j] = eval_func(simplex[j])
    
    return best
#
#Key improvements:
#- **L-SHADE with weighted Lehmer mean** for F adaptation and proper delta-weighted memory updates
#- **Linear population size reduction** to focus search over time
#- **Multi-start local search** on top 3 candidates plus the global best
#- **Adaptive step sizes** in coordinate descent (increase on success)
#- **Nelder-Mead simplex** as a secondary local optimizer
#- **Better time budget allocation** across phases
#- **Improved stagnation handling** with mixed restarts (near-best + random)
