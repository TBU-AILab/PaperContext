

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.95 - elapsed()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    best_pos = None
    
    # --- Phase 1: Multi-restart DE with adaptive parameters ---
    pop_size = min(max(30, 10 * dim), 250)
    
    # Initialize population using Latin Hypercube Sampling
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    population = lhs_sample(pop_size)
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best = fitness[best_idx]
    best_pos = population[best_idx].copy()
    
    # Success history for SHADE-like adaptation
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation_count = 0
    last_best = best
    
    # Track for population size reduction (L-SHADE style)
    n_init = pop_size
    n_min = max(4, dim)
    total_evals_estimate = max(1000, int(max_time * 500))  # rough estimate
    evals_used = pop_size
    
    while time_left() > 0.5:
        generation += 1
        
        S_F = []
        S_CR = []
        delta_f = []
        
        indices = list(range(len(population)))
        np.random.shuffle(indices)
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        # p-best for current-to-pbest/1
        p_min = 2.0 / len(population)
        p_max = 0.2
        
        for i in indices:
            if time_left() <= 0.3:
                return best
            
            # Generate F and CR from history
            ri = np.random.randint(H)
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best selection
            p = np.random.uniform(p_min, p_max)
            p_num = max(1, int(round(p * len(population))))
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[np.random.randint(p_num)]
            
            # Select r1 != i
            candidates = [c for c in range(len(population)) if c != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            # Select r2 from population + archive, != i, != r1
            combined = list(range(len(population))) + list(range(len(population), len(population) + len(archive)))
            combined = [c for c in combined if c != i and c != r1]
            if len(combined) == 0:
                combined = [c for c in range(len(population)) if c != i]
            r2 = combined[np.random.randint(len(combined))]
            
            if r2 < len(population):
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - len(population)]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            cross = np.random.rand(dim) < CRi
            if not np.any(cross):
                cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            # Bounce-back boundary handling
            mask_lo = trial < lower
            mask_hi = trial > upper
            trial[mask_lo] = (lower[mask_lo] + population[i][mask_lo]) / 2
            trial[mask_hi] = (upper[mask_hi] + population[i][mask_hi]) / 2
            trial = clip_to_bounds(trial)
            
            trial_fit = func(trial)
            evals_used += 1
            
            if trial_fit <= fitness[i]:
                if trial_fit < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_fit))
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive_max > 0:
                        archive[np.random.randint(archive_max)] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = trial_fit
                if trial_fit < best:
                    best = trial_fit
                    best_pos = trial.copy()
        
        population = new_pop
        fitness = new_fit
        
        # Update history
        if len(S_F) > 0:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % H
        
        # Stagnation check
        if generation % 15 == 0:
            if abs(best - last_best) < 1e-14:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best = best
            
            if stagnation_count >= 4:
                n_restart = len(population) // 2
                sorted_idx = np.argsort(fitness)
                for j in range(n_restart):
                    idx = sorted_idx[-(j+1)]
                    population[idx] = lower + np.random.rand(dim) * ranges
                    fitness[idx] = func(population[idx])
                    if fitness[idx] < best:
                        best = fitness[idx]
                        best_pos = population[idx].copy()
                stagnation_count = 0
                last_best = best
        
        # Local search around best periodically
        if generation % 8 == 0 and best_pos is not None and time_left() > 1.0:
            local_best = best_pos.copy()
            local_val = best
            scale = 0.01
            for _ in range(3 * dim):
                if time_left() <= 0.5:
                    break
                direction = np.random.randn(dim)
                trial = clip_to_bounds(local_best + scale * ranges * direction)
                tv = func(trial)
                if tv < local_val:
                    local_val = tv
                    local_best = trial.copy()
                    scale *= 1.1
                else:
                    scale *= 0.9
            if local_val < best:
                best = local_val
                best_pos = local_best.copy()
                worst_idx = np.argmax(fitness)
                population[worst_idx] = local_best
                fitness[worst_idx] = local_val
    
    return best