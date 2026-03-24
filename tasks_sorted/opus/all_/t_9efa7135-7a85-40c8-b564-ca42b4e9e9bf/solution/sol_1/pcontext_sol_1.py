

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_solution = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def evaluate(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(15 * dim, 40), 150)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])
    
    # --- Phase 2: CMA-ES inspired + DE hybrid ---
    # Use SHADE (Success-History based Adaptive DE)
    
    H = 50  # history size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    h_idx = 0
    
    archive = []
    archive_max = pop_size
    
    stagnation_counter = 0
    last_best = best
    
    generation = 0
    while elapsed() < max_time * 0.80:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        
        # Generate F and CR from history
        S_F = []
        S_CR = []
        S_w = []
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # p-best fraction
        p_min = 2.0 / pop_size
        p_max = 0.2
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            
            # Pick random history index
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            # Generate CR from Normal
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best index
            p = np.random.uniform(p_min, p_max)
            p_num = max(1, int(p * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p_num)]
            
            # Select r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            # Select r2 from pop + archive, != i, != r1
            combined = list(range(pop_size + len(archive)))
            combined = [c for c in combined if c != i and c != r1]
            r2 = np.random.choice(combined)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            # Current-to-pbest/1
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CRi
            j_rand = np.random.randint(0, dim)
            cross_points[j_rand] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Boundary handling - midpoint
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2.0
            trial[above] = (upper[above] + population[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    delta = abs(fitness[i] - trial_f)
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_w.append(delta)
                new_population[i] = trial
                new_fitness[i] = trial_f
        
        population = new_population
        fitness = new_fitness
        
        # Update history
        if len(S_F) > 0:
            weights = np.array(S_w)
            weights /= weights.sum() + 1e-30
            M_F[h_idx] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[h_idx] = np.sum(weights * np.array(S_CR))
            h_idx = (h_idx + 1) % H
        
        # Check stagnation
        if best < last_best - 1e-12:
            stagnation_counter = 0
            last_best = best
        else:
            stagnation_counter += 1
        
        if stagnation_counter > 20 + dim:
            sorted_idx = np.argsort(fitness)
            keep = max(2, pop_size // 5)
            for idx in sorted_idx[keep:]:
                if elapsed() >= max_time * 0.80:
                    break
                population[idx] = lower + np.random.random(dim) * ranges
                fitness[idx] = evaluate(population[idx])
            stagnation_counter = 0
            M_F[:] = 0.5
            M_CR[:] = 0.5
    
    # --- Phase 3: Local search (coordinate descent + random restarts) ---
    if best_solution is None:
        return best
    
    current = best_solution.copy()
    current_f = best
    step = ranges * 0.05
    
    while elapsed() < max_time * 0.97:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.97:
                return best
            
            trial = current.copy()
            trial[d] = np.clip(trial[d] + step[d], lower[d], upper[d])
            f_trial = evaluate(trial)
            if f_trial < current_f:
                current = trial
                current_f = f_trial
                improved = True
                continue
            
            trial = current.copy()
            trial[d] = np.clip(trial[d] - step[d], lower[d], upper[d])
            f_trial = evaluate(trial)
            if f_trial < current_f:
                current = trial
                current_f = f_trial
                improved = True
                continue
        
        if not improved:
            step *= 0.5
            if np.max(step / (ranges + 1e-30)) < 1e-12:
                # Random restart near best
                current = best_solution + np.random.normal(0, 0.01, dim) * ranges
                current = np.clip(current, lower, upper)
                current_f = evaluate(current)
                step = ranges * 0.02
    
    return best