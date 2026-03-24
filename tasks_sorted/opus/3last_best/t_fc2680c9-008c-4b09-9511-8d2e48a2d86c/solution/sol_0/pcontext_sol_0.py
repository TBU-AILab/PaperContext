#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    # Clip to bounds
    population = np.clip(population, lower, upper)
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.98:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_x = population[i].copy()
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    last_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.85:
            break
        
        generation += 1
        improved_this_gen = False
        
        # Adaptive parameters
        F_base = 0.5 + 0.3 * np.random.random()
        CR_base = 0.1 + 0.8 * np.random.random()
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.85:
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # Mutation: DE/current-to-best/1
            F_i = F_base + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[r1] - population[r2])
            
            # Crossover
            CR_i = CR_base
            cross_points = np.random.random(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
                    improved_this_gen = True
        
        if not improved_this_gen:
            stagnation += 1
        else:
            stagnation = 0
        
        # If stagnating, inject new random individuals
        if stagnation > 10:
            worst_indices = np.argsort(fitness)[-pop_size//3:]
            for idx in worst_indices:
                population[idx] = lower + np.random.random(dim) * ranges
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search around best solution ---
    if best_x is not None:
        # Simple Nelder-Mead implementation
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around best_x
        step = ranges * 0.05
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += step[i] if best_x[i] + step[i] <= upper[i] else -step[i]
        simplex = np.clip(simplex, lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.98:
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_x = simplex[i].copy()
        
        for iteration in range(5000):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.97:
                break
            
            # Sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best:
                best = f_simplex[0]
                best_x = simplex[0].copy()
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, n + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], lower, upper)
                            f_simplex[j] = func(simplex[j])
                else:
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = func(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, n + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], lower, upper)
                            f_simplex[j] = func(simplex[j])
            
            if f_simplex[-1] < best:
                best = f_simplex[-1]
                best_x = simplex[-1].copy() if f_simplex[-1] == min(f_simplex) else best_x
            
            min_idx = np.argmin(f_simplex)
            if f_simplex[min_idx] < best:
                best = f_simplex[min_idx]
                best_x = simplex[min_idx].copy()
    
    # Use remaining time for random restarts of local search
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.99:
            break
        # Random perturbation around best
        scale = 0.01 * ranges
        candidate = best_x + np.random.randn(dim) * scale
        candidate = np.clip(candidate, lower, upper)
        val = func(candidate)
        if val < best:
            best = val
            best_x = candidate.copy()
    
    return best
