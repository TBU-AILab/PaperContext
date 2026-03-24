#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
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
        if elapsed >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    # Sort population by fitness
    sort_idx = np.argsort(fitness)
    population = population[sort_idx]
    fitness = fitness[sort_idx]
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.90:
            break
        
        generation += 1
        improved_this_gen = False
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.90:
                break
            
            # Adaptive F and CR
            Fi = np.random.normal(0.8, 0.1)
            Fi = np.clip(Fi, 0.4, 1.2)
            CRi = np.random.normal(0.9, 0.1)
            CRi = np.clip(CRi, 0.1, 1.0)
            
            # Strategy selection: use multiple strategies
            strategy = np.random.randint(0, 3)
            
            if strategy == 0:
                # DE/best/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = best_params + Fi * (population[idxs[0]] - population[idxs[1]])
            elif strategy == 1:
                # DE/rand/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                mutant = population[idxs[0]] + Fi * (population[idxs[1]] - population[idxs[2]])
            else:
                # DE/current-to-best/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = population[i] + Fi * (best_params - population[i]) + Fi * (population[idxs[0]] - population[idxs[1]])
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounce-back boundary handling
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
            
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved_this_gen = True
        
        if not improved_this_gen:
            stagnation += 1
        else:
            stagnation = 0
        
        # If stagnant, inject some random individuals near the best
        if stagnation > 10:
            n_replace = pop_size // 4
            worst_idx = np.argsort(fitness)[-n_replace:]
            for idx in worst_idx:
                scale = 0.1 * ranges * (0.5 ** (stagnation // 10))
                population[idx] = best_params + np.random.normal(0, 1, dim) * scale
                population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead simplex) around best ---
    if best_params is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time * 0.98 - elapsed
        
        if remaining > 0.1:
            # Simple Nelder-Mead
            n = dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Initialize simplex around best
            scale = 0.05 * ranges
            simplex = np.zeros((n + 1, n))
            simplex_fitness = np.zeros(n + 1)
            simplex[0] = best_params.copy()
            simplex_fitness[0] = best
            
            for i in range(n):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.98:
                    return best
                simplex[i + 1] = best_params.copy()
                simplex[i + 1][i] += scale[i] if best_params[i] + scale[i] <= upper[i] else -scale[i]
                simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
                simplex_fitness[i + 1] = func(simplex[i + 1])
                if simplex_fitness[i + 1] < best:
                    best = simplex_fitness[i + 1]
                    best_params = simplex[i + 1].copy()
            
            while True:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.98:
                    break
                
                # Sort
                order = np.argsort(simplex_fitness)
                simplex = simplex[order]
                simplex_fitness = simplex_fitness[order]
                
                if simplex_fitness[0] < best:
                    best = simplex_fitness[0]
                    best_params = simplex[0].copy()
                
                # Centroid (excluding worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
                fr = func(xr)
                
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.98:
                    if fr < best:
                        best = fr
                    break
                
                if simplex_fitness[0] <= fr < simplex_fitness[-2]:
                    simplex[-1] = xr
                    simplex_fitness[-1] = fr
                elif fr < simplex_fitness[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_fitness[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_fitness[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = func(xc)
                    if fc < simplex_fitness[-1]:
                        simplex[-1] = xc
                        simplex_fitness[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            elapsed = (datetime.now() - start).total_seconds()
                            if elapsed >= max_time * 0.98:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            simplex_fitness[i] = func(simplex[i])
                
                if simplex_fitness[0] < best:
                    best = simplex_fitness[0]
                    best_params = simplex[0].copy()
    
    return best
