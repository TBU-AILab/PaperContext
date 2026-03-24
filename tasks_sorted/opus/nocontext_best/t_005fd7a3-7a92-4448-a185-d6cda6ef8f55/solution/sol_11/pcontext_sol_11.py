#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    
    # CMA-ES inspired + Differential Evolution hybrid approach
    
    # Phase 1: Latin Hypercube Sampling for initial exploration
    pop_size = min(100, max(20, 10 * dim))
    
    # Generate initial population
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[perm[j], i] = (perm[j] + np.random.uniform()) / pop_size
    
    # Scale to bounds
    population = lower + population * (upper - lower)
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # Sort population by fitness
    sort_idx = np.argsort(fitness)
    fitness = fitness[sort_idx]
    population = population[sort_idx]
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        improved = False
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.85):
                break
            
            # Adaptive F and CR
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Select strategy based on stagnation
            if stagnation > 5:
                # current-to-best/2 strategy
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = population[i] + Fi * (population[0] - population[i]) + Fi * (population[r1] - population[r2])
            else:
                # best/1 strategy
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                mutant = population[0] + Fi * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.uniform() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            # Evaluate
            trial_fitness = func(trial)
            
            if trial_fitness < new_fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved = True
        
        population = new_population
        fitness = new_fitness
        
        # Re-sort
        sort_idx = np.argsort(fitness)
        fitness = fitness[sort_idx]
        population = population[sort_idx]
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # If stagnation is high, inject random individuals
        if stagnation > 15:
            num_replace = pop_size // 2
            for i in range(num_replace):
                idx = pop_size - 1 - i
                population[idx] = lower + np.random.uniform(0, 1, dim) * (upper - lower)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            sort_idx = np.argsort(fitness)
            fitness = fitness[sort_idx]
            population = population[sort_idx]
            stagnation = 0
    
    # Phase 3: Local search around best solution (Nelder-Mead style)
    simplex_size = dim + 1
    simplex = np.zeros((simplex_size, dim))
    simplex[0] = best_params.copy()
    scale = 0.05 * (upper - lower)
    
    for i in range(1, simplex_size):
        simplex[i] = best_params.copy()
        simplex[i][i - 1 if i - 1 < dim else 0] += scale[i - 1 if i - 1 < dim else 0]
        simplex[i] = np.clip(simplex[i], lower, upper)
    
    simplex_fitness = np.full(simplex_size, float('inf'))
    for i in range(simplex_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        simplex_fitness[i] = func(simplex[i])
        if simplex_fitness[i] < best:
            best = simplex_fitness[i]
            best_params = simplex[i].copy()
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        
        # Sort simplex
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
        if fr < best:
            best = fr
            best_params = xr.copy()
        
        if simplex_fitness[0] <= fr < simplex_fitness[-2]:
            simplex[-1] = xr
            simplex_fitness[-1] = fr
            continue
        
        if fr < simplex_fitness[0]:
            # Expansion
            xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
            fe = func(xe)
            if fe < best:
                best = fe
                best_params = xe.copy()
            if fe < fr:
                simplex[-1] = xe
                simplex_fitness[-1] = fe
            else:
                simplex[-1] = xr
                simplex_fitness[-1] = fr
            continue
        
        # Contraction
        xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
        fc = func(xc)
        if fc < best:
            best = fc
            best_params = xc.copy()
        
        if fc < simplex_fitness[-1]:
            simplex[-1] = xc
            simplex_fitness[-1] = fc
            continue
        
        # Shrink
        for i in range(1, simplex_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return best
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            simplex[i] = np.clip(simplex[i], lower, upper)
            simplex_fitness[i] = func(simplex[i])
            if simplex_fitness[i] < best:
                best = simplex_fitness[i]
                best_params = simplex[i].copy()
    
    return best
