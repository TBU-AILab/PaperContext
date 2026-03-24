#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like approach
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
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
        remaining = max_time - elapsed
        if remaining < max_time * 0.05:
            break
        
        generation += 1
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.90:
                break
            
            # Mutation: DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Adaptive F
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            # Use best individual or one of top individuals
            best_idx = np.random.randint(0, max(1, pop_size // 5))
            
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Crossover
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back into bounds
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
        
        # Check stagnation
        if best >= prev_best - 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Re-sort
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # If stagnating, inject random individuals in the worse half
        if stagnation > 5:
            n_replace = pop_size // 4
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * (upper - lower)
                # Also try perturbation around best
                if np.random.rand() < 0.5 and best_params is not None:
                    scale = (upper - lower) * 0.1 * np.random.rand()
                    population[j] = best_params + scale * np.random.randn(dim)
                    population[j] = np.clip(population[j], lower, upper)
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_params = population[j].copy()
            stagnation = 0
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
    
    # --- Phase 3: Local refinement with Nelder-Mead-like simplex ---
    if best_params is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time - elapsed
        
        if remaining > 0.5:
            # Simple coordinate-wise refinement
            current = best_params.copy()
            current_f = best
            step = (upper - lower) * 0.01
            
            while (datetime.now() - start).total_seconds() < max_time * 0.98:
                improved = False
                for d in range(dim):
                    if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                        break
                    for direction in [1, -1]:
                        trial = current.copy()
                        trial[d] = trial[d] + direction * step[d]
                        trial = np.clip(trial, lower, upper)
                        f_trial = func(trial)
                        if f_trial < current_f:
                            current = trial
                            current_f = f_trial
                            if f_trial < best:
                                best = f_trial
                                best_params = trial.copy()
                            improved = True
                            step[d] *= 1.2
                            break
                
                if not improved:
                    step *= 0.5
                    if np.max(step / (upper - lower + 1e-30)) < 1e-12:
                        break
    
    return best
