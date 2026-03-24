#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    pop_size = min(max(20 * dim, 50), 200)
    
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
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: CMA-ES inspired + Differential Evolution hybrid ---
    # Use DE/current-to-best with adaptive parameters
    
    F = 0.8  # DE mutation factor
    CR = 0.9  # DE crossover rate
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.8 + 0.2 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.85):
                break
            
            # DE/current-to-best/1/bin
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            mutant = population[i] + F * (population[0] - population[i]) + F * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounds handling - bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
                # Clip as safety
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Re-sort
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # If stagnating, reinitialize bottom half
        if stagnation > 10:
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead style) around best ---
    # Simple pattern search refinement
    step = 0.01 * (upper - lower)
    
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
            break
        
        improved = False
        for j in range(dim):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                break
            
            # Try positive direction
            trial = best_params.copy()
            trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_params = trial.copy()
                improved = True
                continue
            
            # Try negative direction
            trial = best_params.copy()
            trial[j] = np.clip(trial[j] - step[j], lower[j], upper[j])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_params = trial.copy()
                improved = True
                continue
        
        if not improved:
            step *= 0.5
            if np.max(step) < 1e-15:
                break
    
    return best
