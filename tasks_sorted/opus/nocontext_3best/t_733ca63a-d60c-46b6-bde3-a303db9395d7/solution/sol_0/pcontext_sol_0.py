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
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    best_solution = population[0].copy()
    best = fitness[0]
    
    # --- Phase 2: CMA-ES inspired search + Differential Evolution hybrid ---
    # Use a restart strategy with progressively smaller search regions
    
    F = 0.8   # DE mutation factor
    CR = 0.9  # DE crossover rate
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        
        remaining = max_time - elapsed
        if remaining < 0.01:
            return best
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.8 + 0.2 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best + random differential
            jitter = 0.001 * np.random.randn(dim)
            mutant = population[i] + F * (best_solution - population[i]) + F * (population[a] - population[b]) + jitter
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Boundary handling - bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
                # Final clip
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_solution = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        generation += 1
        
        # Check stagnation
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # If stagnated, do a partial restart keeping best individuals
        if stagnation > 15 + dim:
            n_keep = max(3, pop_size // 5)
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Reinitialize the rest around the best or randomly
            for i in range(n_keep, pop_size):
                if np.random.random() < 0.5:
                    # Local restart around best
                    scale = 0.1 * (upper - lower) * np.random.random()
                    population[i] = best_solution + scale * np.random.randn(dim)
                    population[i] = np.clip(population[i], lower, upper)
                else:
                    # Random restart
                    population[i] = lower + np.random.random(dim) * (upper - lower)
                
                if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                    return best
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_solution = population[i].copy()
            
            stagnation = 0
    
    return best
