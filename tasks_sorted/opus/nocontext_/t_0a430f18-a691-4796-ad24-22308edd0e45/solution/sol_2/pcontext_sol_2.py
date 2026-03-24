#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Parameters
    pop_size = min(max(20, 10 * dim), 100)
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    # Initialize population
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(pop_size):
        population[i] = lower + population[i] * (upper - lower)
    
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best = fitness[best_idx]
    best_params = population[best_idx].copy()
    
    # Track stagnation for restarts
    stagnation_counter = 0
    last_best = best
    
    generation = 0
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        
        generation += 1
        
        # Adaptive parameters
        # Use jittered F and CR for diversity
        improved_this_gen = False
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return best
            
            # Select three distinct random indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            
            # current-to-best/1 strategy mixed with rand/1
            if np.random.random() < 0.5:
                # DE/current-to-best/1
                idxs = np.random.choice(candidates, 2, replace=False)
                r1, r2 = idxs
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                mutant = population[i] + Fi * (best_params - population[i]) + Fi * (population[r1] - population[r2])
            else:
                # DE/rand/1
                idxs = np.random.choice(candidates, 3, replace=False)
                r1, r2, r3 = idxs
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                mutant = population[r1] + Fi * (population[r2] - population[r3])
            
            # Clip to bounds
            mutant = np.clip(mutant, lower, upper)
            
            # Binomial crossover
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Evaluate
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved_this_gen = True
        
        # Check stagnation
        if abs(best - last_best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best = best
        
        # If stagnated, do a partial restart keeping the best
        if stagnation_counter > 50:
            stagnation_counter = 0
            # Keep top 20% and reinitialize rest
            sorted_idx = np.argsort(fitness)
            keep = max(2, pop_size // 5)
            for i in range(keep, pop_size):
                # Reinitialize around best with some randomness
                if np.random.random() < 0.5:
                    # Random in full space
                    population[i] = lower + np.random.random(dim) * (upper - lower)
                else:
                    # Near best
                    scale = 0.1 * (upper - lower) * np.random.random()
                    population[i] = best_params + scale * np.random.randn(dim)
                    population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
    
    return best
