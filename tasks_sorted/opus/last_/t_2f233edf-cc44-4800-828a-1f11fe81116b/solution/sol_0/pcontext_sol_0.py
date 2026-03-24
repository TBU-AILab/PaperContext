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
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: CMA-ES inspired + Differential Evolution hybrid ---
    # Sort population by fitness
    sort_idx = np.argsort(fitness)
    population = population[sort_idx]
    fitness = fitness[sort_idx]
    
    # DE parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.95):
            return best
        
        generation += 1
        
        # Adaptive parameters
        F_adaptive = 0.5 + 0.3 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Strategy selection
            strategy = np.random.random()
            
            if strategy < 0.4:
                # DE/best/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = best_params + F_adaptive * (population[r1] - population[r2])
            elif strategy < 0.7:
                # DE/rand/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = population[r1] + F_adaptive * (population[r2] - population[r3])
            elif strategy < 0.85:
                # DE/current-to-best/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = population[i] + F_adaptive * (best_params - population[i]) + F_adaptive * (population[r1] - population[r2])
            else:
                # Local search around best with decreasing radius
                time_ratio = (datetime.now() - start).total_seconds() / max_time
                sigma = (1.0 - time_ratio * 0.9) * (upper - lower) * 0.1
                mutant = best_params + np.random.randn(dim) * sigma
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            CR_adaptive = CR if strategy < 0.85 else 0.3
            for j in range(dim):
                if np.random.random() < CR_adaptive or j == j_rand:
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
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # If stagnating, reinitialize worst half
        if stagnation > 15:
            stagnation = 0
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            half = pop_size // 2
            for i in range(half, pop_size):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
                    return best
                # Generate near best or random
                if np.random.random() < 0.5:
                    sigma = (upper - lower) * 0.3 * np.random.random()
                    population[i] = best_params + np.random.randn(dim) * sigma
                    population[i] = np.clip(population[i], lower, upper)
                else:
                    population[i] = lower + np.random.random(dim) * (upper - lower)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
    
    return best
