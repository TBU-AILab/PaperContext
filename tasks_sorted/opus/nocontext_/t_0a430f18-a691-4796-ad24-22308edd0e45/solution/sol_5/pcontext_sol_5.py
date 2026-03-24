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
    
    # ---- Parameters ----
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8      # differential weight
    CR = 0.9     # crossover probability
    
    # ---- Initialize population ----
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_params = population[best_idx].copy()
    
    # ---- CMA-ES-like restart logic combined with DE ----
    generation = 0
    stagnation_counter = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        generation += 1
        
        # Adaptive parameters
        # Jitter F and CR slightly per generation
        F_gen = F + 0.1 * (np.random.random() - 0.5)
        CR_gen = CR + 0.1 * (np.random.random() - 0.5)
        CR_gen = np.clip(CR_gen, 0.1, 1.0)
        F_gen = np.clip(F_gen, 0.4, 1.2)
        
        new_population = np.empty_like(population)
        new_fitness = np.empty(pop_size)
        
        # Sort population by fitness for current-to-best strategies
        sorted_indices = np.argsort(fitness)
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # DE/current-to-pbest/1 strategy
            # Pick pbest from top 20%
            p_best_size = max(2, pop_size // 5)
            p_best_idx = sorted_indices[np.random.randint(0, p_best_size)]
            
            # Pick 2 random distinct indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # Per-individual F jitter
            Fi = F_gen + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Binomial crossover
            CRi = CR_gen + 0.05 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                # Final clip just in case
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
            
            if trial_fitness < best:
                best = trial_fitness
                best_params = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best = best
        
        # If stagnating, do a partial restart keeping best individuals
        if stagnation_counter > 50:
            stagnation_counter = 0
            # Keep top 20% and reinitialize the rest
            keep = max(2, pop_size // 5)
            sorted_idx = np.argsort(fitness)
            
            new_pop = np.random.uniform(lower, upper, (pop_size, dim))
            new_fit = np.full(pop_size, float('inf'))
            
            for k in range(keep):
                new_pop[k] = population[sorted_idx[k]]
                new_fit[k] = fitness[sorted_idx[k]]
            
            # Also add some local perturbations around best
            if best_params is not None:
                scale = (upper - lower) * 0.1
                for k in range(keep, min(2 * keep, pop_size)):
                    new_pop[k] = best_params + scale * np.random.randn(dim)
                    new_pop[k] = np.clip(new_pop[k], lower, upper)
            
            # Evaluate new individuals
            for k in range(keep, pop_size):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                new_fit[k] = func(new_pop[k])
                if new_fit[k] < best:
                    best = new_fit[k]
                    best_params = new_pop[k].copy()
            
            population = new_pop
            fitness = new_fit
    
    return best
