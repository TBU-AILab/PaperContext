#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Parameters ---
    pop_size = max(20, min(100, 10 * dim))
    F = 0.8       # differential weight
    CR = 0.9      # crossover rate
    
    # Initialize population
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = population[best_idx].copy()
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    # --- Main DE loop with restarts and local search ---
    generation = 0
    stagnation_counter = 0
    prev_best = best
    
    while elapsed() < max_time * 0.95:
        generation += 1
        
        # Adaptive F and CR
        F_cur = 0.5 + 0.3 * np.random.random()
        CR_cur = 0.8 + 0.2 * np.random.random()
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.95:
                return best
            
            # DE/current-to-best/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            mutant = population[i] + F_cur * (best_x - population[i]) + F_cur * (population[r1] - population[r2])
            
            # Crossover
            trial = np.copy(population[i])
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_cur or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            trial_fit = func(trial)
            
            if trial_fit <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fit
                
                if trial_fit < best:
                    best = trial_fit
                    best_x = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            prev_best = best
        
        # Local search (Nelder-Mead-like simplex around best) periodically
        if generation % 20 == 0 and elapsed() < max_time * 0.85:
            # Simple coordinate-wise local search
            step = (upper - lower) * 0.01
            current = best_x.copy()
            current_fit = best
            for _ in range(3):
                for d in range(dim):
                    if elapsed() >= max_time * 0.90:
                        break
                    for sign in [1, -1]:
                        candidate = current.copy()
                        candidate[d] += sign * step[d]
                        candidate = np.clip(candidate, lower, upper)
                        cfit = func(candidate)
                        if cfit < current_fit:
                            current = candidate
                            current_fit = cfit
                step *= 0.5
            if current_fit < best:
                best = current_fit
                best_x = current.copy()
                # Inject into population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_x.copy()
                fitness[worst_idx] = best
        
        # Restart if stagnated
        if stagnation_counter > 50:
            # Keep best, reinitialize rest
            keep = max(2, pop_size // 5)
            sorted_idx = np.argsort(fitness)
            new_pop = np.random.uniform(lower, upper, (pop_size, dim))
            new_fit = np.full(pop_size, float('inf'))
            for k in range(keep):
                new_pop[k] = population[sorted_idx[k]]
                new_fit[k] = fitness[sorted_idx[k]]
            for k in range(keep, pop_size):
                if elapsed() >= max_time * 0.95:
                    return best
                new_fit[k] = func(new_pop[k])
                if new_fit[k] < best:
                    best = new_fit[k]
                    best_x = new_pop[k].copy()
            population = new_pop
            fitness = new_fit
            stagnation_counter = 0
            prev_best = best
    
    return best
