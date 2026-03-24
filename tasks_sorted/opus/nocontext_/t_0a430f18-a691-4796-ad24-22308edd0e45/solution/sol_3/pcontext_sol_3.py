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
    pop_size = max(20, 10 * dim)
    F = 0.8  # differential weight
    CR = 0.9  # crossover probability
    
    # Initialize population
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = population[best_idx].copy()
    
    def time_remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    # --- Main DE loop with restarts and local search ---
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        if time_remaining() < 0.05:
            return best
        
        generation += 1
        
        # Adaptive F and CR
        F_adapted = F + 0.1 * np.random.randn()
        F_adapted = np.clip(F_adapted, 0.1, 1.5)
        
        # DE/current-to-best/1/bin with jitter
        for i in range(pop_size):
            if time_remaining() < 0.05:
                return best
            
            # Select 3 distinct individuals different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: DE/current-to-best/1
            Fi = np.clip(F + 0.1 * np.random.randn(), 0.1, 1.5)
            mutant = population[i] + Fi * (best_x - population[i]) + Fi * (population[r1] - population[r2])
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back bounds handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        # Local search on best solution periodically
        if generation % 5 == 0 and time_remaining() > 0.5:
            # Nelder-Mead-like simplex local search (coordinate-wise refinement)
            local_x = best_x.copy()
            local_f = best
            step = (upper - lower) * 0.01
            
            for _ in range(3):  # few iterations of coordinate descent
                for d in range(dim):
                    if time_remaining() < 0.1:
                        return best
                    
                    # Try positive step
                    trial_x = local_x.copy()
                    trial_x[d] = np.clip(local_x[d] + step[d], lower[d], upper[d])
                    tf = func(trial_x)
                    if tf < local_f:
                        local_x = trial_x
                        local_f = tf
                        step[d] *= 1.5
                        continue
                    
                    # Try negative step
                    trial_x = local_x.copy()
                    trial_x[d] = np.clip(local_x[d] - step[d], lower[d], upper[d])
                    tf = func(trial_x)
                    if tf < local_f:
                        local_x = trial_x
                        local_f = tf
                        step[d] *= 1.5
                        continue
                    
                    step[d] *= 0.5
            
            if local_f < best:
                best = local_f
                best_x = local_x.copy()
                # Inject back into population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_x.copy()
                fitness[worst_idx] = best
        
        # Check stagnation and restart partially
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 50:
            # Restart half the population
            n_restart = pop_size // 2
            worst_indices = np.argsort(fitness)[-n_restart:]
            for idx in worst_indices:
                population[idx] = np.random.uniform(lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
            stagnation = 0
    
    return best
