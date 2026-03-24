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
    F = 0.8   # differential weight
    CR = 0.9  # crossover probability
    
    # ---- Initialize population ----
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_params = population[best_idx].copy()
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    # ---- Main loop: Differential Evolution with restarts and local search ----
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        if time_left() < 0.05:
            return best
        
        generation += 1
        
        # Adaptive parameters
        # Jitter F slightly
        F_arr = 0.5 + 0.3 * np.random.rand(pop_size)
        
        # DE/current-to-best/1 with DE/rand/1 hybrid
        indices = np.arange(pop_size)
        
        for i in range(pop_size):
            if time_left() < 0.05:
                return best
            
            # Select 3 distinct random indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            Fi = F_arr[i]
            
            # Strategy: mix of DE/rand/1 and DE/current-to-best/1
            if np.random.rand() < 0.5:
                # DE/current-to-best/1
                mutant = population[i] + Fi * (best_params - population[i]) + Fi * (population[r1] - population[r2])
            else:
                # DE/rand/1
                mutant = population[r1] + Fi * (population[r2] - population[r3])
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back into bounds
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Local search on best solution periodically
        if generation % 5 == 0 and time_left() > 0.5:
            # Nelder-Mead-like simplex or just a local random search around best
            local_best = best_params.copy()
            local_best_fit = best
            scale = (upper - lower) * 0.01
            
            for _ in range(dim * 5):
                if time_left() < 0.1:
                    break
                perturbation = np.random.randn(dim) * scale
                candidate = np.clip(local_best + perturbation, lower, upper)
                cand_fit = func(candidate)
                if cand_fit < local_best_fit:
                    local_best = candidate
                    local_best_fit = cand_fit
                    scale *= 1.1  # expand if improving
                else:
                    scale *= 0.95  # shrink
            
            if local_best_fit < best:
                best = local_best_fit
                best_params = local_best.copy()
                # Inject into population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = local_best
                fitness[worst_idx] = local_best_fit
        
        # Restart mechanism if stagnated
        if stagnation > 20:
            # Keep best, reinitialize rest
            n_keep = max(1, pop_size // 5)
            sorted_idx = np.argsort(fitness)
            keep_idx = sorted_idx[:n_keep]
            
            new_pop = np.random.uniform(lower, upper, (pop_size, dim))
            new_fit = np.full(pop_size, np.inf)
            
            for j, ki in enumerate(keep_idx):
                new_pop[j] = population[ki]
                new_fit[j] = fitness[ki]
            
            for j in range(n_keep, pop_size):
                if time_left() < 0.1:
                    return best
                new_fit[j] = func(new_pop[j])
                if new_fit[j] < best:
                    best = new_fit[j]
                    best_params = new_pop[j].copy()
            
            population = new_pop
            fitness = new_fit
            stagnation = 0
    
    return best
