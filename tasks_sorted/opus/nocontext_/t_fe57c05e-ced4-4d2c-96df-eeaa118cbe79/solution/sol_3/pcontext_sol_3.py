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
    CR = 0.9      # crossover probability
    
    # Initialize population
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = population[best_idx].copy()
    
    def time_remaining():
        return timedelta(seconds=max_time) - (datetime.now() - start)
    
    generation = 0
    
    # --- Main loop: Differential Evolution with restarts and local search ---
    while True:
        if time_remaining().total_seconds() < 0.1:
            return best
        
        generation += 1
        
        # DE/rand/1/bin with optional current-to-best
        indices = np.arange(pop_size)
        
        for i in range(pop_size):
            if time_remaining().total_seconds() < 0.05:
                return best
            
            # Adaptive F and CR
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Select 3 distinct individuals different from i
            candidates = [c for c in range(pop_size) if c != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutation: DE/current-to-best/1
            mutant = population[i] + Fi * (best_x - population[i]) + Fi * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            # Selection
            trial_fitness = func(trial)
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        # Periodically do local search (Nelder-Mead style simplex) around best
        if generation % 5 == 0 and time_remaining().total_seconds() > 1.0:
            # Simple coordinate-wise local search
            step = (upper - lower) * 0.01
            current = best_x.copy()
            current_fit = best
            
            improved = True
            while improved and time_remaining().total_seconds() > 0.5:
                improved = False
                for j in range(dim):
                    if time_remaining().total_seconds() < 0.2:
                        break
                    
                    # Try positive step
                    trial = current.copy()
                    trial[j] = min(current[j] + step[j], upper[j])
                    tf = func(trial)
                    if tf < current_fit:
                        current = trial
                        current_fit = tf
                        improved = True
                        continue
                    
                    # Try negative step
                    trial = current.copy()
                    trial[j] = max(current[j] - step[j], lower[j])
                    tf = func(trial)
                    if tf < current_fit:
                        current = trial
                        current_fit = tf
                        improved = True
                        continue
                
                step *= 0.5  # Reduce step size
                if np.max(step) < 1e-12:
                    break
            
            if current_fit < best:
                best = current_fit
                best_x = current.copy()
                # Inject back into population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_x.copy()
                fitness[worst_idx] = best
        
        # Restart worst half occasionally to maintain diversity
        if generation % 50 == 0:
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for idx in sorted_idx[half:]:
                population[idx] = np.random.uniform(lower, upper)
                if time_remaining().total_seconds() < 0.2:
                    return best
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
    
    return best
