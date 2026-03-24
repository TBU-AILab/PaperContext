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
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8  # DE mutation factor
    CR = 0.9  # DE crossover rate
    
    # Initialize population using Latin Hypercube-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_x = population[i].copy()
    
    generation = 0
    
    # Adaptive DE with restart mechanism
    stagnation_counter = 0
    last_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            break
        
        generation += 1
        improved = False
        
        # Sort population by fitness for current-to-best strategies
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        
        # Adaptive parameters
        F_base = 0.5 + 0.3 * np.random.rand()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Strategy selection: mix of DE/rand/1, DE/best/1, DE/current-to-best/1
            strategy = np.random.randint(0, 4)
            
            # Select distinct random indices
            candidates = list(range(pop_size))
            candidates.remove(i)
            idxs = np.random.choice(candidates, 3, replace=False)
            r1, r2, r3 = idxs
            
            F_i = F_base + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            if strategy == 0:
                # DE/rand/1
                mutant = population[r1] + F_i * (population[r2] - population[r3])
            elif strategy == 1:
                # DE/best/1
                mutant = best_x + F_i * (population[r1] - population[r2])
            elif strategy == 2:
                # DE/current-to-best/1
                mutant = population[i] + F_i * (best_x - population[i]) + F_i * (population[r1] - population[r2])
            else:
                # DE/current-to-pbest/1 (p-best from top 20%)
                p = max(2, pop_size // 5)
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - population[r2])
            
            # Binomial crossover
            CR_i = CR + 0.1 * np.random.randn()
            CR_i = np.clip(CR_i, 0.1, 1.0)
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
                    improved = True
        
        # Check stagnation
        if abs(best - last_best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best = best
        
        # Restart worst half if stagnant
        if stagnation_counter > 15 + dim:
            stagnation_counter = 0
            sorted_idx = np.argsort(fitness)
            restart_start = pop_size // 3
            for j in range(restart_start, pop_size):
                idx = sorted_idx[j]
                # Reinitialize around best with decreasing radius
                radius = 0.5 * (upper - lower) * np.random.rand()
                population[idx] = best_x + radius * (2 * np.random.rand(dim) - 1)
                population[idx] = np.clip(population[idx], lower, upper)
                
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
    
    # Local search refinement with Nelder-Mead-like simplex at the end
    if best_x is not None:
        passed_time = (datetime.now() - start)
        remaining = max_time * 0.95 - passed_time.total_seconds()
        
        if remaining > 0.5:
            # Simple coordinate descent refinement
            step = 0.01 * (upper - lower)
            for _ in range(100):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    break
                improved_local = False
                for d in range(dim):
                    for direction in [1, -1]:
                        trial = best_x.copy()
                        trial[d] = trial[d] + direction * step[d]
                        trial[d] = np.clip(trial[d], lower[d], upper[d])
                        
                        passed_time = (datetime.now() - start)
                        if passed_time >= timedelta(seconds=max_time * 0.98):
                            return best
                        
                        trial_fitness = func(trial)
                        if trial_fitness < best:
                            best = trial_fitness
                            best_x = trial.copy()
                            improved_local = True
                            step[d] *= 1.2
                            break
                    else:
                        step[d] *= 0.5
                
                if np.all(step < 1e-15 * (upper - lower)):
                    break
    
    return best
