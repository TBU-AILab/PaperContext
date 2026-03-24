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
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness_vals = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness_vals[i] = f
        if f < best:
            best = f
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.90:
            break
        
        generation += 1
        
        # Adaptive parameters
        F_base = 0.5 + 0.3 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness_vals.copy()
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.88:
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness_vals)
            
            F_i = F_base + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[a] - population[b])
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            
            if f_trial <= fitness_vals[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
        
        population = new_population
        fitness_vals = new_fitness
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnating
        if stagnation > 15:
            stagnation = 0
            sorted_idx = np.argsort(fitness_vals)
            half = pop_size // 2
            for j in sorted_idx[half:]:
                population[j] = lower + np.random.random(dim) * (upper - lower)
                # Don't evaluate yet, will be evaluated next generation
                fitness_vals[j] = float('inf')
    
    # --- Phase 3: Nelder-Mead-like local search around best ---
    if best_params is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time * 0.98 - elapsed
        
        if remaining > 0.1:
            # Simple coordinate-wise search
            step = (upper - lower) * 0.01
            x = best_params.copy()
            f_x = best
            
            for _ in range(50):
                improved = False
                for d in range(dim):
                    if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                        return best
                    
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x[d] + direction * step[d], lower[d], upper[d])
                        f_new = func(x_new)
                        if f_new < f_x:
                            x = x_new
                            f_x = f_new
                            improved = True
                            if f_x < best:
                                best = f_x
                                best_params = x.copy()
                
                if not improved:
                    step *= 0.5
                    if np.max(step / (upper - lower + 1e-30)) < 1e-10:
                        break
    
    return best
