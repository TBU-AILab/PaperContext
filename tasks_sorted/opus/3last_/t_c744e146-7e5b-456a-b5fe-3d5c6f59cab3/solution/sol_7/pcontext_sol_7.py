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
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(100, max(20, dim * 10))
    pop_size = max(20, min(50, dim * 5))
    
    # Initialize population
    population = []
    fitness_vals = []
    
    for i in range(n_init):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        params = np.array([np.random.uniform(low, high) for low, high in bounds])
        f = func(params)
        population.append(params.copy())
        fitness_vals.append(f)
        if f < best:
            best = f
            best_params = params.copy()
    
    # Keep best pop_size individuals
    indices = np.argsort(fitness_vals)[:pop_size]
    population = [population[i] for i in indices]
    fitness_vals = [fitness_vals[i] for i in indices]
    
    # Phase 2: Differential Evolution + Local Search hybrid
    F = 0.8  # DE mutation factor
    CR = 0.9  # DE crossover rate
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        
        # Adaptive parameters
        F_adapt = 0.5 + 0.3 * np.random.random()
        CR_adapt = 0.7 + 0.3 * np.random.random()
        
        new_population = []
        new_fitness = []
        
        for i in range(len(population)):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.85):
                break
            
            # DE/rand/1/bin with optional best guidance
            idxs = list(range(len(population)))
            idxs.remove(i)
            
            if np.random.random() < 0.5 and best_params is not None:
                # DE/current-to-best/1
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = population[i] + F_adapt * (best_params - population[i]) + F_adapt * (population[a] - population[b])
            else:
                # DE/rand/1
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F_adapt * (population[b] - population[c])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_adapt or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounds handling - bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                elif trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
                # Clip as safety net
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            f_trial = func(trial)
            
            if f_trial <= fitness_vals[i]:
                new_population.append(trial)
                new_fitness.append(f_trial)
            else:
                new_population.append(population[i])
                new_fitness.append(fitness_vals[i])
            
            if f_trial < best:
                best = f_trial
                best_params = trial.copy()
        
        if len(new_population) == len(population):
            population = new_population
            fitness_vals = new_fitness
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # If stagnating, inject random individuals
        if stagnation > 10:
            n_replace = max(1, len(population) // 4)
            worst_idxs = np.argsort(fitness_vals)[-n_replace:]
            for idx in worst_idxs:
                params = lower + np.random.random(dim) * (upper - lower)
                f = func(params)
                population[idx] = params
                fitness_vals[idx] = f
                if f < best:
                    best = f
                    best_params = params.copy()
            stagnation = 0
    
    # Phase 3: Nelder-Mead-like local search around best
    if best_params is not None:
        step = (upper - lower) * 0.01
        current = best_params.copy()
        current_f = best
        
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                break
            
            improved = False
            for j in range(dim):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    break
                
                # Try positive step
                trial = current.copy()
                trial[j] = min(upper[j], current[j] + step[j])
                f_trial = func(trial)
                if f_trial < current_f:
                    current = trial
                    current_f = f_trial
                    if f_trial < best:
                        best = f_trial
                        best_params = trial.copy()
                    improved = True
                    continue
                
                # Try negative step
                trial = current.copy()
                trial[j] = max(lower[j], current[j] - step[j])
                f_trial = func(trial)
                if f_trial < current_f:
                    current = trial
                    current_f = f_trial
                    if f_trial < best:
                        best = f_trial
                        best_params = trial.copy()
                    improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / (upper - lower + 1e-30)) < 1e-10:
                    break
    
    return best
