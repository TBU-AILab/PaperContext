#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate LHS-like initial population
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    # Track stagnation for restarts
    stagnation_counter = 0
    last_best = best
    
    generation = 0
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.90):
            break
        
        generation += 1
        
        # Adaptive F and CR
        F_base = 0.5 + 0.3 * np.random.random()
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Sort population by fitness for current-to-best strategies
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.90):
                return best
            
            # Mutation strategy selection (randomly pick among strategies)
            strategy = np.random.randint(0, 4)
            
            # Select random indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            
            if strategy == 0:
                # DE/best/1
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F_i = F_base
                mutant = population[best_idx] + F_i * (population[r1] - population[r2])
            elif strategy == 1:
                # DE/rand/1
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                F_i = F_base
                mutant = population[r1] + F_i * (population[r2] - population[r3])
            elif strategy == 2:
                # DE/current-to-best/1
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F_i = F_base
                mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[r1] - population[r2])
            else:
                # DE/rand/2
                if len(candidates) >= 5:
                    r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
                    F_i = F_base
                    mutant = population[r1] + F_i * (population[r2] - population[r3]) + F_i * (population[r4] - population[r5])
                else:
                    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                    F_i = F_base
                    mutant = population[r1] + F_i * (population[r2] - population[r3])
            
            # Binomial crossover
            CR_i = 0.1 + 0.8 * np.random.random()
            cross_points = np.random.random(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                # Clip as safety
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if best < last_best - 1e-12:
            stagnation_counter = 0
            last_best = best
        else:
            stagnation_counter += 1
        
        # Partial restart if stagnated
        if stagnation_counter > 15 + dim:
            sorted_idx = np.argsort(fitness)
            # Keep top 20%
            keep = max(2, pop_size // 5)
            for idx in sorted_idx[keep:]:
                population[idx] = lower + np.random.random(dim) * (upper - lower)
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.90):
                    return best
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
            stagnation_counter = 0
    
    # --- Phase 3: Local search (Nelder-Mead-like) around best solution ---
    sorted_idx = np.argsort(fitness)
    best_solution = population[sorted_idx[0]].copy()
    
    # Simple pattern search around best
    step = (upper - lower) * 0.01
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        
        improved = False
        for d in range(dim):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return best
            
            # Try positive step
            trial = best_solution.copy()
            trial[d] = np.clip(trial[d] + step[d], lower[d], upper[d])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_solution = trial.copy()
                improved = True
                continue
            
            # Try negative step
            trial = best_solution.copy()
            trial[d] = np.clip(trial[d] - step[d], lower[d], upper[d])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_solution = trial.copy()
                improved = True
                continue
        
        if not improved:
            step *= 0.5
            if np.max(step / (upper - lower + 1e-30)) < 1e-10:
                break
    
    return best
