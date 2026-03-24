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
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like approach
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
            best_x = population[i].copy()
    
    # --- Phase 2: CMA-ES inspired + Differential Evolution hybrid ---
    # Use DE/current-to-best with adaptive parameters
    
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    
    # Sort population
    sort_idx = np.argsort(fitness_vals)
    population = population[sort_idx]
    fitness_vals = fitness_vals[sort_idx]
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.90):
            break
        
        generation += 1
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.8 + 0.2 * np.random.random()
        
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.90):
                return best
            
            # DE/current-to-best/1/bin
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            best_idx = 0  # population is sorted
            mutant = population[i] + F * (population[best_idx] - population[i]) + F * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            if f_trial < new_fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        population = new_pop
        fitness_vals = new_fit
        
        # Re-sort
        sort_idx = np.argsort(fitness_vals)
        population = population[sort_idx]
        fitness_vals = fitness_vals[sort_idx]
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # If stagnating, restart worst half
        if stagnation > 15:
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = lower + np.random.random(dim) * (upper - lower)
                # Local perturbation around best for some
                if np.random.random() < 0.5 and best_x is not None:
                    scale = 0.1 * (upper - lower) * np.random.random()
                    population[i] = best_x + scale * np.random.randn(dim)
                    population[i] = np.clip(population[i], lower, upper)
                fitness_vals[i] = func(population[i])
                if fitness_vals[i] < best:
                    best = fitness_vals[i]
                    best_x = population[i].copy()
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx]
            fitness_vals = fitness_vals[sort_idx]
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead style) around best ---
    if best_x is not None:
        # Simple pattern search refinement
        step = 0.01 * (upper - lower)
        while True:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
                break
            improved = False
            for j in range(dim):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
                    break
                for sign in [1, -1]:
                    trial = best_x.copy()
                    trial[j] += sign * step[j]
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    if f_trial < best:
                        best = f_trial
                        best_x = trial.copy()
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step) < 1e-15:
                    break
    
    return best
