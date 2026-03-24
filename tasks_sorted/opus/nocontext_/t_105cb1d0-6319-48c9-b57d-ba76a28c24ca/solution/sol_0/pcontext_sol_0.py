#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population using LHS-like approach
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
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    best_params = population[0].copy()
    best = fitness[0]
    
    # --- Phase 2: CMA-ES inspired search + Differential Evolution hybrid ---
    # We'll use a simplified approach combining DE with local search
    
    F = 0.8  # DE mutation factor
    CR = 0.9  # DE crossover rate
    
    generation = 0
    no_improve_count = 0
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time - elapsed
        if remaining < max_time * 0.05:
            break
        
        generation += 1
        improved = False
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.7 + 0.3 * np.random.random()
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.90:
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best + difference vector
            jitter = 0.001 * np.random.randn(dim)
            mutant = population[i] + F * (best_params - population[i]) + F * (population[r1] - population[r2]) + jitter
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved = True
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # If stuck, do local search around best
        if no_improve_count >= 5:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.90:
                break
            
            # Nelder-Mead-like local search: just do coordinate-wise shrinking search
            scale = 0.1 * (upper - lower) * (0.5 ** (no_improve_count // 5))
            for _ in range(dim * 2):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.90:
                    break
                perturbation = np.random.randn(dim) * scale
                candidate = np.clip(best_params + perturbation, lower, upper)
                cf = func(candidate)
                if cf < best:
                    best = cf
                    best_params = candidate.copy()
                    no_improve_count = 0
            
            # Restart worst half of population
            if no_improve_count >= 15:
                half = pop_size // 2
                for k in range(half, pop_size):
                    population[k] = lower + np.random.random(dim) * (upper - lower)
                    # Bias towards best region
                    if np.random.random() < 0.5:
                        population[k] = best_params + 0.3 * (upper - lower) * (np.random.random(dim) - 0.5)
                        population[k] = np.clip(population[k], lower, upper)
                    elapsed = (datetime.now() - start).total_seconds()
                    if elapsed >= max_time * 0.90:
                        break
                    fitness[k] = func(population[k])
                    if fitness[k] < best:
                        best = fitness[k]
                        best_params = population[k].copy()
                no_improve_count = 0
    
    # --- Phase 3: Final local refinement ---
    scale = 0.01 * (upper - lower)
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.98:
            break
        perturbation = np.random.randn(dim) * scale
        candidate = np.clip(best_params + perturbation, lower, upper)
        cf = func(candidate)
        if cf < best:
            best = cf
            best_params = candidate.copy()
        else:
            scale *= 0.95
    
    return best
