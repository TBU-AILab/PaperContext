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
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
            best_x = population[i].copy()
    
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
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.8 + 0.2 * np.random.random()
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.88:
                break
            
            # DE/current-to-best/1 with jitter
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2 = np.random.choice(indices, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # Mutation: current-to-best/1
            mutant = population[i] + F * (population[best_idx] - population[i]) + F * (population[r1] - population[r2])
            
            # Crossover
            cross_points = np.random.random(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            # Evaluate
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        # Check stagnation
        if best >= prev_best - 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # If stagnant, partially reinitialize worst members
        if stagnation > 15:
            stagnation = 0
            worst_indices = np.argsort(fitness)[-pop_size // 3:]
            for idx in worst_indices:
                population[idx] = lower + np.random.random(dim) * (upper - lower)
                # Small perturbation around best sometimes
                if np.random.random() < 0.5 and best_x is not None:
                    scale = 0.1 * (upper - lower)
                    population[idx] = best_x + np.random.randn(dim) * scale
                    population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
    
    # --- Phase 3: Nelder-Mead-like local search around best ---
    if best_x is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time * 0.98 - elapsed
        
        if remaining > 0.1:
            # Simple coordinate-wise search with shrinking step
            step = 0.01 * (upper - lower)
            x_current = best_x.copy()
            f_current = best
            
            for iteration in range(50):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.97:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        x_trial = x_current.copy()
                        x_trial[d] = x_current[d] + sign * step[d]
                        x_trial = np.clip(x_trial, lower, upper)
                        f_trial = func(x_trial)
                        if f_trial < f_current:
                            x_current = x_trial
                            f_current = f_trial
                            improved = True
                            if f_current < best:
                                best = f_current
                                best_x = x_current.copy()
                            break
                if not improved:
                    step *= 0.5
                    if np.max(step / (upper - lower + 1e-30)) < 1e-10:
                        break
    
    return best
