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
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
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
        if stagnation > 10:
            F = np.random.uniform(0.4, 1.0)
            CR = np.random.uniform(0.1, 1.0)
        
        improved_this_gen = False
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.90:
                break
            
            # Select strategy based on progress
            strategy = np.random.choice(['best1', 'rand1', 'best2', 'currenttobest1'], 
                                         p=[0.3, 0.3, 0.1, 0.3])
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if strategy == 'best1':
                best_idx = np.argmin(fitness_vals)
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = best_params + F * (a - b)
            elif strategy == 'rand1':
                r1, r2, r3 = population[np.random.choice(idxs, 3, replace=False)]
                mutant = r1 + F * (r2 - r3)
            elif strategy == 'best2':
                a, b, c, d = population[np.random.choice(idxs, 4, replace=False)]
                mutant = best_params + F * (a - b) + F * (c - d)
            else:  # currenttobest1
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = population[i] + F * (best_params - population[i]) + F * (a - b)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d_idx in range(dim):
                if trial[d_idx] < lower[d_idx]:
                    trial[d_idx] = lower[d_idx] + np.random.rand() * (population[i][d_idx] - lower[d_idx])
                if trial[d_idx] > upper[d_idx]:
                    trial[d_idx] = upper[d_idx] - np.random.rand() * (upper[d_idx] - population[i][d_idx])
            
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            
            if f_trial <= fitness_vals[i]:
                population[i] = trial
                fitness_vals[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                    improved_this_gen = True
        
        if not improved_this_gen:
            stagnation += 1
        else:
            stagnation = 0
        
        # Restart worst members if stagnating too long
        if stagnation > 30:
            worst_idxs = np.argsort(fitness_vals)[-pop_size//3:]
            for idx in worst_idxs:
                population[idx] = lower + np.random.rand(dim) * (upper - lower)
                fitness_vals[idx] = func(population[idx])
                if fitness_vals[idx] < best:
                    best = fitness_vals[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead-like simplex) around best ---
    if best_params is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time * 0.95 - elapsed
        
        if remaining > 0.5:
            # Simple coordinate-wise local search
            step = (upper - lower) * 0.01
            current = best_params.copy()
            current_f = best
            
            while True:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    break
                
                improved = False
                for d_idx in range(dim):
                    elapsed = (datetime.now() - start).total_seconds()
                    if elapsed >= max_time * 0.95:
                        break
                    
                    # Try positive step
                    trial = current.copy()
                    trial[d_idx] = min(current[d_idx] + step[d_idx], upper[d_idx])
                    f_trial = func(trial)
                    if f_trial < current_f:
                        current = trial
                        current_f = f_trial
                        improved = True
                        if f_trial < best:
                            best = f_trial
                            best_params = trial.copy()
                        continue
                    
                    # Try negative step
                    trial = current.copy()
                    trial[d_idx] = max(current[d_idx] - step[d_idx], lower[d_idx])
                    f_trial = func(trial)
                    if f_trial < current_f:
                        current = trial
                        current_f = f_trial
                        improved = True
                        if f_trial < best:
                            best = f_trial
                            best_params = trial.copy()
                
                if not improved:
                    step *= 0.5
                    if np.max(step / (upper - lower + 1e-30)) < 1e-10:
                        break
    
    return best
