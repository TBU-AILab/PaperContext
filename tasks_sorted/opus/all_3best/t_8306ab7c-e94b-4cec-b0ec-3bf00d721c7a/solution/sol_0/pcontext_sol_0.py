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
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[perm[j], i] = (j + np.random.random()) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    while elapsed() < max_time * 0.70:
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            # Select 3 distinct random individuals different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)
            
            # Best-guided mutation: DE/current-to-best/1
            best_idx = np.argmin(fitness)
            
            # Adaptive F and CR
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[a] - population[b])
            
            # Crossover
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        generation += 1
    
    # --- Phase 3: Nelder-Mead-like local search around best solution ---
    # Use a simple pattern search / coordinate descent refinement
    current = best_params.copy()
    current_fit = best
    
    step_sizes = (upper - lower) * 0.05
    
    while elapsed() < max_time * 0.90:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.90:
                break
            
            # Try positive step
            trial = current.copy()
            trial[d] = min(current[d] + step_sizes[d], upper[d])
            trial_fit = func(trial)
            if trial_fit < current_fit:
                current = trial
                current_fit = trial_fit
                if current_fit < best:
                    best = current_fit
                    best_params = current.copy()
                improved = True
                continue
            
            # Try negative step
            trial = current.copy()
            trial[d] = max(current[d] - step_sizes[d], lower[d])
            trial_fit = func(trial)
            if trial_fit < current_fit:
                current = trial
                current_fit = trial_fit
                if current_fit < best:
                    best = current_fit
                    best_params = current.copy()
                improved = True
                continue
        
        if not improved:
            step_sizes *= 0.5
            if np.max(step_sizes / (upper - lower + 1e-30)) < 1e-12:
                break
    
    # --- Phase 4: Fine-grained random search around best ---
    scale = 0.01
    while elapsed() < max_time * 0.98:
        perturbation = np.random.randn(dim) * (upper - lower) * scale
        trial = np.clip(best_params + perturbation, lower, upper)
        trial_fit = func(trial)
        if trial_fit < best:
            best = trial_fit
            best_params = trial.copy()
            scale *= 1.0  # keep scale
        else:
            scale *= 0.999
            if scale < 1e-15:
                scale = 0.01  # restart scale
    
    return best
