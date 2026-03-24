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
    pop_size = min(100, max(20, 10 * dim))
    
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
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
            best_params = population[i].copy()
    
    # --- Phase 2: CMA-ES inspired search + Differential Evolution hybrid ---
    # We'll use a restart strategy with Nelder-Mead and DE
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    # Sort population by fitness
    idx = np.argsort(fitness)
    population = population[idx]
    fitness = fitness[idx]
    
    # --- DE/best/1/bin with adaptive parameters ---
    F = 0.8
    CR = 0.9
    generation = 0
    
    while remaining() > 0:
        generation += 1
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.5 + 0.4 * np.random.random()
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            # Mutation: DE/current-to-best/1
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # Use best individual
            mutant = population[i] + F * (population[0] - population[i]) + F * (population[r1] - population[r2])
            mutant = clip(mutant)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial = clip(trial)
            f_trial = func(trial)
            
            if f_trial < new_fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
            
            if f_trial < best:
                best = f_trial
                best_params = trial.copy()
        
        population = new_pop
        fitness = new_fit
        
        # Sort
        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]
        
        # Periodically inject random individuals to avoid stagnation
        if generation % 20 == 0:
            n_replace = max(1, pop_size // 5)
            for k in range(n_replace):
                if remaining() <= 0:
                    return best
                idx_replace = pop_size - 1 - k
                new_ind = lower + np.random.random(dim) * (upper - lower)
                f_new = func(new_ind)
                population[idx_replace] = new_ind
                fitness[idx_replace] = f_new
                if f_new < best:
                    best = f_new
                    best_params = new_ind.copy()
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
    
    # --- Phase 3: Local refinement with Nelder-Mead style simplex on best solution ---
    if best_params is not None and remaining() > 0:
        # Simple coordinate descent refinement
        x = best_params.copy()
        step = (upper - lower) * 0.01
        
        while remaining() > 0:
            improved = False
            for i in range(dim):
                if remaining() <= 0:
                    return best
                
                # Try positive step
                x_new = x.copy()
                x_new[i] = min(x[i] + step[i], upper[i])
                f_new = func(x_new)
                if f_new < best:
                    best = f_new
                    x = x_new
                    best_params = x.copy()
                    improved = True
                    continue
                
                # Try negative step
                x_new = x.copy()
                x_new[i] = max(x[i] - step[i], lower[i])
                f_new = func(x_new)
                if f_new < best:
                    best = f_new
                    x = x_new
                    best_params = x.copy()
                    improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / (upper - lower + 1e-30)) < 1e-10:
                    break
    
    return best
