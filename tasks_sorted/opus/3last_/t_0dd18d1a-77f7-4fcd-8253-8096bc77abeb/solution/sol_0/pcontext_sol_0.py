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
    
    # Initialize population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[perm[j], i] = (perm[j] + np.random.uniform()) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    best_params = None
    
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
            
            # Select 3 distinct random indices different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r = np.random.choice(idxs, 3, replace=False)
            a, b, c = population[r[0]], population[r[1]], population[r[2]]
            
            # Current-to-best mutation strategy
            best_idx = np.argmin(fitness)
            
            # Adaptive: mix strategies
            if np.random.random() < 0.5:
                # DE/best/1
                mutant = population[best_idx] + F * (b - c)
            else:
                # DE/current-to-best/1
                mutant = population[i] + F * (population[best_idx] - population[i]) + F * (b - c)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        generation += 1
    
    # --- Phase 3: CMA-ES inspired local search around best ---
    if best_params is None:
        best_params = lower + np.random.random(dim) * (upper - lower)
    
    # Nelder-Mead style simplex search
    sigma = 0.1 * (upper - lower)
    
    # Restart-based local search
    n_restarts = 0
    while elapsed() < max_time * 0.95:
        # Local search using adaptive random walk
        current = best_params.copy()
        current_fit = best
        temp_sigma = sigma.copy() * (0.5 ** n_restarts) if n_restarts < 5 else sigma * 0.01
        
        no_improve = 0
        while elapsed() < max_time * 0.95 and no_improve < 50 * dim:
            # Generate candidate
            perturbation = np.random.randn(dim) * temp_sigma
            candidate = current + perturbation
            candidate = np.clip(candidate, lower, upper)
            
            cand_fit = func(candidate)
            if cand_fit < current_fit:
                current = candidate
                current_fit = cand_fit
                if cand_fit < best:
                    best = cand_fit
                    best_params = candidate.copy()
                no_improve = 0
                temp_sigma *= 1.05  # expand on success
            else:
                no_improve += 1
                temp_sigma *= 0.98  # shrink on failure
        
        n_restarts += 1
        # Restart from a point near best with some randomness
        if elapsed() < max_time * 0.95:
            if np.random.random() < 0.3:
                # Random restart
                best_params_restart = lower + np.random.random(dim) * (upper - lower)
                r_fit = func(best_params_restart)
                if r_fit < best:
                    best = r_fit
                    best_params = best_params_restart.copy()
    
    return best
