#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # ---- Parameters ----
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8       # differential weight
    CR = 0.9      # crossover probability
    
    # ---- Initialize population ----
    population = np.random.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_sol = population[best_idx].copy()
    if best_val < best:
        best = best_val
    
    generation = 0
    
    # ---- CMA-ES-like restart logic with DE as main engine ----
    def time_remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    def check_time():
        return (datetime.now() - start) < timedelta(seconds=max_time * 0.98)
    
    # Main DE loop
    while check_time():
        generation += 1
        
        # Adaptive parameters
        # Use jittered F and CR for diversity
        for i in range(pop_size):
            if not check_time():
                return best
            
            # Mutation: DE/best/1 with probability 0.5, else DE/rand/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if np.random.random() < 0.5:
                # DE/best/1
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                mutant = best_sol + Fi * (a - b)
            else:
                # DE/rand/1
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                mutant = a + Fi * (b - c)
            
            # Bounce-back boundary handling
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                # Final clip just in case
                mutant[d] = np.clip(mutant[d], lower[d], upper[d])
            
            # Crossover: binomial
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Selection
            trial_fitness = func(trial)
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_sol = trial.copy()
        
        # Local search around best solution periodically
        if generation % 5 == 0 and check_time():
            # Nelder-Mead-like simplex shrink around best
            scale = 0.01 * (upper - lower)
            for _ in range(dim * 2):
                if not check_time():
                    return best
                perturbation = np.random.randn(dim) * scale
                candidate = best_sol + perturbation
                candidate = np.clip(candidate, lower, upper)
                cf = func(candidate)
                if cf < best:
                    best = cf
                    best_sol = candidate.copy()
                    # Update worst in population
                    worst_idx = np.argmax(fitness)
                    population[worst_idx] = candidate.copy()
                    fitness[worst_idx] = cf
        
        # Restart stagnation check: if population diversity is too low, partially reinitialize
        if generation % 50 == 0:
            std = np.std(population, axis=0)
            if np.all(std < 1e-10 * (upper - lower)):
                # Keep best, reinitialize rest
                keep = max(1, pop_size // 5)
                sorted_idx = np.argsort(fitness)
                new_pop = np.random.uniform(lower, upper, size=(pop_size, dim))
                new_pop[:keep] = population[sorted_idx[:keep]]
                population = new_pop
                fitness = np.array([func(ind) if idx >= keep else fitness[sorted_idx[idx]] 
                                   for idx, ind in enumerate(population)])
                bi = np.argmin(fitness)
                if fitness[bi] < best:
                    best = fitness[bi]
                    best_sol = population[bi].copy()
    
    return best
