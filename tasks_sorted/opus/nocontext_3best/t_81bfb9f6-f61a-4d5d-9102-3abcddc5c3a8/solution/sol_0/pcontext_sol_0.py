#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Parameters
    pop_size = min(max(20, 10 * dim), 100)
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    # Initialize population using Latin Hypercube-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        population[:, i] = lower[i] + population[:, i] * (upper[i] - lower[i])
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    best_idx = np.argmin(fitness)
    
    generation = 0
    # Main loop: Differential Evolution with adaptive strategies
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time):
            return best
        
        generation += 1
        
        # Adaptive parameters
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time):
                return best
            
            # Strategy selection: rand/1/bin, best/1/bin, current-to-best/1/bin
            strategy = np.random.randint(0, 3)
            
            # Adaptive F and CR with jitter
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Select 3 distinct random indices different from i
            indices = list(range(pop_size))
            indices.remove(i)
            chosen = np.random.choice(indices, 3, replace=False)
            r1, r2, r3 = chosen
            
            if strategy == 0:
                # DE/rand/1
                mutant = population[r1] + Fi * (population[r2] - population[r3])
            elif strategy == 1:
                # DE/best/1
                mutant = best_params + Fi * (population[r2] - population[r3])
            else:
                # DE/current-to-best/1
                mutant = population[i] + Fi * (best_params - population[i]) + Fi * (population[r1] - population[r2])
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounce-back boundary handling
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                elif trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
                # Final clip just in case
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        # Periodically do local search around best (Nelder-Mead-like simplex or random local search)
        if generation % 5 == 0:
            # Local search: small perturbations around best
            for _ in range(dim * 2):
                if (datetime.now() - start) >= timedelta(seconds=max_time):
                    return best
                
                scale = 0.01 * (upper - lower) * (0.99 ** generation)
                perturbation = best_params + np.random.randn(dim) * scale
                perturbation = np.clip(perturbation, lower, upper)
                pf = func(perturbation)
                if pf < best:
                    best = pf
                    best_params = perturbation.copy()
                    # Also inject into population
                    worst_idx = np.argmax(fitness)
                    population[worst_idx] = perturbation.copy()
                    fitness[worst_idx] = pf
    
    return best
