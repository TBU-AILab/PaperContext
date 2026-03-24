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
    
    # Parameters
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    # Initialize population using Latin Hypercube-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        population[:, i] = lower[i] + population[:, i] * (upper[i] - lower[i])
    
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate initial population
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_x = population[i].copy()
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        # Adaptive parameters
        generation += 1
        
        # Track stagnation for restart
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # Restart if stagnated (partial restart keeping best)
        if stagnation > max(50, 20 * dim):
            stagnation = 0
            # Keep best solution, reinitialize rest
            best_idx = np.argmin(fitness)
            for i in range(pop_size):
                if i != best_idx:
                    # Reinitialize around best or randomly
                    if np.random.random() < 0.5:
                        # Local reinitialization around best
                        scale = 0.1 * (upper - lower)
                        population[i] = best_x + scale * np.random.randn(dim)
                        population[i] = np.clip(population[i], lower, upper)
                    else:
                        population[i] = lower + np.random.rand(dim) * (upper - lower)
                    fitness[i] = func(population[i])
                    if fitness[i] < best:
                        best = fitness[i]
                        best_x = population[i].copy()
        
        # DE/current-to-pbest/1 with archive (SHADE-like simplified)
        sorted_indices = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Adaptive F and CR with jitter
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + F, 0.1, 1.0)
            CRi = np.clip(np.random.normal(CR, 0.1), 0.0, 1.0)
            
            # Select p-best
            p_best_idx = sorted_indices[np.random.randint(0, p_best_size)]
            
            # Select two random distinct indices
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # Mutation: current-to-pbest/1
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            
            mutant = np.clip(mutant, lower, upper)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(0, dim)
            for d in range(dim):
                if np.random.random() < CRi or d == j_rand:
                    trial[d] = mutant[d]
            
            # Selection
            trial_fitness = func(trial)
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        population = new_population
        fitness = new_fitness
    
    return best
