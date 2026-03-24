#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    
    # Parameters for CMA-ES-like search
    pop_size = 4 + int(3 * np.log(dim))
    if pop_size < 10:
        pop_size = 10
    
    # Initialize with Latin Hypercube-like sampling
    best_x = None
    
    # Initial random sampling phase
    n_init = max(pop_size * 2, 50)
    for i in range(n_init):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        x = np.random.uniform(lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
    
    if best_x is None:
        best_x = np.random.uniform(lower, upper)
    
    # Differential Evolution phase
    # Initialize population
    population = []
    fitness = []
    
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        x = np.random.uniform(lower, upper)
        f = func(x)
        population.append(x.copy())
        fitness.append(f)
        if f < best:
            best = f
            best_x = x.copy()
    
    population = np.array(population)
    fitness = np.array(fitness)
    
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    generation = 0
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.90):
            break
        
        generation += 1
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.90):
                return best
            
            # Mutation: DE/rand-to-best/1
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            # Adaptive F
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.0)
            
            mutant = population[a] + Fi * (best_x - population[a]) + Fi * (population[b] - population[c])
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bound handling - reflection
            for d in range(dim):
                while trial[d] < lower[d] or trial[d] > upper[d]:
                    if trial[d] < lower[d]:
                        trial[d] = 2 * lower[d] - trial[d]
                    if trial[d] > upper[d]:
                        trial[d] = 2 * upper[d] - trial[d]
            
            trial = np.clip(trial, lower, upper)
            
            # Selection
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                population[i] = trial.copy()
                fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
    
    # Local search refinement with Nelder-Mead-like simplex
    passed_time = (datetime.now() - start)
    remaining = max_time - passed_time.total_seconds()
    
    if remaining > 0.5:
        # Simple coordinate descent refinement
        step = (upper - lower) * 0.01
        improved = True
        while improved:
            improved = False
            for d in range(dim):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    return best
                
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[d] = trial[d] + direction * step[d]
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    if f_trial < best:
                        best = f_trial
                        best_x = trial.copy()
                        improved = True
    
    return best
