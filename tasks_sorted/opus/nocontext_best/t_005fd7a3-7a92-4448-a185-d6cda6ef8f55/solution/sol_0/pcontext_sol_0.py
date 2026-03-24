#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Phase 1: Latin Hypercube Sampling for initial population
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population using LHS-like approach
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * (upper[d] - lower[d])
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # Phase 2: Differential Evolution with restarts and local search
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        improved = False
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.8 + 0.2 * np.random.random()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.85):
                break
            
            # DE/current-to-best/1 strategy
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2 = np.random.choice(indices, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # Mutation: current-to-best/1
            mutant = population[i] + F * (population[best_idx] - population[i]) + F * (population[r1] - population[r2])
            
            # Clip to bounds
            mutant = np.clip(mutant, lower, upper)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved = True
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # If stagnating, inject some random individuals
        if stagnation > 10:
            n_replace = pop_size // 4
            worst_indices = np.argsort(fitness)[-n_replace:]
            for idx in worst_indices:
                population[idx] = lower + np.random.random(dim) * (upper - lower)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # Phase 3: Nelder-Mead local search around best solution
    passed_time = (datetime.now() - start)
    remaining = max_time * 0.95 - passed_time.total_seconds()
    
    if remaining > 0.1 and dim <= 100:
        # Simple Nelder-Mead implementation
        # Initialize simplex around best_params
        scale = 0.05 * (upper - lower)
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_params.copy()
        simplex_fitness = np.zeros(dim + 1)
        simplex_fitness[0] = best
        
        for i in range(dim):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += scale[i]
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            simplex_fitness[i + 1] = func(simplex[i + 1])
            if simplex_fitness[i + 1] < best:
                best = simplex_fitness[i + 1]
                best_params = simplex[i + 1].copy()
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            order = np.argsort(simplex_fitness)
            simplex = simplex[order]
            simplex_fitness = simplex_fitness[order]
            
            if simplex_fitness[0] < best:
                best = simplex_fitness[0]
                best_params = simplex[0].copy()
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if simplex_fitness[0] <= fr < simplex_fitness[-2]:
                simplex[-1] = xr
                simplex_fitness[-1] = fr
            elif fr < simplex_fitness[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    simplex_fitness[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_fitness[-1] = fr
            else:
                # Contraction
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < simplex_fitness[-1]:
                    simplex[-1] = xc
                    simplex_fitness[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        passed_time = (datetime.now() - start)
                        if passed_time >= timedelta(seconds=max_time * 0.95):
                            return best
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        simplex_fitness[i] = func(simplex[i])
                        if simplex_fitness[i] < best:
                            best = simplex_fitness[i]
                            best_params = simplex[i].copy()
    
    return best
