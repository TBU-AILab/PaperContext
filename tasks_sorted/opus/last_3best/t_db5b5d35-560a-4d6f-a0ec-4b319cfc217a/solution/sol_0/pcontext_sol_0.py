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
    
    # ---- Phase 1: Latin Hypercube Sampling for initial population ----
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population via LHS
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
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
            best_params = population[i].copy()
    
    # ---- Phase 2: Differential Evolution with restarts and local search ----
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.85):
            break
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Adaptive F and CR
        F_cur = 0.5 + 0.3 * np.random.random()
        CR_cur = 0.8 + 0.2 * np.random.random()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.85):
                break
            
            # DE/current-to-best/1/bin
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # Mutation: current-to-best
            mutant = population[i] + F_cur * (population[best_idx] - population[i]) + F_cur * (population[a] - population[b])
            
            # Crossover
            trial = np.copy(population[i])
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_cur or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            if f_trial < new_fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
        
        population = new_population
        fitness = new_fitness
        generation += 1
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        # If stagnating, partially reinitialize
        if stagnation_count > 10:
            sorted_idx = np.argsort(fitness)
            keep = max(pop_size // 5, 2)
            for i in range(keep, pop_size):
                population[i] = lower + np.random.random(dim) * (upper - lower)
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.85):
                    break
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
            stagnation_count = 0
    
    # ---- Phase 3: Nelder-Mead local search around best ----
    if best_params is not None:
        # Simple Nelder-Mead implementation
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around best_params
        scale = 0.05 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += scale[i]
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                break
            
            # Sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best:
                best = f_simplex[0]
                best_params = simplex[0].copy()
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                continue
            
            if fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                    break
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                continue
            
            # Contraction
            xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                break
            fc = func(xc)
            if fc < best:
                best = fc
                best_params = xc.copy()
            
            if fc < f_simplex[-1]:
                simplex[-1] = xc
                f_simplex[-1] = fc
                continue
            
            # Shrink
            for i in range(1, n + 1):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    break
                simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lower, upper)
                f_simplex[i] = func(simplex[i])
                if f_simplex[i] < best:
                    best = f_simplex[i]
                    best_params = simplex[i].copy()
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    return best
