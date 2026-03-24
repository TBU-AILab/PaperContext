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
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
            best_x = population[i].copy()
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    last_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.90:
            break
        
        generation += 1
        
        # Adaptive parameters
        F_cur = 0.5 + 0.3 * np.random.random()
        CR_cur = 0.8 + 0.2 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Sort population by fitness for current-to-best strategy
        sorted_idx = np.argsort(fitness)
        x_best_local = population[sorted_idx[0]].copy()
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.90:
                return best
            
            # DE/current-to-pbest/1 strategy
            p_best_size = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p_best_size)]
            
            # Select 2 distinct random indices != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r = np.random.choice(candidates, 2, replace=False)
            r1, r2 = r[0], r[1]
            
            # Mutation: current-to-pbest
            mutant = population[i] + F_cur * (population[p_best_idx] - population[i]) + F_cur * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_cur or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            # Evaluate
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_x = trial.copy()
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1
        
        # If stagnated, do partial restart keeping best individuals
        if stagnation > 15 + dim:
            stagnation = 0
            sorted_idx = np.argsort(fitness)
            keep = max(3, pop_size // 5)
            for i in range(keep, pop_size):
                # Reinitialize around best with shrinking radius, or random
                if np.random.random() < 0.5:
                    radius = (upper - lower) * np.random.uniform(0.01, 0.3)
                    population[i] = best_x + radius * np.random.randn(dim)
                    population[i] = np.clip(population[i], lower, upper)
                else:
                    population[i] = lower + np.random.random(dim) * (upper - lower)
                
                if (datetime.now() - start).total_seconds() >= max_time * 0.90:
                    return best
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_x = population[i].copy()
    
    # --- Phase 3: Nelder-Mead local search around best ---
    if best_x is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time * 0.95 - elapsed
        
        if remaining > 0.1:
            # Simple Nelder-Mead
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = best_x.copy()
            scale = (upper - lower) * 0.05
            for i in range(n):
                simplex[i + 1] = best_x.copy()
                simplex[i + 1][i] += scale[i] if scale[i] > 1e-10 else 0.01
                simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            
            f_simplex = np.zeros(n + 1)
            for i in range(n + 1):
                if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                    return best
                f_simplex[i] = func(simplex[i])
                if f_simplex[i] < best:
                    best = f_simplex[i]
                    best_x = simplex[i].copy()
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            
            for iteration in range(5000):
                if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                    return best
                
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                if f_simplex[0] < best:
                    best = f_simplex[0]
                    best_x = simplex[0].copy()
                
                centroid = np.mean(simplex[:n], axis=0)
                
                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[n]), lower, upper)
                fr = func(xr)
                if fr < best:
                    best = fr
                    best_x = xr.copy()
                
                if f_simplex[0] <= fr < f_simplex[n - 1]:
                    simplex[n] = xr
                    f_simplex[n] = fr
                elif fr < f_simplex[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                    fe = func(xe)
                    if fe < best:
                        best = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[n] = xe
                        f_simplex[n] = fe
                    else:
                        simplex[n] = xr
                        f_simplex[n] = fr
                else:
                    # Contraction
                    if fr < f_simplex[n]:
                        xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    else:
                        xc = np.clip(centroid + rho * (simplex[n] - centroid), lower, upper)
                    fc = func(xc)
                    if fc < best:
                        best = fc
                        best_x = xc.copy()
                    if fc < min(fr, f_simplex[n]):
                        simplex[n] = xc
                        f_simplex[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                                return best
                            f_simplex[i] = func(simplex[i])
                            if f_simplex[i] < best:
                                best = f_simplex[i]
                                best_x = simplex[i].copy()
                
                # Convergence check
                if np.std(f_simplex) < 1e-15:
                    break
    
    return best
