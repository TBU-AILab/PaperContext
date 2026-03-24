#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Parameters ---
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8   # differential weight
    CR = 0.9  # crossover probability
    
    # --- Initialize population ---
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best = fitness[best_idx]
    best_params = population[best_idx].copy()
    
    # --- CMA-ES-like local search helper ---
    def local_search_nelder_mead(start_point, start_fitness, time_limit_sec):
        nonlocal best, best_params
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = start_point.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = start_fitness
        
        scale = (upper - lower) * 0.05
        for i in range(n):
            point = start_point.copy()
            point[i] += scale[i] if start_point[i] + scale[i] <= upper[i] else -scale[i]
            point = np.clip(point, lower, upper)
            simplex[i + 1] = point
            f_simplex[i + 1] = func(point)
            if f_simplex[i + 1] < best:
                best = f_simplex[i + 1]
                best_params = point.copy()
        
        local_start = datetime.now()
        
        while True:
            if (datetime.now() - local_start).total_seconds() > time_limit_sec:
                break
            if (datetime.now() - start).total_seconds() >= max_time * 0.99:
                break
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best:
                best = f_simplex[0]
                best_params = simplex[0].copy()
            
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
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
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
                simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), lower, upper)
                f_simplex[i] = func(simplex[i])
                if f_simplex[i] < best:
                    best = f_simplex[i]
                    best_params = simplex[i].copy()
    
    # --- Main DE loop ---
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.75:
            break
        
        # Adaptive parameters
        gen_F = F + 0.1 * np.random.randn()
        gen_F = np.clip(gen_F, 0.4, 1.2)
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.75:
                break
            
            # DE/current-to-best/1/bin
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # Mutation: current-to-best with jitter
            jF = gen_F * (0.8 + 0.4 * np.random.rand())
            mutant = population[i] + jF * (best_params - population[i]) + jF * (population[r1] - population[r2])
            mutant = np.clip(mutant, lower, upper)
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        generation += 1
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst individuals if stagnating
        if stagnation > 10:
            n_restart = pop_size // 3
            worst_indices = np.argsort(fitness)[-n_restart:]
            for idx in worst_indices:
                # Random restart near best or random
                if np.random.rand() < 0.5:
                    scale = (upper - lower) * 0.1 * np.random.rand()
                    population[idx] = np.clip(best_params + scale * np.random.randn(dim), lower, upper)
                else:
                    population[idx] = np.random.uniform(lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # --- Local refinement with Nelder-Mead ---
    remaining = max_time - (datetime.now() - start).total_seconds()
    if remaining > 0.5:
        local_search_nelder_mead(best_params, best, remaining * 0.9)
    
    return best
