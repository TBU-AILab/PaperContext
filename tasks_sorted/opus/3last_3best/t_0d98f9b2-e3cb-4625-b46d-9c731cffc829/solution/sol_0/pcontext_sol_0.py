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
    pop_size = min(max(20, 10 * dim), 200)
    
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
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.90:
            break
        
        generation += 1
        
        # Adaptive parameters
        F_base = 0.5 + 0.3 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Sort by fitness for current-to-best strategies
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.88:
                break
            
            # DE/current-to-pbest/1/bin
            p_best_size = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p_best_size)]
            
            # Select 2 distinct random indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r = np.random.choice(candidates, 2, replace=False)
            r1, r2 = r[0], r[1]
            
            F_i = F_base + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.0)
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * (population[r1] - population[r2])
            
            # Binomial crossover
            CR_i = CR + 0.1 * np.random.randn()
            CR_i = np.clip(CR_i, 0.1, 1.0)
            
            cross_points = np.random.random(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if best >= prev_best - 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnation
        if stagnation > max(15, 30 // max(1, dim // 10)):
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for i in range(half, pop_size):
                idx = sorted_idx[i]
                # Re-initialize around best with some exploration
                if np.random.random() < 0.5:
                    population[idx] = best_x + 0.1 * (upper - lower) * np.random.randn(dim)
                else:
                    population[idx] = lower + np.random.random(dim) * (upper - lower)
                population[idx] = np.clip(population[idx], lower, upper)
                if (datetime.now() - start).total_seconds() >= max_time * 0.85:
                    break
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search around best ---
    if best_x is not None:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time * 0.95 - elapsed
        
        if remaining > 0.5 and dim <= 100:
            # Simple Nelder-Mead
            n = dim
            # Initialize simplex around best_x
            scale = 0.02 * (upper - lower)
            simplex = np.zeros((n + 1, n))
            simplex[0] = best_x.copy()
            simplex_f = np.zeros(n + 1)
            simplex_f[0] = best
            
            for i in range(n):
                if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                    return best
                simplex[i + 1] = best_x.copy()
                simplex[i + 1][i] += scale[i] if scale[i] > 1e-15 else 1e-10
                simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
                simplex_f[i + 1] = func(simplex[i + 1])
                if simplex_f[i + 1] < best:
                    best = simplex_f[i + 1]
                    best_x = simplex[i + 1].copy()
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            
            for _ in range(5000):
                if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                    break
                
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                if simplex_f[0] < best:
                    best = simplex_f[0]
                    best_x = simplex[0].copy()
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
                fr = func(xr)
                if fr < best:
                    best = fr
                    best_x = xr.copy()
                
                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expand
                    xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                    fe = func(xe)
                    if fe < best:
                        best = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contract
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = func(xc)
                    if fc < best:
                        best = fc
                        best_x = xc.copy()
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            simplex_f[i] = func(simplex[i])
                            if simplex_f[i] < best:
                                best = simplex_f[i]
                                best_x = simplex[i].copy()
    
    return best
