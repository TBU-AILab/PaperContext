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
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[perm[j], i] = (perm[j] + np.random.uniform()) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        generation += 1
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # Adaptive F
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[a] - population[b])
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnant
        if stagnation > 10:
            sorted_idx = np.argsort(fitness)
            for idx in sorted_idx[pop_size // 2:]:
                population[idx] = lower + np.random.rand(dim) * (upper - lower)
                if elapsed() >= max_time * 0.70:
                    break
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search around best ---
    def nelder_mead(x0, initial_step=None, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        nonlocal best, best_params
        
        n = len(x0)
        if initial_step is None:
            initial_step = (upper - lower) * 0.05
        
        # Build initial simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += initial_step[i]
            p = np.clip(p, lower, upper)
            simplex[i + 1] = p
        
        f_values = np.zeros(n + 1)
        for i in range(n + 1):
            if elapsed() >= max_time * 0.95:
                return
            f_values[i] = func(simplex[i])
            if f_values[i] < best:
                best = f_values[i]
                best_params = simplex[i].copy()
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            if elapsed() >= max_time * 0.95:
                return
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_values[0] <= fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
                continue
            
            if fr < f_values[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                if elapsed() >= max_time * 0.95:
                    return
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
                continue
            
            # Contraction
            if fr < f_values[-1]:
                xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                if elapsed() >= max_time * 0.95:
                    return
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc <= fr:
                    simplex[-1] = xc
                    f_values[-1] = fc
                    continue
            else:
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                if elapsed() >= max_time * 0.95:
                    return
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < f_values[-1]:
                    simplex[-1] = xc
                    f_values[-1] = fc
                    continue
            
            # Shrink
            for i in range(1, n + 1):
                simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lower, upper)
                if elapsed() >= max_time * 0.95:
                    return
                f_values[i] = func(simplex[i])
                if f_values[i] < best:
                    best = f_values[i]
                    best_params = simplex[i].copy()
    
    # Run Nelder-Mead from best point
    nelder_mead(best_params)
    
    # If time remains, run NM with smaller step from current best
    if elapsed() < max_time * 0.90:
        nelder_mead(best_params, initial_step=(upper - lower) * 0.01)
    
    # Use remaining time for random restarts of NM
    while elapsed() < max_time * 0.95:
        # Pick from top solutions
        sorted_idx = np.argsort(fitness)
        restart_idx = np.random.choice(sorted_idx[:max(3, pop_size // 5)])
        x0 = population[restart_idx] + np.random.randn(dim) * (upper - lower) * 0.02
        x0 = np.clip(x0, lower, upper)
        nelder_mead(x0, initial_step=(upper - lower) * 0.02)
    
    return best
