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
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness_vals = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness_vals[i] = f
        if f < best:
            best = f
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        improved = False
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.85):
                break
            
            # Adaptive F and CR
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # DE/current-to-best/1 strategy
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness_vals)
            
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[a] - population[b])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounce-back boundary handling
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.rand() * (population[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.rand() * (upper[j] - population[i][j])
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            f_trial = func(trial)
            
            if f_trial <= fitness_vals[i]:
                population[i] = trial
                fitness_vals[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                    improved = True
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # If stagnant, inject some random individuals
        if stagnation > 10:
            n_replace = pop_size // 4
            worst_idxs = np.argsort(fitness_vals)[-n_replace:]
            for idx in worst_idxs:
                population[idx] = lower + np.random.rand(dim) * (upper - lower)
                fitness_vals[idx] = func(population[idx])
                if fitness_vals[idx] < best:
                    best = fitness_vals[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search around best solution ---
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
            point = best_params.copy()
            point[i] += scale[i]
            point = np.clip(point, lower, upper)
            simplex[i + 1] = point
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return best
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        while True:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
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
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                if fr < best:
                    best = fr
                break
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                continue
            
            if fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
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
            if fc < f_simplex[-1]:
                simplex[-1] = xc
                f_simplex[-1] = fc
                continue
            
            # Shrink
            for i in range(1, n + 1):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    return best
                simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lower, upper)
                f_simplex[i] = func(simplex[i])
                if f_simplex[i] < best:
                    best = f_simplex[i]
                    best_params = simplex[i].copy()
    
    return best
