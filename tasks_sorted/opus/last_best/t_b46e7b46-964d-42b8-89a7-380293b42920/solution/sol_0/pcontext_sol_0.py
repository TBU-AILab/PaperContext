#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
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
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        improved = False
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.85):
                break
            
            # Adaptive F and CR
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Strategy selection: current-to-best/1 with some probability, rand/1 otherwise
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if np.random.random() < 0.5:
                # current-to-best/1
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = population[i] + Fi * (best_params - population[i]) + Fi * (population[a] - population[b])
            else:
                # rand/1/bin
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + Fi * (population[b] - population[c])
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
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
        
        # If stagnating, reinitialize worst half
        if stagnation > 15:
            stagnation = 0
            half = pop_size // 2
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            for i in range(half, pop_size):
                # Reinitialize around best with some randomness
                if np.random.random() < 0.5:
                    population[i] = best_params + 0.3 * (upper - lower) * (np.random.rand(dim) - 0.5)
                else:
                    population[i] = lower + np.random.rand(dim) * (upper - lower)
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
    
    # --- Phase 3: Nelder-Mead local search around best ---
    def nelder_mead_step():
        nonlocal best, best_params
        
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around best_params
        scale = 0.05 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = best
        
        for i in range(1, n + 1):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return
            simplex[i] = best_params.copy()
            simplex[i][i - 1] += scale[i - 1]
            simplex[i] = np.clip(simplex[i], lower, upper)
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        for iteration in range(500):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return
            
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
            
            if fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    if fr < best:
                        best = fr
                        best_params = xr.copy()
                    return
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                else:
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    return
                fc = func(xc)
                
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        passed_time = (datetime.now() - start)
                        if passed_time >= timedelta(seconds=max_time * 0.98):
                            return
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = func(simplex[i])
            
            if f_simplex.min() < best:
                idx = np.argmin(f_simplex)
                best = f_simplex[idx]
                best_params = simplex[idx].copy()
            
            # Check convergence
            if np.std(f_simplex) < 1e-12:
                break
    
    nelder_mead_step()
    
    return best
