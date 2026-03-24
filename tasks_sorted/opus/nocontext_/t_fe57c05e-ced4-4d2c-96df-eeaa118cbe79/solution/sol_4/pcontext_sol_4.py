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
    
    # --- Parameters ---
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8   # differential weight
    CR = 0.9  # crossover probability
    
    # --- Initialize population ---
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = population[best_idx].copy()
    
    def time_remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    # --- CMA-ES-like local search around best ---
    def local_search_nelder_mead(x0, budget_seconds):
        """Simple Nelder-Mead simplex search."""
        n = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = func(x0)
        
        nonlocal best, best_x
        if f_simplex[0] < best:
            best = f_simplex[0]
            best_x = x0.copy()
        
        for i in range(n):
            point = x0.copy()
            step = max(abs(x0[i]) * 0.05, (upper[i] - lower[i]) * 0.01)
            point[i] += step
            point = np.clip(point, lower, upper)
            simplex[i + 1] = point
            f_simplex[i + 1] = func(point)
            if f_simplex[i + 1] < best:
                best = f_simplex[i + 1]
                best_x = point.copy()
        
        start_local = datetime.now()
        max_iter = 5000
        
        for iteration in range(max_iter):
            if (datetime.now() - start_local).total_seconds() > budget_seconds:
                break
            if time_remaining() < 0.1:
                break
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Centroid (exclude worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_x = xr.copy()
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            elif fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_x = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            else:
                # Contraction
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_x = xc.copy()
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lower, upper)
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_x = simplex[i].copy()
        
        best_local_idx = np.argmin(f_simplex)
        return simplex[best_local_idx], f_simplex[best_local_idx]
    
    # --- Main DE loop ---
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        if time_remaining() < 0.5:
            break
        
        generation += 1
        
        # Adaptive parameters
        if stagnation > 20:
            F_cur = np.random.uniform(0.4, 1.0)
            CR_cur = np.random.uniform(0.1, 1.0)
        else:
            F_cur = F
            CR_cur = CR
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if time_remaining() < 0.3:
                break
            
            # DE/current-to-best/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            F_i = F_cur + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            mutant = population[i] + F_i * (best_x - population[i]) + F_i * (population[a] - population[b])
            mutant = np.clip(mutant, lower, upper)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR_cur
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            trial = np.clip(trial, lower, upper)
            
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
        if abs(best - prev_best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst individuals if stagnating
        if stagnation > 50:
            n_restart = pop_size // 2
            worst_idxs = np.argsort(fitness)[-n_restart:]
            for idx in worst_idxs:
                population[idx] = np.random.uniform(lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
            stagnation = 0
        
        # Periodically run local search on best
        if generation % 30 == 0 and time_remaining() > 2.0:
            budget = min(time_remaining() * 0.3, 3.0)
            lx, lf = local_search_nelder_mead(best_x.copy(), budget)
            if lf < best:
                best = lf
                best_x = lx.copy()
            # Insert into population
            worst_idx = np.argmax(fitness)
            population[worst_idx] = best_x.copy()
            fitness[worst_idx] = best
    
    # Final local search with remaining time
    remaining = time_remaining()
    if remaining > 0.5 and best_x is not None:
        lx, lf = local_search_nelder_mead(best_x.copy(), remaining - 0.2)
        if lf < best:
            best = lf
            best_x = lx.copy()
    
    return best
