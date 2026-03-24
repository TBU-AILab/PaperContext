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
    
    # ---- Parameters ----
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8       # DE mutation factor
    CR = 0.9      # DE crossover rate
    
    # ---- Initialize population ----
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = population[best_idx].copy()
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    # ---- Main loop: Hybrid DE + CMA-ES-like local search ----
    generation = 0
    
    while True:
        if time_left() < 0.05:
            return best
        
        generation += 1
        
        # --- Differential Evolution step ---
        indices = np.arange(pop_size)
        for i in range(pop_size):
            if time_left() < 0.05:
                return best
            
            # DE/best/1/bin with occasional DE/rand/1
            idxs = np.delete(indices, i)
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # Adaptive: use best vector with some probability
            if np.random.rand() < 0.8:
                base = best_x
            else:
                base = a
            
            # Adaptive F
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = base + Fi * (b - c)
            mutant = np.clip(mutant, lower, upper)
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        # --- Periodic Nelder-Mead simplex local search around best ---
        if generation % 5 == 0 and time_left() > 0.5:
            # Simple Nelder-Mead
            simplex_best = _nelder_mead(func, best_x.copy(), lower, upper, dim, 
                                         start, max_time, max_iter=200 * dim)
            sb_fitness = func(simplex_best)
            if sb_fitness < best:
                best = sb_fitness
                best_x = simplex_best.copy()
                # Inject back into population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_x.copy()
                fitness[worst_idx] = best
        
        # --- Restart worst members occasionally ---
        if generation % 15 == 0:
            sorted_idx = np.argsort(fitness)
            n_restart = pop_size // 4
            for j in sorted_idx[-n_restart:]:
                # Restart around best with some noise
                scale = (upper - lower) * 0.1 * np.random.rand()
                population[j] = best_x + scale * np.random.randn(dim)
                population[j] = np.clip(population[j], lower, upper)
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_x = population[j].copy()
                if time_left() < 0.05:
                    return best
    
    return best


def _nelder_mead(func, x0, lower, upper, dim, start, max_time, max_iter=1000):
    """Simplified Nelder-Mead simplex method."""
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    n = dim
    # Build initial simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    scale = (upper - lower) * 0.05
    for i in range(n):
        point = x0.copy()
        point[i] += scale[i] if scale[i] > 1e-10 else 0.05
        point = np.clip(point, lower, upper)
        simplex[i + 1] = point
    
    f_values = np.array([func(s) for s in simplex])
    
    for iteration in range(max_iter):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time - 0.1:
            break
        
        # Order
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        
        # Centroid (excluding worst)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, lower, upper)
        fr = func(xr)
        
        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        elif fr < f_values[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, lower, upper)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, lower, upper)
            fc = func(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    f_values[i] = func(simplex[i])
        
        # Convergence check
        if np.std(f_values) < 1e-12:
            break
    
    best_idx = np.argmin(f_values)
    return simplex[best_idx]
