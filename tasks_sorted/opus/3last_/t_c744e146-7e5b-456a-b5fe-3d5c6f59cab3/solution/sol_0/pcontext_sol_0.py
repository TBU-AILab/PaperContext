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
    
    # ---- Main DE loop with restarts and local search ----
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        if time_left() < 0.05:
            return best
        
        generation += 1
        
        # Adaptive F and CR
        F_cur = F + 0.1 * np.random.randn()
        F_cur = np.clip(F_cur, 0.1, 1.5)
        CR_cur = CR + 0.1 * np.random.randn()
        CR_cur = np.clip(CR_cur, 0.1, 1.0)
        
        # DE/current-to-best/1/bin
        indices = np.arange(pop_size)
        
        for i in range(pop_size):
            if time_left() < 0.05:
                return best
            
            # Pick 2 random individuals different from i
            candidates = indices[indices != i]
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # Mutation: current-to-best
            mutant = population[i] + F_cur * (best_x - population[i]) + F_cur * (population[r1] - population[r2])
            
            # Crossover
            cross_points = np.random.rand(dim) < CR_cur
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounds handling: bounce back
            for d in range(dim):
                if trial[d] < lower[d] or trial[d] > upper[d]:
                    trial[d] = np.random.uniform(lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Local search (Nelder-Mead style simplex) on best solution periodically
        if generation % 15 == 0 and time_left() > 1.0:
            # Simple coordinate-wise local search
            step = (upper - lower) * 0.01
            improved = True
            ls_iters = 0
            x_local = best_x.copy()
            f_local = best
            while improved and ls_iters < 3 and time_left() > 0.5:
                improved = False
                ls_iters += 1
                for d in range(dim):
                    if time_left() < 0.1:
                        break
                    for sign in [1, -1]:
                        x_try = x_local.copy()
                        x_try[d] = np.clip(x_try[d] + sign * step[d], lower[d], upper[d])
                        f_try = func(x_try)
                        if f_try < f_local:
                            f_local = f_try
                            x_local = x_try
                            improved = True
            if f_local < best:
                best = f_local
                best_x = x_local.copy()
                # Inject back into population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_x.copy()
                fitness[worst_idx] = best
        
        # Restart worst half if stagnated
        if stagnation > 20:
            stagnation = 0
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for j in sorted_idx[half:]:
                population[j] = np.random.uniform(lower, upper, dim)
                # Bias some towards best
                if np.random.rand() < 0.3:
                    population[j] = best_x + 0.1 * (upper - lower) * np.random.randn(dim)
                    population[j] = np.clip(population[j], lower, upper)
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_x = population[j].copy()
    
    return best
