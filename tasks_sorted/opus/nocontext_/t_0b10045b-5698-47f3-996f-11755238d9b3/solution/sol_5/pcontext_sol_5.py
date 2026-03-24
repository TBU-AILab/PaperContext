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
    
    # ---- Parameters ----
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8       # DE mutation factor
    CR = 0.9      # DE crossover rate
    
    # ---- Initialize population ----
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.98:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # ---- Main loop: Differential Evolution with restarts and local search ----
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        
        generation += 1
        improved_any = False
        
        # Adaptive F and CR
        F_cur = 0.5 + 0.3 * np.random.rand()
        CR_cur = 0.1 + 0.8 * np.random.rand()
        
        # Sort population by fitness for current-to-best strategy
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95:
                return best
            
            # DE/current-to-best/1/bin with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            F_i = F_cur * (0.8 + 0.4 * np.random.rand())
            mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[a] - population[b])
            
            # Crossover
            cross_points = np.random.rand(dim) < CR_cur
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back into bounds
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved_any = True
        
        # Periodically do local search around best (Nelder-Mead-like simplex or coordinate descent)
        if generation % 5 == 0 and best_params is not None:
            elapsed = (datetime.now() - start).total_seconds()
            remaining = max_time * 0.95 - elapsed
            if remaining > 0.5:
                # Simple local search: coordinate-wise golden section / perturbation
                local_best = best_params.copy()
                local_fit = best
                scale = 0.01 * (upper - lower)
                
                for _ in range(3):  # few rounds
                    for d in range(dim):
                        elapsed = (datetime.now() - start).total_seconds()
                        if elapsed >= max_time * 0.93:
                            break
                        
                        for sign in [1, -1]:
                            trial = local_best.copy()
                            trial[d] = np.clip(trial[d] + sign * scale[d], lower[d], upper[d])
                            tf = func(trial)
                            if tf < local_fit:
                                local_fit = tf
                                local_best = trial.copy()
                    
                    scale *= 0.5
                
                if local_fit < best:
                    best = local_fit
                    best_params = local_best.copy()
                    # Update worst member with this
                    worst_idx = np.argmax(fitness)
                    population[worst_idx] = local_best.copy()
                    fitness[worst_idx] = local_fit
        
        # Check stagnation for restart
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart half the population if stagnating
        if stagnation > 15:
            stagnation = 0
            n_restart = pop_size // 2
            worst_indices = np.argsort(fitness)[-n_restart:]
            for idx in worst_indices:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    return best
                population[idx] = np.random.uniform(lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
    
    return best
