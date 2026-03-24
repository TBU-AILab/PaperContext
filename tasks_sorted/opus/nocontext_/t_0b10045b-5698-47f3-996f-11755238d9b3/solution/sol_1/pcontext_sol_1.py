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
    pop_size = max(40, 10 * dim)
    F = 0.8       # Differential weight
    CR = 0.9      # Crossover probability
    
    # Initialize population using Latin Hypercube-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_x = population[i].copy()
    
    generation = 0
    
    # Adaptive DE with restarts and local search
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        generation += 1
        
        # Sort population by fitness for current-to-pbest strategy
        sorted_indices = np.argsort(fitness)
        
        # Adaptive F and CR
        new_population = np.empty_like(population)
        new_fitness = np.full(pop_size, float('inf'))
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Adaptive parameters with jitter
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Strategy selection: current-to-pbest/1/bin
            p_best_size = max(2, int(0.2 * pop_size))
            p_best_idx = sorted_indices[np.random.randint(0, p_best_size)]
            
            # Select r1, r2 different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            candidates.remove(r1)
            r2 = np.random.choice(candidates)
            
            # Mutation: current-to-pbest
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounds handling - bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.rand() * (population[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.rand() * (upper[j] - population[i][j])
                # Final clip
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
        
        population = new_population
        fitness = new_fitness
        
        # Periodically do local search around best solution (Nelder-Mead style simplex moves)
        if generation % 5 == 0 and best_x is not None:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Simple local search: coordinate-wise perturbation
            local_x = best_x.copy()
            local_f = best
            step = 0.01 * (upper - lower)
            
            for _ in range(3):  # few rounds of local refinement
                for j in range(dim):
                    passed_time = (datetime.now() - start)
                    if passed_time >= timedelta(seconds=max_time * 0.95):
                        return best
                    
                    # Try positive step
                    trial = local_x.copy()
                    trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
                    tf = func(trial)
                    if tf < local_f:
                        local_x = trial
                        local_f = tf
                        if tf < best:
                            best = tf
                            best_x = trial.copy()
                        continue
                    
                    # Try negative step
                    trial = local_x.copy()
                    trial[j] = np.clip(trial[j] - step[j], lower[j], upper[j])
                    tf = func(trial)
                    if tf < local_f:
                        local_x = trial
                        local_f = tf
                        if tf < best:
                            best = tf
                            best_x = trial.copy()
                
                step *= 0.5
            
            # Inject best back into population
            worst_idx = np.argmax(fitness)
            population[worst_idx] = best_x.copy()
            fitness[worst_idx] = best
        
        # Check for stagnation and restart partially
        if generation % 50 == 0:
            std_fitness = np.std(fitness)
            if std_fitness < 1e-12:
                # Restart half the population
                n_restart = pop_size // 2
                restart_indices = sorted_indices[pop_size // 2:]
                for idx in restart_indices:
                    passed_time = (datetime.now() - start)
                    if passed_time >= timedelta(seconds=max_time * 0.95):
                        return best
                    population[idx] = lower + np.random.rand(dim) * (upper - lower)
                    fitness[idx] = func(population[idx])
                    if fitness[idx] < best:
                        best = fitness[idx]
                        best_x = population[idx].copy()
    
    return best
