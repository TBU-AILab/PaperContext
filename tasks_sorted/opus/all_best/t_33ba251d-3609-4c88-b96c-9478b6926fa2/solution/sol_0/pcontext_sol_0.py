#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Phase 1: Latin Hypercube Sampling for initial population
    pop_size = min(max(20, 10 * dim), 200)
    
    def random_solution():
        return lower + np.random.rand(dim) * (upper - lower)
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    # Evaluate initial population
    population = []
    fitness_vals = []
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        x = random_solution()
        f = func(x)
        population.append(x.copy())
        fitness_vals.append(f)
        if f < best:
            best = f
            best_x = x.copy()
    
    population = np.array(population)
    fitness_vals = np.array(fitness_vals)
    
    # Sort population
    idx = np.argsort(fitness_vals)
    population = population[idx]
    fitness_vals = fitness_vals[idx]
    best_x = population[0].copy()
    best = fitness_vals[0]
    
    # Phase 2: CMA-ES inspired + Differential Evolution hybrid
    # Use a restart strategy
    
    def nelder_mead_step(simplex, simplex_f, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """One step of Nelder-Mead"""
        n = simplex.shape[1]
        idx = np.argsort(simplex_f)
        simplex = simplex[idx]
        simplex_f = simplex_f[idx]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = clip(centroid + alpha * (centroid - simplex[-1]))
        fr = func(xr)
        
        if simplex_f[0] <= fr < simplex_f[-2]:
            simplex[-1] = xr
            simplex_f[-1] = fr
        elif fr < simplex_f[0]:
            # Expansion
            xe = clip(centroid + gamma * (xr - centroid))
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                simplex_f[-1] = fe
            else:
                simplex[-1] = xr
                simplex_f[-1] = fr
        else:
            # Contraction
            xc = clip(centroid + rho * (simplex[-1] - centroid))
            fc = func(xc)
            if fc < simplex_f[-1]:
                simplex[-1] = xc
                simplex_f[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    simplex_f[i] = func(simplex[i])
        
        return simplex, simplex_f
    
    # DE parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.90:
        generation += 1
        
        # Differential Evolution step
        new_population = population.copy()
        new_fitness = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.90:
                break
            
            # Mutation: DE/best/1 with some DE/rand/1
            if np.random.rand() < 0.5:
                # DE/best/1
                idxs = np.random.choice(pop_size, 2, replace=False)
                while i in idxs:
                    idxs = np.random.choice(pop_size, 2, replace=False)
                mutant = best_x + F * (population[idxs[0]] - population[idxs[1]])
            else:
                # DE/rand/1
                idxs = np.random.choice(pop_size, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(pop_size, 3, replace=False)
                mutant = population[idxs[0]] + F * (population[idxs[1]] - population[idxs[2]])
            
            mutant = clip(mutant)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial = clip(trial)
            f_trial = func(trial)
            
            if f_trial <= fitness_vals[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        population = new_population
        fitness_vals = new_fitness
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnated
        if stagnation > 10 + dim:
            idx_sort = np.argsort(fitness_vals)
            population = population[idx_sort]
            fitness_vals = fitness_vals[idx_sort]
            
            half = pop_size // 2
            for i in range(half, pop_size):
                if elapsed() >= max_time * 0.90:
                    break
                # Random around best with varying scale
                scale = np.random.rand() * (upper - lower)
                population[i] = clip(best_x + scale * np.random.randn(dim))
                fitness_vals[i] = func(population[i])
                if fitness_vals[i] < best:
                    best = fitness_vals[i]
                    best_x = population[i].copy()
            stagnation = 0
            
            # Also adapt DE parameters
            F = 0.5 + 0.5 * np.random.rand()
            CR = 0.5 + 0.5 * np.random.rand()
    
    # Phase 3: Local search around best solution using Nelder-Mead
    if elapsed() < max_time * 0.98 and dim <= 50:
        simplex_size = dim + 1
        simplex = np.zeros((simplex_size, dim))
        simplex[0] = best_x.copy()
        simplex_f = np.zeros(simplex_size)
        simplex_f[0] = best
        
        scale = 0.01 * (upper - lower)
        for i in range(1, simplex_size):
            simplex[i] = clip(best_x + scale * np.random.randn(dim))
            simplex_f[i] = func(simplex[i])
            if simplex_f[i] < best:
                best = simplex_f[i]
                best_x = simplex[i].copy()
        
        while elapsed() < max_time * 0.98:
            simplex, simplex_f = nelder_mead_step(simplex, simplex_f)
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best:
                best = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()
    
    return best
