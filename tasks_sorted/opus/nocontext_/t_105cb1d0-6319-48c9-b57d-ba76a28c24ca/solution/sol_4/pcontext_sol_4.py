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
    
    # Parameters for CMA-ES-like strategy combined with differential evolution
    # We'll implement a hybrid approach:
    # 1. Start with Latin Hypercube Sampling
    # 2. Use a simplified CMA-ES / Nelder-Mead hybrid
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Phase 1: Latin Hypercube Sampling for initialization
    pop_size = min(max(20, 4 * dim), 100)
    
    # Generate initial population via LHS
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = lower[i] + (perm[j] + np.random.random()) / pop_size * (upper[i] - lower[i])
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size)])
    
    if elapsed() >= max_time * 0.95:
        return best
    
    # Sort population by fitness
    order = np.argsort(fitness)
    population = population[order]
    fitness = fitness[order]
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            # Select strategy adaptively
            strategy = np.random.random()
            
            if strategy < 0.4:
                # DE/best/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = population[0] + F * (population[idxs[0]] - population[idxs[1]])
            elif strategy < 0.7:
                # DE/rand/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                mutant = population[idxs[0]] + F * (population[idxs[1]] - population[idxs[2]])
            else:
                # DE/current-to-best/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = population[i] + F * (population[0] - population[i]) + F * (population[idxs[0]] - population[idxs[1]])
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial = clip(trial)
            f_trial = evaluate(trial)
            
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Sort
        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]
        
        # Adaptive F and CR
        generation += 1
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 5:
            F = np.random.uniform(0.4, 1.0)
            CR = np.random.uniform(0.5, 1.0)
            # Inject some random individuals to replace worst
            n_replace = pop_size // 4
            for k in range(pop_size - n_replace, pop_size):
                population[k] = np.array([np.random.uniform(l, u) for l, u in bounds])
                fitness[k] = evaluate(population[k])
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]
            stagnation = 0
    
    # Phase 3: Local search using Nelder-Mead simplex on the best solution
    def nelder_mead(x0, initial_step=None, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        nonlocal best, best_params
        
        n = len(x0)
        if initial_step is None:
            initial_step = (upper - lower) * 0.05
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            point = x0.copy()
            point[i] += initial_step[i]
            simplex[i + 1] = clip(point)
        
        f_values = np.array([evaluate(simplex[i]) for i in range(n + 1)])
        
        max_iter = 5000
        for iteration in range(max_iter):
            if elapsed() >= max_time * 0.95:
                break
            
            # Sort
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if f_values[0] <= fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_values[i] = evaluate(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc)
                    if fc < f_values[-1]:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_values[i] = evaluate(simplex[i])
            
            # Check convergence
            if np.max(np.abs(f_values - f_values[0])) < 1e-15:
                break
    
    # Run Nelder-Mead from top candidates
    n_starts = min(5, pop_size)
    for s in range(n_starts):
        if elapsed() >= max_time * 0.95:
            break
        step = (upper - lower) * 0.01 * (1 + s)
        nelder_mead(population[s], initial_step=step)
    
    # Phase 4: Fine-grained local search around best with random perturbations
    if best_params is not None:
        scale = (upper - lower) * 0.001
        while elapsed() < max_time * 0.98:
            perturbation = np.random.randn(dim) * scale
            candidate = clip(best_params + perturbation)
            evaluate(candidate)
            # Decrease scale gradually
            scale *= 0.999
    
    return best
