

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling initialization ---
    pop_size = min(max(15 * dim, 40), 150)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: SHADE (Success-History based Adaptive DE) ---
    H = 50  # history size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.80:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            
            # Select from history
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            # Generate CR from Normal
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # DE/current-to-pbest/1 with archive
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.randint(0, p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            if i in combined:
                combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            if len(combined) == 0:
                combined = [r1]
            r2_idx = np.random.choice(combined)
            
            if r2_idx < pop_size:
                xr2 = population[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Boundary: midpoint reflection
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = eval_f(trial)
            
            if trial_fitness < new_fitness[i]:
                archive.append(population[i].copy())
                if len(archive) > archive_max:
                    archive.pop(np.random.randint(len(archive)))
                delta = fitness[i] - trial_fitness
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(delta)
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if len(S_F) > 0:
            weights = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % H
        
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 25:
            half = pop_size // 2
            for i in range(half, pop_size):
                if np.random.random() < 0.5:
                    population[i] = best_params + 0.05 * ranges * np.random.randn(dim)
                else:
                    population[i] = lower + np.random.rand(dim) * ranges
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = eval_f(population[i])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search ---
    n = dim
    # Initialize simplex around best
    simplex = np.zeros((n + 1, dim))
    simplex[0] = best_params.copy()
    f_simplex = np.zeros(n + 1)
    f_simplex[0] = best
    
    scale = 0.05 * ranges
    for i in range(n):
        if elapsed() >= max_time * 0.98:
            return best
        simplex[i + 1] = best_params.copy()
        simplex[i + 1][i] += scale[i] if scale[i] != 0 else 0.01
        simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        f_simplex[i + 1] = eval_f(simplex[i + 1])
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    while elapsed() < max_time * 0.97:
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, lower, upper)
        fr = eval_f(xr)
        
        if elapsed() >= max_time * 0.97:
            return best
        
        if fr < f_simplex[-2] and fr >= f_simplex[0]:
            simplex[-1] = xr
            f_simplex[-1] = fr
        elif fr < f_simplex[0]:
            # Expand
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, lower, upper)
            fe = eval_f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_simplex[-1] = fe
            else:
                simplex[-1] = xr
                f_simplex[-1] = fr
        else:
            # Contract
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, lower, upper)
            fc = eval_f(xc)
            if fc < f_simplex[-1]:
                simplex[-1] = xc
                f_simplex[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    if elapsed() >= max_time * 0.97:
                        return best
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    f_simplex[i] = eval_f(simplex[i])
    
    # --- Phase 4: Final fine local search ---
    while elapsed() < max_time * 0.995:
        scale = 0.001 * ranges
        candidate = best_params + scale * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best