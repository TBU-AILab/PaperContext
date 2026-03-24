

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    range_width = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
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

    # Parameters
    pop_size = min(max(30, 8 * dim), 150)
    
    # Initialize population with Latin Hypercube Sampling
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, j] = lower[j] + (perm[i] + np.random.random()) / pop_size * range_width[j]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time:
            return best
        fitness[i] = evaluate(population[i])
    
    # SHADE-like adaptive DE
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size
    
    generation = 0
    
    while elapsed() < max_time * 0.92:
        generation += 1
        
        S_F = []
        S_CR = []
        delta_f = []
        
        # Sort population by fitness for current-to-pbest
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.92:
                break
            
            # Pick from memory
            ri = np.random.randint(memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            # Cauchy for F, Normal for CR
            Fi = mu_F + 0.1 * np.random.standard_cauchy()
            while Fi <= 0:
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            indices = list(range(pop_size))
            indices.remove(i)
            r1 = np.random.choice(indices)
            
            # r2 from population + archive
            combined_size = pop_size + len(archive)
            r2_idx = np.random.randint(combined_size)
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(combined_size)
            if r2_idx < pop_size:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Boundary: bounce-back
            out_low = trial < lower
            out_high = trial > upper
            trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
            trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
            trial = clip(trial)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_f))
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    else:
                        archive[np.random.randint(archive_max)] = population[i].copy()
                population[i] = trial
                fitness[i] = trial_f
        
        # Update memory
        if S_F:
            weights = np.array(delta_f)
            weights = weights / weights.sum()
            new_F = np.sum(weights * np.array(S_F)**2) / max(np.sum(weights * np.array(S_F)), 1e-30)
            new_CR = np.sum(weights * np.array(S_CR))
            M_F[mem_idx] = new_F
            M_CR[mem_idx] = new_CR
            mem_idx = (mem_idx + 1) % memory_size
    
    # Local search phase with remaining time
    if best_params is not None:
        scale = 0.05
        while elapsed() < max_time * 0.99:
            candidate = best_params + scale * range_width * np.random.randn(dim)
            cf = evaluate(candidate)
            if cf < best:
                scale = min(scale * 1.2, 0.1)
            else:
                scale *= 0.85
                if scale < 1e-8:
                    scale = 0.02
    
    return best