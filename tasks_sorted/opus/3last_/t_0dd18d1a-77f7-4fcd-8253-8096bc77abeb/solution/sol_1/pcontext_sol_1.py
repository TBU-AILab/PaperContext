

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    best_params = None
    
    # --- Phase 1: Latin Hypercube Sampling initialization ---
    pop_size = min(max(30, 15 * dim), 300)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = (perm[j] + np.random.uniform()) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with SHADE-like adaptation ---
    F_mem = [0.5] * 5
    CR_mem = [0.5] * 5
    archive = []
    mem_idx = 0
    
    while elapsed() < max_time * 0.60:
        best_idx = np.argmin(fitness)
        
        S_F = []
        S_CR = []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.60:
                break
            
            # Sample F and CR from memory
            k = np.random.randint(0, len(F_mem))
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + F_mem[k], 0.1, 1.0)
            CRi = np.clip(np.random.normal(CR_mem[k], 0.1), 0.0, 1.0)
            
            # Select indices
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            # p-best: top 10-20%
            p = max(2, int(0.15 * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            r1 = np.random.choice(idxs)
            # r2 from population + archive
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial = clip(trial)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                if trial_fitness < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(Fi)
                    S_CR.append(CRi)
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        if S_F:
            F_mem[mem_idx % len(F_mem)] = np.mean(np.array(S_F)**2) / np.mean(S_F) if np.mean(S_F) > 0 else 0.5
            CR_mem[mem_idx % len(CR_mem)] = np.mean(S_CR)
            mem_idx += 1
    
    # --- Phase 3: Local search with restarts ---
    sigma = 0.05 * ranges
    current = best_params.copy()
    current_fit = best
    
    no_improve = 0
    while elapsed() < max_time * 0.95:
        perturbation = np.random.randn(dim) * sigma
        candidate = clip(current + perturbation)
        cand_fit = func(candidate)
        if cand_fit < current_fit:
            current = candidate
            current_fit = cand_fit
            if cand_fit < best:
                best = cand_fit
                best_params = candidate.copy()
            no_improve = 0
            sigma *= 1.05
        else:
            no_improve += 1
            sigma *= 0.97
        
        if no_improve > 80 * dim:
            sigma = 0.05 * ranges
            current = best_params + np.random.randn(dim) * 0.1 * ranges
            current = clip(current)
            current_fit = func(current)
            if current_fit < best:
                best = current_fit
                best_params = current.copy()
            no_improve = 0
    
    return best