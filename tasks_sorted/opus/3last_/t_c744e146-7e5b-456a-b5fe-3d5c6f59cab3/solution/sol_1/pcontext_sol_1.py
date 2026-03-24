

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    # ---- Parameters ----
    pop_size = min(max(30, 10 * dim), 300)
    
    # ---- Initialize population with Latin Hypercube Sampling ----
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = lower[d] + (perm[i] + np.random.rand()) / n * ranges[d]
        return samples
    
    population = lhs_sample(pop_size)
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best:
        best = fitness[best_idx]
        best_x = population[best_idx].copy()
    
    # ---- SHADE-like adaptive DE ----
    memory_size = 10
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    # Track function evaluations for budget management
    p_min = max(2, int(0.1 * pop_size))
    
    while True:
        if time_left() < 0.1:
            return best
        
        generation += 1
        
        S_F = []
        S_CR = []
        delta_f = []
        
        indices = np.arange(pop_size)
        
        # Generate F and CR for each individual
        ri = np.random.randint(0, memory_size, pop_size)
        F_vals = np.zeros(pop_size)
        CR_vals = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Cauchy for F
            while True:
                Fi = M_F[ri[i]] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            F_vals[i] = min(Fi, 1.0)
            # Normal for CR
            CRi = np.clip(M_CR[ri[i]] + 0.1 * np.random.randn(), 0, 1)
            CR_vals[i] = CRi
        
        for i in range(pop_size):
            if time_left() < 0.05:
                return best
            
            Fi = F_vals[i]
            CRi = CR_vals[i]
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            candidates = np.delete(indices, i)
            r1 = np.random.choice(candidates)
            
            # r2 from population + archive
            combined_pop = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined_pop = [x for x in combined_pop if x != i and x != r1]
            r2_idx = np.random.choice(combined_pop)
            
            if r2_idx < pop_size:
                xr2 = population[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CRi
            jrand = np.random.randint(0, dim)
            cross_points[jrand] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back bounds
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                if trial_fitness < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_fitness))
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        # Update memory
        if S_F:
            weights = np.array(delta_f)
            weights /= weights.sum() + 1e-30
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[mem_idx] = mean_F
            M_CR[mem_idx] = mean_CR
            mem_idx = (mem_idx + 1) % memory_size
        
        # Local search periodically
        if generation % 10 == 0 and time_left() > 1.0:
            x_local = best_x.copy()
            f_local = best
            step = ranges * 0.005
            for _ in range(5):
                if time_left() < 0.5:
                    break
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
                step *= 0.5
            if f_local < best:
                best = f_local
                best_x = x_local.copy()
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_x.copy()
                fitness[worst_idx] = best
        
        # Check stagnation
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 30:
            stagnation = 0
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for j in sorted_idx[half:]:
                if np.random.rand() < 0.5:
                    population[j] = best_x + 0.05 * ranges * np.random.randn(dim)
                    population[j] = clip(population[j])
                else:
                    population[j] = np.random.uniform(lower, upper)
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_x = population[j].copy()
    
    return best