

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
    
    def remaining():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(30, 15 * dim), 300)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: Differential Evolution with multiple strategies ---
    # Success history adaptive DE (SHADE-like)
    memory_size = 20
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    while elapsed() < max_time * 0.75:
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            # Select from memory
            ri = np.random.randint(0, memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            # Generate F from Cauchy
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            # Generate CR from Normal
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # p-best index
            p = max(2, int(pop_size * np.random.uniform(0.05, 0.2)))
            p_best_idx = np.random.randint(0, p)
            
            # Mutation: current-to-pbest/1 with archive
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            if i in combined:
                combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            r2_idx = np.random.choice(combined) if combined else r1
            
            if r2_idx < pop_size:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            # Bounce-back
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            # Crossover
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial_fitness = evaluate(trial)
            
            if trial_fitness <= fitness[i]:
                if trial_fitness < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(abs(fitness[i] - trial_fitness))
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[mem_idx] = mean_F
            M_CR[mem_idx] = mean_CR
            mem_idx = (mem_idx + 1) % memory_size
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 30:
            n_replace = pop_size // 3
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: CMA-ES-like local search around best ---
    if best_params is not None:
        sigma = 0.05
        mean = best_params.copy()
        lam = max(8, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_cma = weights_cma / weights_cma.sum()
        
        while elapsed() < max_time * 0.92:
            samples = []
            f_samples = []
            for _ in range(lam):
                if elapsed() >= max_time * 0.92:
                    break
                z = np.random.randn(dim)
                x = clip(mean + sigma * ranges * z)
                f = evaluate(x)
                samples.append(x)
                f_samples.append(f)
            
            if len(samples) < mu:
                break
            
            idx_sorted = np.argsort(f_samples)
            selected = np.array([samples[idx_sorted[j]] for j in range(mu)])
            new_mean = np.dot(weights_cma, selected)
            
            # Adapt sigma
            if f_samples[idx_sorted[0]] < best:
                sigma *= 1.05
            else:
                sigma *= 0.85
            sigma = max(sigma, 1e-8)
            
            mean = clip(new_mean)
    
    # --- Phase 4: Fine local search ---
    if best_params is not None:
        scale = 0.001
        while elapsed() < max_time * 0.98:
            candidate = best_params + scale * ranges * np.random.randn(dim)
            candidate = clip(candidate)
            evaluate(candidate)
    
    return best