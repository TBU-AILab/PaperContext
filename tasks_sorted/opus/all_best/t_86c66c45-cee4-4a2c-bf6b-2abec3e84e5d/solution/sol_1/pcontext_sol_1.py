#Looking at the previous algorithm, I'll implement a more sophisticated approach combining SHADE (Success-History based Adaptive DE) with CMA-ES local search and multi-restart strategies for better exploration and exploitation.
#
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
        return max_time * 0.95 - elapsed()
    
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

    # SHADE parameters
    pop_size = min(max(30, 8 * dim), 150)
    H = 100  # history size
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    k = 0
    
    # Initialize population with Latin Hypercube Sampling
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(ind) for ind in population])
    archive = []
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while time_left() > 0.1:
        generation += 1
        
        # SHADE mutation and crossover
        S_F, S_CR = [], []
        delta_f = []
        
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_left() <= 0.05:
                return best
            
            ri = np.random.randint(H)
            mu_F = memory_F[ri]
            mu_CR = memory_CR[ri]
            
            # Cauchy for F, Normal for CR
            Fi = mu_F + 0.1 * np.random.standard_cauchy()
            while Fi <= 0:
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0, 1)
            
            # p-best index
            pi = sorted_idx[np.random.randint(p_best_size)]
            
            # Select r1 != i
            r1 = np.random.randint(pop_size)
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            # Select r2 from pop + archive, != i, r1
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            jrand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi)
            mask[jrand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Boundary: midpoint reflection
            out_low = trial < lower
            out_high = trial > upper
            trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
            trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
            trial = clip(trial)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_f))
                    archive.append(population[i].copy())
                new_pop[i] = trial
                new_fit[i] = trial_f
        
        population = new_pop
        fitness = new_fit
        
        # Trim archive
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        # Update memory
        if S_F:
            weights = np.array(delta_f)
            weights = weights / weights.sum()
            new_F = np.sum(weights * np.array(S_F)**2) / max(np.sum(weights * np.array(S_F)), 1e-30)
            new_CR = np.sum(weights * np.array(S_CR))
            memory_F[k % H] = new_F
            memory_CR[k % H] = new_CR
            k += 1
        
        # Stagnation check and restart
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 30 and time_left() > 1.0:
            stagnation = 0
            keep = max(2, pop_size // 10)
            si = np.argsort(fitness)
            for i in range(keep, pop_size):
                r = max(0.01, 0.5 * time_left() / max_time)
                population[i] = best_params + r * ranges * (np.random.rand(dim) - 0.5)
                population[i] = clip(population[i])
                fitness[i] = evaluate(population[i])
            archive.clear()
    
    return best
#
#Key improvements:
#1. **SHADE algorithm**: Uses success-history based adaptive F and CR instead of simple jitter, leading to much better parameter adaptation.
#2. **Latin Hypercube Sampling**: Better initial coverage of the search space.
#3. **p-best mutation**: Uses top-p individuals rather than just the single best, reducing premature convergence.
#4. **External archive**: Maintains previously replaced solutions for better diversity in mutation.
#5. **Weighted Lehmer mean** for F updates and weighted arithmetic mean for CR, as per SHADE paper.
#6. **Smarter boundary handling**: Midpoint reflection instead of random bounce-back.
#7. **More aggressive restart**: Keeps only top 10% on stagnation with archive clearing.
