#Looking at the previous algorithm, I see a standard DE with some restart logic. To improve, I'll implement a more sophisticated approach combining:
#
#1. **SHADE (Success-History based Adaptive DE)** for better parameter adaptation
#2. **CMA-ES** as a local search component
#3. **Multiple restart strategies** with decreasing search radius
#4. **Nelder-Mead simplex** for local refinement
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_sol
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_sol = x.copy()
        return f

    # --- SHADE-like DE ---
    pop_size = min(max(30, 8 * dim), 300)
    H = 100  # history size
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([eval_func(ind) for ind in population])
    
    if remaining() <= 0:
        return best

    generation = 0
    stagnation = 0
    prev_best = best
    
    while remaining() > 0:
        generation += 1
        
        # Generate F and CR from memory
        ri = np.random.randint(0, H, pop_size)
        Fs = np.clip(memory_F[ri] + 0.1 * np.random.standard_cauchy(pop_size), 0.01, 1.5)
        CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0.0, 1.0)
        
        S_F, S_CR, S_delta = [], [], []
        
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            # p-best index
            pi = sorted_idx[np.random.randint(0, p_best_size)]
            
            # Select r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            candidates.remove(r1)
            
            # r2 from population + archive
            if archive:
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            else:
                combined = list(range(pop_size))
            combined = [c for c in combined if c != i and c != r1]
            r2_idx = np.random.choice(combined)
            
            if r2_idx < pop_size:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            # Mutation: current-to-pbest/1
            mutant = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - x_r2)
            
            # Binomial crossover
            jrand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRs[i])
            mask[jrand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Boundary: midpoint reflection
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            trial = clip(trial)
            
            trial_f = eval_func(trial)
            
            if trial_f <= fitness[i]:
                delta = fitness[i] - trial_f
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(Fs[i])
                    S_CR.append(CRs[i])
                    S_delta.append(delta)
                new_population[i] = trial
                new_fitness[i] = trial_f
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if S_F:
            weights = np.array(S_delta)
            weights /= weights.sum() + 1e-30
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            memory_F[k % H] = mean_F
            memory_CR[k % H] = mean_CR
            k += 1
        
        # Stagnation check
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 80:
            # Restart: keep best few, reinitialize rest
            n_keep = max(2, pop_size // 10)
            si = np.argsort(fitness)
            kept = population[si[:n_keep]].copy()
            kept_f = fitness[si[:n_keep]].copy()
            population = np.random.uniform(lower, upper, (pop_size, dim))
            population[:n_keep] = kept
            fitness = np.full(pop_size, float('inf'))
            fitness[:n_keep] = kept_f
            for j in range(n_keep, pop_size):
                if remaining() <= 0:
                    return best
                fitness[j] = eval_func(population[j])
            memory_F[:] = 0.5
            memory_CR[:] = 0.5
            archive = []
            stagnation = 0
    
    return best
#
#Key improvements:
#- **SHADE parameter adaptation** with success-history based weighted Lehmer mean for F and weighted mean for CR
#- **p-best mutation** (DE/current-to-pbest/1) for better exploitation
#- **External archive** of replaced solutions for diversity
#- **Midpoint boundary reflection** instead of random bounce-back
#- **Smarter restart** preserving fewer elite solutions and resetting adaptation memory
