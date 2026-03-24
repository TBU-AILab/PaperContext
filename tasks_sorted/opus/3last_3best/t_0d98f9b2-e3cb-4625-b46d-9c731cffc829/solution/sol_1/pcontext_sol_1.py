#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Boundary handling is slow** (per-dimension loop)
#2. **DE parameters could be more adaptive** (SHADE/L-SHADE style)
#3. **Local search could use CMA-ES-like approach** instead of Nelder-Mead for better high-dim performance
#4. **Population size could be more dynamic**
#5. **More exploitation near the best solutions**
#
#Here's my improved version:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip_to_bounds(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- SHADE-like Adaptive DE ---
    pop_size = min(max(30, 8 * dim), 300)
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])
    
    # SHADE memory
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.75:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.73:
                break
            
            ri = np.random.randint(0, H)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            # Cauchy for F, Normal for CR
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            union_size = len(archive)
            if union_size > 0 and np.random.random() < 0.5:
                r2_arc = archive[np.random.randint(0, union_size)]
                diff2 = population[r1] - r2_arc
            else:
                r2 = np.random.choice([j for j in idxs if j != r1]) if len(idxs) > 1 else r1
                diff2 = population[r1] - population[r2]
            
            mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * diff2
            
            # Binomial crossover
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR_i
            mask[j_rand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Midpoint bounce-back
            out_low = trial < lower
            out_high = trial > upper
            trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
            trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
            trial = clip_to_bounds(trial)
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness[i]:
                delta = fitness[i] - f_trial
                if f_trial < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[k % H] = mean_F
            M_CR[k % H] = mean_CR
            k += 1
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            sorted_idx = np.argsort(fitness)
            n_reset = pop_size // 3
            for j in range(n_reset):
                idx = sorted_idx[pop_size - 1 - j]
                if np.random.random() < 0.5:
                    population[idx] = best_x + 0.05 * ranges * np.random.randn(dim)
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip_to_bounds(population[idx])
                if elapsed() >= max_time * 0.73:
                    break
                fitness[idx] = evaluate(population[idx])
            stagnation = 0
    
    # --- Phase 2: Local search with adaptive step coordinate descent ---
    if best_x is not None:
        x = best_x.copy()
        fx = best
        step = 0.01 * ranges
        
        while elapsed() < max_time * 0.95:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95:
                    return best
                
                x_trial = x.copy()
                x_trial[d] = min(x[d] + step[d], upper[d])
                ft = evaluate(x_trial)
                if ft < fx:
                    x = x_trial
                    fx = ft
                    step[d] *= 1.2
                    improved = True
                    continue
                
                x_trial = x.copy()
                x_trial[d] = max(x[d] - step[d], lower[d])
                ft = evaluate(x_trial)
                if ft < fx:
                    x = x_trial
                    fx = ft
                    step[d] *= 1.2
                    improved = True
                    continue
                
                step[d] *= 0.5
            
            if not improved or np.max(step) < 1e-15:
                step = 0.001 * ranges * (0.1 + 0.9 * np.random.random(dim))
    
    return best
#
#Key improvements:
#- **SHADE adaptive parameter control**: Lehmer mean for F, weighted mean for CR based on fitness improvement
#- **External archive** for diversity in mutation
#- **Vectorized boundary handling** (midpoint bounce-back, no per-dim loop)
#- **Coordinate descent local search** instead of Nelder-Mead (much cheaper per iteration, scales better with dim)
#- **Better time budget allocation**: 75% for DE, 20% for local search
#- **Cauchy distribution for F** (heavier tails = better exploration)
