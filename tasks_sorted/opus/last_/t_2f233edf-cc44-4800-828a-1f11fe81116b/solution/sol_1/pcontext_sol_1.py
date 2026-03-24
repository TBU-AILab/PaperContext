#Looking at the previous algorithm, I'll improve it with several enhancements:
#
#1. **Better initialization** with multiple strategies
#2. **SHADE (Success-History based Adaptive DE)** for better parameter adaptation
#3. **Nelder-Mead local search** on the best solution periodically
#4. **Restart mechanism** with opposition-based learning
#5. **More efficient time management**
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
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Nelder-Mead local search ---
    def nelder_mead(start_point, budget_evals=None, initial_scale=0.05):
        n = dim
        if budget_evals is None:
            budget_evals = 200 * n
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = start_point.copy()
        f_values = np.zeros(n + 1)
        f_values[0] = eval_func(simplex[0])
        evals = 1
        
        for i in range(n):
            if time_left() <= 0:
                return evals
            point = start_point.copy()
            point[i] += initial_scale * ranges[i]
            if point[i] > upper[i]:
                point[i] = start_point[i] - initial_scale * ranges[i]
            simplex[i + 1] = clip(point)
            f_values[i + 1] = eval_func(simplex[i + 1])
            evals += 1
        
        while evals < budget_evals and time_left() > 0:
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_func(xr); evals += 1
            
            if fr < f_values[0]:
                # Expand
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe); evals += 1
                if fe < fr:
                    simplex[-1], f_values[-1] = xe, fe
                else:
                    simplex[-1], f_values[-1] = xr, fr
            elif fr < f_values[-2]:
                simplex[-1], f_values[-1] = xr, fr
            else:
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc); evals += 1
                    if fc <= fr:
                        simplex[-1], f_values[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if time_left() <= 0: return evals
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_values[i] = eval_func(simplex[i]); evals += 1
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[-1]))
                    fc = eval_func(xc); evals += 1
                    if fc < f_values[-1]:
                        simplex[-1], f_values[-1] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if time_left() <= 0: return evals
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_values[i] = eval_func(simplex[i]); evals += 1
            
            # Convergence check
            if np.max(np.abs(f_values - f_values[0])) < 1e-15:
                break
        
        return evals

    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(15 * dim, 40), 150)
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.array([eval_func(population[i]) for i in range(pop_size) if time_left() > 0])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
    pop_size = len(fitness)
    
    # --- SHADE parameters ---
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    gen = 0
    
    # --- Phase 2: SHADE + periodic local search ---
    while time_left() > 0:
        gen += 1
        
        # Periodic local search
        if gen % 25 == 0 and best_params is not None and time_left() > 1:
            scale = max(0.001, 0.05 * (time_left() / max_time))
            nelder_mead(best_params, budget_evals=min(500, 50 * dim), initial_scale=scale)
        
        S_F, S_CR = [], []
        delta_f = []
        
        sort_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_left() <= 0:
                return best
            
            ri = np.random.randint(H)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0, 1)
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0, 1)
            
            # p-best index
            pi = sort_idx[np.random.randint(p_best_size)]
            
            # Select r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            # Select r2 from pop + archive, != i, r1
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(union_size)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - xr2)
            
            # Bounce-back clipping
            for j in range(dim):
                if mutant[j] < lower[j]:
                    mutant[j] = (lower[j] + population[i][j]) / 2
                elif mutant[j] > upper[j]:
                    mutant[j] = (upper[j] + population[i][j]) / 2
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            trial_f = eval_func(trial)
            
            if trial_f < fitness[i]:
                # Archive old vector
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif len(archive) > 0:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                
                new_pop[i] = trial
                new_fit[i] = trial_f
                S_F.append(Fi)
                S_CR.append(CRi)
                delta_f.append(fitness[i] - trial_f)
            elif trial_f == fitness[i]:
                new_pop[i] = trial
                new_fit[i] = trial_f
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)
            
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            
            M_F[k] = mean_F
            M_CR[k] = mean_CR
            k = (k + 1) % H
        
        # Stagnation check
        if best < prev_best - 1e-14:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 30:
            stagnation = 0
            sort_idx = np.argsort(fitness)
            keep = max(3, pop_size // 4)
            for i in range(keep, pop_size):
                if time_left() <= 0:
                    return best
                if np.random.random() < 0.4:
                    # Opposition-based
                    population[i] = lower + upper - population[sort_idx[np.random.randint(keep)]]
                elif np.random.random() < 0.5:
                    population[i] = best_params + np.random.randn(dim) * ranges * 0.1 * np.random.random()
                else:
                    population[i] = lower + np.random.random(dim) * ranges
                population[i] = clip(population[i])
                fitness[i] = eval_func(population[i])
            
            M_F = np.full(H, 0.5)
            M_CR = np.full(H, 0.5)
            archive = []
    
    # Final local search
    if best_params is not None and time_left() > 0:
        nelder_mead(best_params, budget_evals=100, initial_scale=0.001)
    
    return best
#
#Key improvements over the previous version:
#
#1. **SHADE algorithm**: Uses success-history based adaptation of F and CR parameters, which is much more effective than random parameter selection.
#2. **Weighted Lehmer mean** for F adaptation and weighted arithmetic mean for CR - proven to converge better.
#3. **External archive** of replaced solutions for better exploration.
#4. **Nelder-Mead local search** periodically refines the best solution found.
#5. **Bounce-back clipping** instead of simple clipping to maintain population diversity near boundaries.
#6. **Opposition-based learning** in restart mechanism for better exploration after stagnation.
#7. **p-best mutation strategy** (current-to-pbest/1) which is the core of SHADE and balances exploration/exploitation.
