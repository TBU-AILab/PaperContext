

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(200, max(30, 15 * dim))
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if remaining() <= 0:
            return best
        fitness[i] = evaluate(population[i])
    
    # Sort population by fitness
    idx = np.argsort(fitness)
    population = population[idx]
    fitness = fitness[idx]
    
    # --- Phase 2: Hybrid DE with multiple strategies and adaptive restart ---
    
    # Success history for adaptive F and CR (SHADE-like)
    H = 50
    MF = np.full(H, 0.5)
    MCR = np.full(H, 0.5)
    hist_pos = 0
    
    stagnation_counter = 0
    prev_best = best
    generation = 0
    
    while remaining() > max_time * 0.15:
        generation += 1
        
        SF = []
        SCR = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        # Generate F and CR from history
        ri = np.random.randint(0, H, pop_size)
        F_vals = np.clip(np.random.standard_cauchy(pop_size) * 0.1 + MF[ri], 0.01, 1.0)
        CR_vals = np.clip(np.random.normal(MCR[ri], 0.1), 0.0, 1.0)
        
        # p-best indices for current-to-pbest
        p = max(2, int(0.1 * pop_size))
        
        for i in range(pop_size):
            if remaining() <= max_time * 0.15:
                break
            
            Fi = F_vals[i]
            CRi = CR_vals[i]
            
            # current-to-pbest/1
            pbest_idx = np.random.randint(0, p)
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Bounce-back clipping
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial = clip(trial)
            f_trial = evaluate(trial)
            
            if f_trial <= new_fit[i]:
                if f_trial < new_fit[i]:
                    SF.append(Fi)
                    SCR.append(CRi)
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        # Update history
        if SF:
            weights = np.array([abs(fitness[j] - new_fit[j]) for j in range(pop_size) if fitness[j] > new_fit[j]])
            if len(weights) > 0 and np.sum(weights) > 0:
                weights /= np.sum(weights)
                MF[hist_pos] = np.sum(weights * np.array(SF)**2) / (np.sum(weights * np.array(SF)) + 1e-30)
                MCR[hist_pos] = np.sum(weights * np.array(SCR))
                hist_pos = (hist_pos + 1) % H
        
        population = new_pop
        fitness = new_fit
        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            prev_best = best
        
        if stagnation_counter > 30:
            n_replace = pop_size // 2
            for k in range(n_replace):
                if remaining() <= max_time * 0.15:
                    break
                idx_r = pop_size - 1 - k
                new_ind = lower + np.random.random(dim) * ranges
                population[idx_r] = new_ind
                fitness[idx_r] = evaluate(new_ind)
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
            stagnation_counter = 0
    
    # --- Phase 3: Nelder-Mead local search on best solution ---
    if best_params is not None and remaining() > 0:
        n = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        # Build simplex around best
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step = ranges * 0.05
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([evaluate(clip(simplex[i])) for i in range(n + 1) if remaining() > 0])
        if len(f_simplex) < n + 1:
            return best
        
        while remaining() > 0:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            if remaining() <= 0: break
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if remaining() <= 0: break
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = evaluate(xc)
                if remaining() <= 0: break
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        if remaining() <= 0: break
                        simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        f_simplex[i] = evaluate(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-12:
                break
    
    return best