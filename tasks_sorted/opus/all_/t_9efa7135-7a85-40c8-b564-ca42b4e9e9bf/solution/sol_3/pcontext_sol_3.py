#Looking at the results, algorithm #2 performed best (8.53). I'll improve upon it by:
#
#1. Better population sizing and initialization
#2. More aggressive L-SHADE with population reduction
#3. Improved local search with Nelder-Mead simplex method
#4. Multiple local search restarts from top individuals
#5. Better time allocation
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_solution = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    # --- Phase 1: LHS initial population ---
    pop_size_init = min(max(12 * dim, 40), 200)
    pop_size = pop_size_init
    min_pop_size = max(4, dim)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # --- Phase 2: L-SHADE ---
    H = 100
    M_F = np.full(H, 0.3)
    M_CR = np.full(H, 0.8)
    h_idx = 0
    archive = []
    archive_max = pop_size_init
    
    stagnation_counter = 0
    last_best = best
    gen = 0
    total_evals_estimate = pop_size_init  # rough count
    
    while elapsed() < max_time * 0.65:
        gen += 1
        sorted_idx = np.argsort(fitness)
        
        S_F, S_CR, S_w = [], [], []
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        p_min = max(2.0 / pop_size, 0.05)
        p_max = 0.25
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            Fi = -1
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CRi = 0.0
            else:
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best
            p = np.random.uniform(p_min, p_max)
            p_num = max(2, int(np.ceil(p * pop_size)))
            pbest_idx = sorted_idx[np.random.randint(0, p_num)]
            
            # r1
            candidates = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(candidates)
            
            # r2 from pop + archive
            combined_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CRi
            cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Boundary handling
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2.0
            trial[above] = (upper[above] + population[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    delta = abs(fitness[i] - trial_f)
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_w.append(delta)
                new_population[i] = trial
                new_fitness[i] = trial_f
        
        population = new_population
        fitness = new_fitness
        
        if len(S_F) > 0:
            weights = np.array(S_w)
            weights /= weights.sum() + 1e-30
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[h_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[h_idx] = np.sum(weights * scr)
            h_idx = (h_idx + 1) % H
        
        # L-SHADE population reduction
        ratio = elapsed() / (max_time * 0.65)
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        if new_pop_size < pop_size:
            sidx = np.argsort(fitness)
            population = population[sidx[:new_pop_size]]
            fitness = fitness[sidx[:new_pop_size]]
            pop_size = new_pop_size
        
        if best < last_best - 1e-12:
            stagnation_counter = 0
            last_best = best
        else:
            stagnation_counter += 1
        
        if stagnation_counter > 25 + dim:
            sidx = np.argsort(fitness)
            keep = max(2, pop_size // 4)
            for idx in sidx[keep:]:
                if elapsed() >= max_time * 0.65:
                    break
                population[idx] = lower + np.random.random(dim) * ranges
                fitness[idx] = evaluate(population[idx])
            stagnation_counter = 0
    
    # --- Phase 3: Local search from multiple starting points ---
    if best_solution is None:
        return best
    
    sidx = np.argsort(fitness)
    starts = [best_solution.copy()] + [population[sidx[i]].copy() for i in range(min(3, pop_size))]
    
    for s_idx, current in enumerate(starts):
        if elapsed() >= max_time * 0.98:
            return best
        current_f = evaluate(current)
        step = ranges * 0.02
        
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    return best
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[d] = np.clip(trial[d] + sign * step[d], lower[d], upper[d])
                    f_trial = evaluate(trial)
                    if f_trial < current_f:
                        current = trial
                        current_f = f_trial
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-14:
                    break
    
    return best
