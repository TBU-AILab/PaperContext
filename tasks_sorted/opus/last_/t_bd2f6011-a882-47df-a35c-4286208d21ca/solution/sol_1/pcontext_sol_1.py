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
    
    def time_ok():
        return elapsed() < max_time * 0.97
    
    # ---- Parameters ----
    pop_size = min(max(30, 10 * dim), 300)
    
    # ---- Initialize population with Latin Hypercube Sampling ----
    def lhs_sample(n):
        result = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                result[i, d] = (perm[i] + np.random.random()) / n
        return lower + result * ranges
    
    population = lhs_sample(pop_size)
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_sol = population[best_idx].copy()
    best = best_val
    
    # Success history for SHADE-like adaptation
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    # Archive for DE/current-to-pbest
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation_counter = 0
    prev_best = best
    
    while time_ok():
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        # Sort population for p-best
        sorted_idx = np.argsort(fitness)
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if not time_ok():
                return best
            
            # Generate F and CR from history
            ri = np.random.randint(0, H)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.5)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best: pick from top p fraction
            p = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p)]
            
            # Select r1 != i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # Select r2 from population + archive, != i, r1
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined = [x for x in combined if x != i and x != r1]
            r2 = np.random.choice(combined)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # Mutation: current-to-pbest/1
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Boundary: bounce-back
            mask_lo = mutant < lower
            mask_hi = mutant > upper
            mutant[mask_lo] = lower[mask_lo] + np.random.random(np.sum(mask_lo)) * (population[i][mask_lo] - lower[mask_lo])
            mutant[mask_hi] = upper[mask_hi] - np.random.random(np.sum(mask_hi)) * (upper[mask_hi] - population[i][mask_hi])
            mutant = np.clip(mutant, lower, upper)
            
            # Binomial crossover
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            trial_fit = func(trial)
            if trial_fit <= fitness[i]:
                delta = fitness[i] - trial_fit
                if delta > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    else:
                        archive[np.random.randint(archive_max)] = population[i].copy()
                new_population[i] = trial
                new_fitness[i] = trial_fit
                if trial_fit < best:
                    best = trial_fit
                    best_sol = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Update history
        if S_F:
            weights = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % H
        
        # Stagnation restart
        if abs(prev_best - best) < 1e-15:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best = best
        
        if stagnation_counter > 30 + dim:
            keep = max(1, pop_size // 5)
            si = np.argsort(fitness)
            population[keep:] = lhs_sample(pop_size - keep)
            for j in range(keep, pop_size):
                if not time_ok():
                    return best
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_sol = population[j].copy()
            population[:keep] = population[si[:keep]]
            fitness[:keep] = fitness[si[:keep]]
            stagnation_counter = 0
            M_F[:] = 0.5
            M_CR[:] = 0.5
            archive.clear()
    
    return best