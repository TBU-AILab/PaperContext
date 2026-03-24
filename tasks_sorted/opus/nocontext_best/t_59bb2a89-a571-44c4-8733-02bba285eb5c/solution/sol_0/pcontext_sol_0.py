#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate LHS-like initial population
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    # DE/current-to-pbest/1 with archive (SHADE-like simplified)
    
    F_mean = 0.5
    CR_mean = 0.5
    archive = []
    max_archive = pop_size
    
    generation = 0
    no_improve_count = 0
    prev_best = best
    
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.92):
            break
        
        generation += 1
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Sort population by fitness for pbest selection
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        S_F = []
        S_CR = []
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.92):
                return best
            
            # Generate F and CR
            F = np.clip(np.random.standard_cauchy() * 0.1 + F_mean, 0, 1)
            CR = np.clip(np.random.normal(CR_mean, 0.1), 0, 1)
            
            # Select pbest
            p_best_idx = sorted_idx[np.random.randint(0, p_best_size)]
            
            # Select r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            candidates.remove(r1)
            
            # Select r2 from population + archive
            if len(archive) > 0:
                all_pool = np.vstack([population, np.array(archive)])
            else:
                all_pool = population
            r2 = np.random.randint(0, len(all_pool))
            # Make sure r2 is different from i and r1 in population part
            max_tries = 20
            while r2 < pop_size and (r2 == i or r2 == r1) and max_tries > 0:
                r2 = np.random.randint(0, len(all_pool))
                max_tries -= 1
            
            # Mutation: current-to-pbest/1
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - all_pool[r2])
            
            # Binomial crossover
            trial = np.copy(population[i])
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bound handling: bounce back
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
                # Final clip
                trial[j] = np.clip(trial[j], lower[j], upper[j])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                if trial_fitness < best:
                    best = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        # Adapt F_mean and CR_mean
        if len(S_F) > 0:
            F_mean = 0.5 * F_mean + 0.5 * (np.sum(np.array(S_F)**2) / np.sum(S_F))
            CR_mean = 0.5 * CR_mean + 0.5 * np.mean(S_CR)
        
        # Check stagnation
        if best < prev_best:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # If stagnated, do a partial restart of worst members
        if no_improve_count > 20:
            no_improve_count = 0
            sorted_idx = np.argsort(fitness)
            n_restart = pop_size // 2
            for k in range(n_restart):
                idx = sorted_idx[pop_size - 1 - k]
                population[idx] = lower + np.random.random(dim) * (upper - lower)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
    
    # --- Phase 3: Local search (Nelder-Mead style) on best solution ---
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    
    # Simple pattern search / coordinate descent
    step = 0.01 * (upper - lower)
    
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
            return best
        
        improved = False
        for j in range(dim):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
                return best
            
            trial = best_x.copy()
            trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_x = trial
                improved = True
                continue
            
            trial = best_x.copy()
            trial[j] = np.clip(trial[j] - step[j], lower[j], upper[j])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_x = trial
                improved = True
                continue
        
        if not improved:
            step *= 0.5
            if np.max(step / (upper - lower)) < 1e-12:
                break
    
    return best
