#Looking at the current algorithm, I can see several areas for improvement:
#
#1. **Better DE strategy**: Use SHADE-like adaptive DE with history-based parameter adaptation
#2. **CMA-ES local search**: More effective than Nelder-Mead for continuous optimization
#3. **Better population management and restart strategy**
#4. **Opposition-based learning for initialization**
#
import numpy as np
from datetime import datetime, timedelta
import time

def run(func, dim, bounds, max_time):
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    best = float('inf')
    best_params = None
    
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
    
    # --- Phase 1: Initial sampling with LHS + opposition ---
    pop_size = min(max(10 * dim, 40), 150)
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if time_left() <= 0:
            return best
        fitness[i] = eval_func(population[i])
    
    # Opposition-based candidates
    opp_pop = lower + upper - population
    opp_fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if time_left() <= 0:
            return best
        opp_fitness[i] = eval_func(opp_pop[i])
    
    # Merge: keep best pop_size
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.concatenate([fitness, opp_fitness])
    order = np.argsort(all_fit)[:pop_size]
    population = all_pop[order]
    fitness = all_fit[order]
    
    # --- Phase 2: SHADE-like Differential Evolution ---
    H = 50  # history size
    M_CR = np.full(H, 0.5)
    M_F = np.full(H, 0.5)
    k = 0  # history index
    
    archive = []
    archive_max = pop_size
    
    p_min = max(2 / pop_size, 0.05)
    p_max = 0.25
    
    generation = 0
    stagnation_counter = 0
    prev_best_gen = best
    
    while time_left() > max_time * 0.15:
        generation += 1
        S_CR = []
        S_F = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_left() <= max_time * 0.15:
                break
            
            # Generate CR and F from history
            ri = np.random.randint(H)
            CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            F_i = min(1.0, max(0.01, np.random.standard_cauchy() * 0.1 + M_F[ri]))
            while F_i <= 0:
                F_i = min(1.0, max(0.01, np.random.standard_cauchy() * 0.1 + M_F[ri]))
            
            # current-to-pbest/1
            p = np.random.uniform(p_min, p_max)
            n_pbest = max(1, int(p * pop_size))
            pbest_idx = np.random.choice(np.argsort(fitness)[:n_pbest])
            
            indices = list(range(pop_size))
            indices.remove(i)
            r1 = np.random.choice(indices)
            
            # r2 from pop + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            if i in combined:
                combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            r2_idx = np.random.choice(combined) if combined else r1
            
            if r2_idx < pop_size:
                xr2 = population[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - xr2)
            
            # Bounce-back clipping
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR_i
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial_f = eval_func(trial)
            
            if trial_f <= fitness[i]:
                delta = fitness[i] - trial_f
                if delta > 0:
                    S_CR.append(CR_i)
                    S_F.append(F_i)
                    S_delta.append(delta)
                archive.append(population[i].copy())
                if len(archive) > archive_max:
                    archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fit[i] = trial_f
        
        population = new_pop
        fitness = new_fit
        
        # Update history
        if S_CR and S_F:
            weights = np.array(S_delta)
            weights = weights / weights.sum()
            s_f = np.array(S_F)
            s_cr = np.array(S_CR)
            M_CR[k] = np.sum(weights * s_cr)
            M_F[k] = np.sum(weights * s_f ** 2) / (np.sum(weights * s_f) + 1e-30)
            k = (k + 1) % H
        
        # Check stagnation
        if best < prev_best_gen - 1e-12:
            stagnation_counter = 0
            prev_best_gen = best
        else:
            stagnation_counter += 1
        
        # Restart worst half if stagnating
        if stagnation_counter > max(15, 50 // max(1, dim // 10)):
            order = np.argsort(fitness)
            n_replace = pop_size // 2
            for idx in order[-n_replace:]:
                population[idx] = lower + np.random.random(dim) * ranges
                if time_left() <= max_time * 0.15:
                    break
                fitness[idx] = eval_func(population[idx])
            stagnation_counter = 0
            M_CR[:] = 0.5
            M_F[:] = 0.5
    
    # --- Phase 3: CMA-ES-like local search around best ---
    if best_params is not None and time_left() > 0.5:
        sigma = 0.02 * np.mean(ranges)
        x_mean = best_params.copy()
        lam = min(4 + int(3 * np.log(dim)), 30)
        mu = lam // 2
        weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_cma = weights_cma / weights_cma.sum()
        C = np.eye(dim)
        
        for _ in range(500):
            if time_left() <= 0.1:
                break
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                L = np.eye(dim)
            
            samples = []
            sample_f = []
            for j in range(lam):
                if time_left() <= 0.05:
                    return best
                z = np.random.randn(dim)
                x = x_mean + sigma * L @ z
                x = clip(x)
                f = eval_func(x)
                samples.append(x)
                sample_f.append(f)
            
            order = np.argsort(sample_f)
            x_new = np.zeros(dim)
            for j in range(mu):
                x_new += weights_cma[j] * samples[order[j]]
            
            # Rank-mu update (simplified)
            C_new = np.zeros((dim, dim))
            for j in range(mu):
                diff = (samples[order[j]] - x_mean) / (sigma + 1e-30)
                C_new += weights_cma[j] * np.outer(diff, diff)
            
            C = 0.8 * C + 0.2 * C_new
            x_mean = x_new
            sigma *= 0.999
    
    return best
