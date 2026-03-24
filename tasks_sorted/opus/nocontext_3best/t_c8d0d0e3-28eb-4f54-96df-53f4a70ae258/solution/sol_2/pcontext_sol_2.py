#Looking at the two algorithms, #1 performs significantly better (13.38 vs 28.18). I'll improve upon #1 by incorporating:
#
#1. Better SHADE implementation with linear population size reduction (L-SHADE)
#2. Multiple restarts with different strategies
#3. Improved local search with adaptive step sizes
#4. Better time budget allocation
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def evaluate(x):
        nonlocal best, best_params, evals
        evals += 1
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size_init = min(max(40, 18 * dim), 400)
    pop_size = pop_size_init
    min_pop_size = max(6, dim)
    
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
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: L-SHADE ---
    memory_size = 30
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size_init
    
    stagnation = 0
    prev_best = best
    generation = 0
    max_gen_estimate = max(100, int(max_time * 50))
    
    while elapsed() < max_time * 0.70:
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            ri = np.random.randint(0, memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            # Cauchy for F
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            # Normal for CR
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            if mu_CR < 0:
                CR_i = 0.0
            
            # p-best
            p = max(2, int(pop_size * np.random.uniform(0.05, 0.25)))
            p_best_idx = np.random.randint(0, p)
            
            # Mutation: current-to-pbest/1 with archive
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            # Bounce-back
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
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
        
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[mem_idx] = mean_F
            M_CR[mem_idx] = mean_CR
            mem_idx = (mem_idx + 1) % memory_size
        
        # Linear population size reduction
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * generation / max_gen_estimate)))
        if new_pop_size < pop_size:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        else:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 40:
            n_replace = max(1, pop_size // 3)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: CMA-ES-like local search ---
    if best_params is not None:
        sigma = 0.02
        mean = best_params.copy()
        lam = max(8, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        
        while elapsed() < max_time * 0.93:
            samples, f_s = [], []
            for _ in range(lam):
                if elapsed() >= max_time * 0.93:
                    break
                x = clip(mean + sigma * ranges * np.random.randn(dim))
                f = evaluate(x)
                samples.append(x); f_s.append(f)
            if len(samples) < mu:
                break
            idx_s = np.argsort(f_s)
            sel = np.array([samples[idx_s[j]] for j in range(mu)])
            new_mean = np.dot(w, sel)
            if f_s[idx_s[0]] < func(mean):
                sigma *= 1.1
            else:
                sigma *= 0.82
            sigma = np.clip(sigma, 1e-10, 0.5)
            mean = clip(new_mean)
    
    # --- Phase 4: Fine-grained local search ---
    if best_params is not None:
        scale = 0.0005
        while elapsed() < max_time * 0.99:
            candidate = clip(best_params + scale * ranges * np.random.randn(dim))
            evaluate(candidate)
    
    return best
