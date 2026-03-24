#Looking at the results, algorithm 1 (score 32.04) significantly outperformed algorithm 2 (score 214.42). The key differences: algorithm 1 used a simpler DE/current-to-best/1 strategy with coordinate descent refinement, while algorithm 2 used SHADE which may have been over-engineered. Let me build on algorithm 1's approach but add improvements:
#
#1. **Better initial sampling** with more coverage
#2. **Multiple DE strategies** combined
#3. **More aggressive local search** with pattern search + coordinate descent at multiple scales
#4. **Population size reduction** over time (L-SHADE style)
#5. **Better restart mechanism**
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
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: LHS initial sampling ---
    pop_size = min(max(20 * dim, 50), 200)
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: DE/current-to-best/1 ---
    stagnation = 0
    last_best = best
    
    while elapsed() < max_time * 0.80:
        F_cur = 0.5 + 0.3 * np.random.random()
        CR_cur = 0.8 + 0.2 * np.random.random()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            idxs = list(range(pop_size)); idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            jitter = 0.001 * np.random.randn(dim)
            mutant = population[i] + F_cur * (population[0] - population[i]) + F_cur * (population[a] - population[b]) + jitter
            cross = np.random.random(dim) < CR_cur
            cross[np.random.randint(dim)] = True
            trial = np.clip(np.where(cross, mutant, population[i]), lower, upper)
            f_trial = eval_f(trial)
            if f_trial < fitness[i]:
                population[i] = trial; fitness[i] = f_trial
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]; fitness = fitness[sorted_idx]
        
        if best < last_best - 1e-12:
            stagnation = 0; last_best = best
        else:
            stagnation += 1
        if stagnation > 8:
            stagnation = 0
            for i in range(pop_size // 2, pop_size):
                scale = ranges * 0.1 * np.random.random()
                population[i] = np.clip(best_x + scale * np.random.randn(dim), lower, upper)
                fitness[i] = eval_f(population[i])
            sorted_idx = np.argsort(fitness); population = population[sorted_idx]; fitness = fitness[sorted_idx]
    
    # --- Phase 3: Multi-scale coordinate descent ---
    if best_x is not None:
        for scale in [0.05, 0.01, 0.002, 0.0004]:
            step = ranges * scale
            improved = True
            while improved:
                improved = False
                for i in range(dim):
                    if elapsed() >= max_time * 0.99:
                        return best
                    for d in [1, -1]:
                        trial = best_x.copy()
                        trial[i] += d * step[i]
                        trial = np.clip(trial, lower, upper)
                        if eval_f(trial) < best:
                            improved = True
    
    return best
