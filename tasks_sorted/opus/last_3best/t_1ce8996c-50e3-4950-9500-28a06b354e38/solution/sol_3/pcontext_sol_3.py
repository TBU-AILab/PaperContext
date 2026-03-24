#Looking at the results, algorithm 1 (score 32.04) is the best. Let me analyze what works and what could be improved:
#
#1. DE/current-to-best/1 with coordinate descent refinement works well
#2. I should add: **Nelder-Mead** after DE (it helped in algo 3 concept), **adaptive F/CR per individual**, **better population diversity management**, **golden section search** for local refinement, and **multiple restarts of local search from top individuals**
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
    
    evals = [0]
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: LHS initial sampling ---
    pop_size = min(max(15 * dim, 50), 150)
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
    
    # --- Phase 2: DE/current-to-best/1 with per-individual adaptive params ---
    ind_F = np.full(pop_size, 0.5)
    ind_CR = np.full(pop_size, 0.9)
    
    stagnation = 0
    last_best = best
    gen = 0
    
    while elapsed() < max_time * 0.70:
        gen += 1
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            # Adaptive F and CR with Cauchy/Normal
            Fi = np.clip(ind_F[i] + 0.1 * np.random.standard_cauchy(), 0.1, 1.0)
            CRi = np.clip(ind_CR[i] + 0.1 * np.random.randn(), 0.05, 1.0)
            
            # p-best selection (top 15%)
            p = max(2, int(0.15 * pop_size))
            p_best_idx = np.random.randint(p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[a] - population[b])
            
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.clip(np.where(cross, mutant, population[i]), lower, upper)
            f_trial = eval_f(trial)
            
            if f_trial < fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                ind_F[i] = Fi
                ind_CR[i] = CRi
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        ind_F = ind_F[sorted_idx]
        ind_CR = ind_CR[sorted_idx]
        
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
                ind_F[i] = 0.5; ind_CR[i] = 0.9
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]; fitness = fitness[sorted_idx]
            ind_F = ind_F[sorted_idx]; ind_CR = ind_CR[sorted_idx]
    
    # --- Phase 3: Multi-scale coordinate descent from top candidates ---
    candidates = [population[i].copy() for i in range(min(3, pop_size))]
    for cand in candidates:
        cur = cand.copy(); cur_f = eval_f(cur)
        for scale in [0.05, 0.01, 0.002, 0.0004, 0.00008]:
            step = ranges * scale
            improved = True
            while improved:
                improved = False
                for i in range(dim):
                    if elapsed() >= max_time * 0.99:
                        return best
                    for d in [1, -1]:
                        trial = cur.copy(); trial[i] += d * step[i]
                        trial = np.clip(trial, lower, upper)
                        ft = eval_f(trial)
                        if ft < cur_f:
                            cur = trial; cur_f = ft; improved = True
    
    return best
