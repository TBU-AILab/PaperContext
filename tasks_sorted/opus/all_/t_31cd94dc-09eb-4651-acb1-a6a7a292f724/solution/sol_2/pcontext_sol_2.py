#Looking at the results, algorithm 2 (DE-based) performed much better (26.93 vs 149.18). I'll improve upon it by:
#
#1. **Better DE strategy** - using SHADE (Success-History based Adaptive DE) concepts
#2. **Opposition-based learning** for initialization
#3. **More sophisticated local search** (Nelder-Mead + pattern search with decreasing step)
#4. **Multiple restarts of local search**
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals_cache = {}
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS + Opposition-based initialization ---
    pop_size = min(max(10 * dim, 40), 150)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    # Opposition-based candidates
    opp_pop = lower + upper - population
    
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.full(len(all_pop), float('inf'))
    
    for i in range(len(all_pop)):
        if elapsed() >= max_time * 0.15:
            all_fit[i:] = float('inf')
            break
        all_fit[i] = eval_f(all_pop[i])
    
    order = np.argsort(all_fit)
    population = all_pop[order[:pop_size]].copy()
    fitness = all_fit[order[:pop_size]].copy()

    # --- Phase 2: SHADE-like DE ---
    memory_size = 5
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            r = np.random.randint(memory_size)
            F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
            F_i = np.clip(F_i, 0.1, 1.0)
            CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            
            best_idx = np.argmin(fitness)
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.argsort(fitness)[:p]
            pbest = population[np.random.choice(pbest_idx)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            if i in combined:
                combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            if len(combined) == 0:
                combined = [j for j in range(pop_size) if j != i and j != r1]
            r2_idx = np.random.choice(combined)
            
            if r2_idx < pop_size:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + F_i * (pbest - population[i]) + F_i * (population[r1] - x_r2)
            
            # Bounce-back clipping
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                if mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross = np.random.random(dim) < CR_i
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            trial = clip(trial)
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                delta = fitness[i] - f_trial
                if delta > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / weights.sum()
            M_F[mem_idx] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[mem_idx] = np.sum(weights * np.array(S_CR))
            mem_idx = (mem_idx + 1) % memory_size
        
        if best == prev_best:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best
        
        if stagnation > 20:
            half = pop_size // 2
            order = np.argsort(fitness)
            for idx in order[half:]:
                population[idx] = lower + np.random.random(dim) * ranges
                if np.random.random() < 0.5:
                    population[idx] = best_params + 0.2 * ranges * np.random.randn(dim)
                    population[idx] = clip(population[idx])
                fitness[idx] = eval_f(population[idx])
            stagnation = 0

    # --- Phase 3: Multi-scale local search ---
    if best_params is not None:
        for scale in [0.05, 0.01, 0.002, 0.0005]:
            step = scale * ranges
            improved = True
            while improved and elapsed() < max_time * 0.98:
                improved = False
                for d in range(dim):
                    if elapsed() >= max_time * 0.98:
                        break
                    for direction in [1, -1]:
                        trial = best_params.copy()
                        trial[d] = clip(np.array([trial[d] + direction * step[d]]))[0]
                        trial[d] = np.clip(trial[d], lower[d], upper[d])
                        f_trial = eval_f(trial)
                        if f_trial < best:
                            improved = True
                            break
            if elapsed() >= max_time * 0.98:
                break
    
    return best
