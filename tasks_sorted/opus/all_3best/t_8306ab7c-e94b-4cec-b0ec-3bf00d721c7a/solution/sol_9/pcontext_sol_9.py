#Looking at the results, the best algorithm scores 3.26 with SHADE + multi-start local search. I'll analyze what works and what doesn't:
#
#**Key observations:**
#1. SHADE phase is effective for exploration
#2. The local search wastes evals with `eval_func(mean)` calls
#3. Population reduction (L-SHADE) sometimes helps, sometimes not
#4. The local search sigma adaptation could be better
#5. Need better diversity in restart points and more aggressive exploitation
#
#**Improvements:**
#1. Remove redundant `eval_func(mean)` call
#2. Use adaptive population sizing based on dim
#3. Better local search: combine ES with coordinate-wise refinement
#4. Smarter restart strategy with increasing perturbation
#5. More time for SHADE when it's productive
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
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(30, 8 * dim), 200)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[perm[j], i] = (j + np.random.random()) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_func(population[i])

    # --- Phase 2: SHADE ---
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    init_pop_size = pop_size
    min_pop_size = max(4, dim // 2)
    
    shade_time = 0.55
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * shade_time:
        S_F, S_CR, S_delta = [], [], []
        sorted_idx = np.argsort(fitness[:pop_size])
        
        for i in range(pop_size):
            if elapsed() >= max_time * shade_time:
                break
            
            ri = np.random.randint(memory_size)
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 10:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi >= 2.0:
                    Fi = -1
                attempts += 1
            if Fi <= 0:
                Fi = 0.5
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            pbest = sorted_idx[np.random.randint(0, p)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = idxs[np.random.randint(len(idxs))]
            
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            trial_fit = eval_func(trial)
            
            if trial_fit <= fitness[i]:
                delta = fitness[i] - trial_fit
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = trial_fit
        
        if S_F:
            w = np.array(S_delta) / (sum(S_delta) + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % memory_size
        
        if best < prev_best:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        ratio = elapsed() / (max_time * shade_time)
        new_pop = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * ratio)))
        if new_pop < pop_size:
            idx = np.argsort(fitness[:pop_size])[:new_pop]
            population = population[idx].copy()
            fitness = fitness[idx].copy()
            pop_size = new_pop
        
        if stagnation > 5 and pop_size > min_pop_size:
            break

    # --- Phase 3: Multi-start ES local search ---
    sorted_idx = np.argsort(fitness[:pop_size])
    unique_starts = [best_params.copy()]
    for i in range(min(10, pop_size)):
        c = population[sorted_idx[i]]
        if all(np.linalg.norm(c - u) > 0.01 * np.linalg.norm(ranges) for u in unique_starts):
            unique_starts.append(c.copy())
        if len(unique_starts) >= 5:
            break

    restart_num = 0
    for ci, cand in enumerate(unique_starts):
        if elapsed() >= max_time * 0.94:
            break
        time_budget = max_time * min(0.94, shade_time + (0.94 - shade_time) * (ci + 1) / len(unique_starts))
        mean = cand.copy()
        sigma = 0.025 * (1 + ci * 0.4)
        lam = max(8, 3 * dim)
        no_improve = 0
        best_local = best
        while elapsed() < time_budget:
            zs = np.random.randn(lam, dim)
            samples = [clip(mean + sigma * ranges * zs[j]) for j in range(lam)]
            fits = []
            for s in samples:
                if elapsed() >= time_budget:
                    break
                fits.append(eval_func(s))
            if len(fits) < 3:
                break
            order = np.argsort(fits)
            mu = max(1, len(fits) // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            top = np.array([samples[order[j]] for j in range(mu)])
            new_mean = np.sum(weights[:, None] * top, axis=0)
            if fits[order[0]] < best_local:
                mean = new_mean
                best_local = fits[order[0]]
                sigma *= 1.05
                no_improve = 0
            else:
                sigma *= 0.76
                no_improve += 1
            if sigma < 1e-15 or no_improve > 18:
                restart_num += 1
                sigma = 0.012 * (1 + restart_num * 0.3)
                mean = best_params + np.random.randn(dim) * ranges * 0.008 * restart_num
                no_improve = 0
                best_local = best

    # --- Phase 4: Coordinate-wise + random fine-tuning ---
    scale = 0.002
    while elapsed() < max_time * 0.997:
        if np.random.random() < 0.3:
            d = np.random.randint(dim)
            trial = best_params.copy()
            trial[d] += np.random.randn() * ranges[d] * scale
            eval_func(clip(trial))
        else:
            trial = best_params + np.random.randn(dim) * ranges * scale
            f = eval_func(clip(trial))
        if f >= best if 'f' in dir() else True:
            scale *= 0.97
            if scale < 1e-16:
                scale = 0.004
    
    return best
