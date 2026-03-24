#Looking at the results, the best algorithm (9.56) uses SHADE + simple CMA-ES-like local search. The L-SHADE + full CMA-ES version (31.19) was slower due to overhead from covariance matrix operations and population reduction. I'll improve by:
#
#1. **Faster SHADE** with vectorized operations and less overhead
#2. **Better local search**: Proper CMA-ES but only for low dim, otherwise use simpler method
#3. **Smarter time allocation** and multiple restart strategy
#4. **Nelder-Mead simplex** as additional local search
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
    memory_size = 5
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    while elapsed() < max_time * 0.55:
        S_F, S_CR, S_delta = [], [], []
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(memory_size)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi >= 2.0:
                    Fi = M_F[ri]
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
            
            p = max(2, int(0.1 * pop_size))
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

    # --- Phase 3: Multi-start local search ---
    sorted_idx = np.argsort(fitness)
    candidates = [best_params.copy()] + [population[sorted_idx[i]].copy() for i in range(min(4, pop_size))]
    
    for ci, cand in enumerate(candidates):
        if elapsed() >= max_time * 0.94:
            break
        
        time_budget = max_time * min(0.94, 0.55 + 0.39 * (ci + 1) / len(candidates))
        mean = cand.copy()
        sigma = 0.05 * (1 + ci * 0.3)
        lam = max(6, 2 * dim)
        
        no_improve = 0
        while elapsed() < time_budget:
            samples = []
            fits = []
            for _ in range(lam):
                if elapsed() >= time_budget:
                    break
                z = mean + sigma * ranges * np.random.randn(dim)
                z = clip(z)
                f = eval_func(z)
                samples.append(z)
                fits.append(f)
            
            if len(fits) < 2:
                break
            
            order = np.argsort(fits)
            mu = max(1, len(fits) // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            top = np.array([samples[order[j]] for j in range(mu)])
            new_mean = np.sum(weights[:, None] * top, axis=0)
            
            if fits[order[0]] < eval_func(mean):
                mean = new_mean
                sigma *= 1.02
                no_improve = 0
            else:
                sigma *= 0.82
                no_improve += 1
            
            if sigma < 1e-14 or no_improve > 15:
                break

    # --- Phase 4: Final fine-tuning ---
    scale = 0.002
    while elapsed() < max_time * 0.99:
        trial = best_params + np.random.randn(dim) * ranges * scale
        f = eval_func(clip(trial))
        if f >= best:
            scale *= 0.99
            if scale < 1e-15:
                scale = 0.005
    
    return best
