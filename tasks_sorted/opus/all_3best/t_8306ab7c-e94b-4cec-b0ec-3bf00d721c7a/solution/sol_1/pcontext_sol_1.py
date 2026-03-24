#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better DE strategy**: Use SHADE-like adaptive parameter control
#2. **Multiple restarts** with different strategies to avoid local optima
#3. **CMA-ES-inspired local search** instead of simple coordinate descent
#4. **Better time allocation** and more aggressive exploitation
#
#Here's my improved algorithm:
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

    # --- Phase 1: Diverse initialization with LHS + opposition ---
    pop_size = min(max(30, 8 * dim), 300)
    
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

    # --- Phase 2: SHADE-like Differential Evolution ---
    memory_size = 5
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    p_min = max(2, int(0.05 * pop_size))
    
    while elapsed() < max_time * 0.65:
        S_F, S_CR, S_delta = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
                break
            
            ri = np.random.randint(memory_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.5)
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
            
            # DE/current-to-pbest/1 with archive
            p = max(2, int(0.1 * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest = sorted_idx[np.random.randint(0, p)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            combined = list(range(pop_size)) + list(range(len(archive)))
            combined = [j for j in combined if j != i and j != r1]
            r2_idx = np.random.choice(len(combined))
            r2_val = combined[r2_idx]
            if r2_val < pop_size:
                xr2 = population[r2_val]
            else:
                xr2 = archive[r2_val - pop_size]
            
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Midpoint reflection for out-of-bounds
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            trial = clip(trial)
            
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

    # --- Phase 3: CMA-ES-like local search around best ---
    sigma = 0.1
    mean = best_params.copy()
    mean_fit = best
    lam = max(6, 2 * dim)
    
    while elapsed() < max_time * 0.95:
        samples = []
        fits = []
        for _ in range(lam):
            if elapsed() >= max_time * 0.95:
                break
            z = mean + sigma * ranges * np.random.randn(dim)
            z = clip(z)
            f = eval_func(z)
            samples.append(z)
            fits.append(f)
        
        if not samples:
            break
        
        order = np.argsort(fits)
        mu = max(1, lam // 2)
        top = np.array([samples[order[j]] for j in range(mu)])
        new_mean = np.mean(top, axis=0)
        
        if fits[order[0]] < mean_fit:
            mean = new_mean
            mean_fit = fits[order[0]]
            sigma *= 1.0
        else:
            sigma *= 0.85
            if sigma < 1e-14:
                mean = best_params.copy()
                sigma = 0.05
    
    # --- Phase 4: Final fine-tuning ---
    while elapsed() < max_time * 0.99:
        trial = best_params + np.random.randn(dim) * ranges * 0.001
        eval_func(clip(trial))
    
    return best
