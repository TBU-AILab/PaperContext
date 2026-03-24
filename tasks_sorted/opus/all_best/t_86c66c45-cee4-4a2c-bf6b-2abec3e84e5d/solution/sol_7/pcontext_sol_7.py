#Looking at the results, the pure SHADE (7.46) was the best. The multi-phase approaches with restarts performed worse, likely due to overhead and poor time allocation. Let me improve by:
#
#1. **L-SHADE with proper population reduction** in a single continuous loop
#2. **Larger initial population** for better exploration
#3. **Better stagnation handling** - opposition-based reinitialization + local perturbation
#4. **Nelder-Mead polishing** at the end using remaining time
#5. **Tuned parameters** based on what worked
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
    
    def time_left():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # L-SHADE parameters
    N_init = min(max(40, 10 * dim), 200)
    N_min = 4
    pop_size = N_init
    H = 100
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    mk = 0
    
    # Initialize with LHS
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(ind) for ind in population])
    archive = []
    total_evals = pop_size
    max_evals = N_init * 500
    
    stagnation = 0
    prev_best = best
    restart_count = 0
    
    while time_left() > 0.1:
        S_F, S_CR, delta_f = [], [], []
        sorted_idx = np.argsort(fitness)
        
        ratio = min(1.0, total_evals / max(max_evals, 1))
        p_rate = max(2.0 / pop_size, 0.2 - 0.15 * ratio)
        p_best_size = max(2, int(p_rate * pop_size))
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_left() <= 0.1:
                break
            
            ri = np.random.randint(H)
            Fi = memory_F[ri] + 0.1 * np.random.standard_cauchy()
            while Fi <= 0:
                Fi = memory_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            CRi = np.clip(memory_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            pi = sorted_idx[np.random.randint(p_best_size)]
            r1 = np.random.randint(pop_size)
            while r1 == i: r1 = np.random.randint(pop_size)
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1: r2 = np.random.randint(pool_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - x_r2)
            jrand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi; mask[jrand] = True
            trial = np.where(mask, mutant, population[i])
            ol = trial < lower; oh = trial > upper
            trial[ol] = (lower[ol] + population[i][ol]) / 2
            trial[oh] = (upper[oh] + population[i][oh]) / 2
            trial = clip(trial)
            trial_f = evaluate(trial); total_evals += 1
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_f))
                    archive.append(population[i].copy())
                new_pop[i] = trial; new_fit[i] = trial_f
        
        population = new_pop; fitness = new_fit
        while len(archive) > pop_size: archive.pop(np.random.randint(len(archive)))
        if S_F:
            w = np.array(delta_f); w /= w.sum()
            memory_F[mk % H] = np.sum(w * np.array(S_F)**2) / max(np.sum(w * np.array(S_F)), 1e-30)
            memory_CR[mk % H] = np.sum(w * np.array(S_CR)); mk += 1
        
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * total_evals / max_evals)))
        if new_size < pop_size:
            si = np.argsort(fitness)[:new_size]
            population = population[si]; fitness = fitness[si]; pop_size = new_size
        
        if abs(prev_best - best) < 1e-14: stagnation += 1
        else: stagnation = 0
        prev_best = best
        
        if stagnation > 30 and time_left() > 1.0:
            stagnation = 0; restart_count += 1; archive.clear()
            pop_size = min(N_init, max(20, 6 * dim))
            total_evals = 0; max_evals = pop_size * 500
            memory_F[:] = 0.5; memory_CR[:] = 0.5; mk = 0
            population = np.zeros((pop_size, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size)
                for ii in range(pop_size):
                    population[ii, d] = lower[d] + (perm[ii] + np.random.rand()) / pop_size * ranges[d]
            if best_params is not None:
                population[0] = best_params.copy()
                n_local = min(pop_size // 3, 10)
                for ii in range(1, n_local + 1):
                    r = max(0.01, 0.3 * time_left() / max_time)
                    population[ii] = best_params + r * ranges * (np.random.rand(dim) - 0.5)
                    population[ii] = clip(population[ii])
            fitness = np.array([evaluate(ind) for ind in population])
            total_evals += pop_size
    
    return best
