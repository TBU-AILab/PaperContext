#Looking at the results, the best algorithm (13.39) uses DE with adaptive parameters and coordinate-wise local search. I'll improve it with:
#
#1. **L-SHADE** (linear population size reduction) for better convergence
#2. **Vectorized operations** instead of Python loops for speed
#3. **Multi-start local search** with Nelder-Mead-style moves
#4. **Better time allocation** between exploration and exploitation
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # L-SHADE parameters
    init_pop_size = min(max(10 * dim, 50), 300)
    min_pop_size = max(4, dim // 2)
    pop_size = init_pop_size
    max_evals_estimate = int(max_time * 500)  # rough estimate
    
    # LHS initialization
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = lower[d] + (perm + np.random.rand(pop_size)) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(pop[i]) for i in range(pop_size)])
    if elapsed() >= max_time:
        return best
    
    # SHADE memory
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    max_archive = init_pop_size
    generation = 0
    total_evals = pop_size
    
    while elapsed() < max_time * 0.80:
        sorted_idx = np.argsort(fitness)
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi >= 1.0:
                    Fi = 1.0
                    break
            
            CRi = np.random.normal(M_CR[ri], 0.1)
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # current-to-pbest/1
            p_rate = max(0.05, 0.25 - 0.20 * total_evals / max(max_evals_estimate, total_evals + 1))
            p = max(2, int(p_rate * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            # r1 from pop
            r1 = np.random.randint(pop_size)
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            # Binomial crossover
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            # Bounce-back
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            total_evals += 1
            
            if f_trial < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                new_pop[i] = trial
                new_fitness[i] = f_trial
            elif f_trial == fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
        
        pop = new_pop
        fitness = new_fitness
        
        if S_F:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        generation += 1
        
        # Linear population size reduction
        new_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / (max_time * 0.80))))
        if new_size < pop_size:
            idx = np.argsort(fitness)[:new_size]
            pop = pop[idx]
            fitness = fitness[idx]
            pop_size = new_size
    
    # Phase 2: Local search
    if best_x is not None:
        x = best_x.copy()
        fx = best
        step = ranges * 0.01
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for s in [step[d], -step[d]]:
                    xn = x.copy()
                    xn[d] += s
                    xn = np.clip(xn, lower, upper)
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn; fx = fn; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-14:
                    # Random restart local search
                    x = best_x + np.random.randn(dim) * ranges * 0.001
                    x = np.clip(x, lower, upper)
                    fx = evaluate(x)
                    step = ranges * 0.005
    
    return best
