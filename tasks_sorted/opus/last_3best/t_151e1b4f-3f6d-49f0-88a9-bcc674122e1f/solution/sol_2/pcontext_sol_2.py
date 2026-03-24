#Looking at the results, the DE + Nelder-Mead approach (algorithm 1) significantly outperformed the CMA-ES approach (algorithm 2). Let me build on the winning strategy with improvements:
#
#1. **Better initialization** (Latin Hypercube Sampling)
#2. **Improved DE** with multiple mutation strategies (DE/rand/1, DE/best/1, DE/current-to-best/1) selected adaptively
#3. **More aggressive local search** with multiple Nelder-Mead restarts
#4. **Population diversity management**
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
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS initialization ---
    pop_size = min(max(20, 10 * dim), 200)
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.array([ev(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.1])
    if len(fitness) < pop_size:
        fitness = np.append(fitness, [float('inf')] * (pop_size - len(fitness)))

    # --- Phase 2: Adaptive DE with jDE-like self-adaptation ---
    F_arr = np.full(pop_size, 0.5)
    CR_arr = np.full(pop_size, 0.9)
    stagnation = 0
    prev_best = best

    while elapsed() < max_time * 0.75:
        sorted_idx = np.argsort(fitness)
        bi = sorted_idx[0]
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            # Self-adaptive F and CR
            Fi = F_arr[i] if np.random.random() > 0.1 else 0.1 + 0.9 * np.random.random()
            CRi = CR_arr[i] if np.random.random() > 0.1 else np.random.random()
            
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            # Strategy selection
            strat = np.random.random()
            if strat < 0.4:
                mutant = population[bi] + Fi * (population[r1] - population[r2])
            elif strat < 0.7:
                mutant = population[i] + Fi * (population[bi] - population[i]) + Fi * (population[r1] - population[r2])
            else:
                mutant = population[r1] + Fi * (population[r2] - population[r3])
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial_f = ev(trial)
            if trial_f <= fitness[i]:
                population[i] = clip(trial)
                fitness[i] = trial_f
                F_arr[i] = Fi
                CR_arr[i] = CRi
        
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            half = pop_size // 2
            for i in range(half, pop_size):
                idx = sorted_idx[i]
                population[idx] = lower + np.random.random(dim) * ranges
                if elapsed() >= max_time * 0.75: break
                fitness[idx] = ev(population[idx])
            stagnation = 0

    # --- Phase 3: Multiple Nelder-Mead restarts from top solutions ---
    top_k = min(5, pop_size)
    top_idx = np.argsort(fitness)[:top_k]
    for ti in top_idx:
        if elapsed() >= max_time * 0.97: break
        x0 = population[ti].copy()
        n = dim
        scale = 0.02 * ranges
        simplex = np.vstack([x0, x0 + np.diag(scale)])
        fs = np.array([ev(simplex[i]) for i in range(n+1) if elapsed()<max_time*0.97])
        if len(fs)<n+1: break
        while elapsed()<max_time*0.97:
            o=np.argsort(fs); simplex=simplex[o]; fs=fs[o]
            c=np.mean(simplex[:-1],axis=0)
            xr=clip(c+1.0*(c-simplex[-1])); fr=ev(xr)
            if fs[0]<=fr<fs[-2]: simplex[-1]=xr;fs[-1]=fr
            elif fr<fs[0]:
                xe=clip(c+2.0*(xr-c));fe=ev(xe)
                simplex[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
            else:
                xc=clip(c+0.5*(simplex[-1]-c));fc=ev(xc)
                if fc<fs[-1]: simplex[-1]=xc;fs[-1]=fc
                else:
                    for i in range(1,n+1):
                        if elapsed()>=max_time*0.97: return best
                        simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0]));fs[i]=ev(simplex[i])
    return best
