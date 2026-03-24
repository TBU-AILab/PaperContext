#Looking at the results, the best algorithm (score 13.39) uses DE with adaptive parameters and local search. I'll improve it by combining SHADE (Success-History based Adaptive DE) with CMA-ES restarts and a more effective local search. Key improvements:
#
#1. **SHADE-style adaptation** for F and CR based on successful parameters
#2. **Better population archive** for diversity
#3. **Hybrid CMA-ES/DE** switching based on dimension
#4. **More efficient local search** with multiple strategies
#5. **Better time management**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # SHADE-based DE
    pop_size = min(max(8 * dim, 40), 200)
    
    # LHS initialization
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = lower[d] + (perm + np.random.rand(pop_size)) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(pop[i]) for i in range(pop_size)])
    if remaining() <= 0:
        return best
    
    # SHADE memory
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    # External archive
    archive = []
    max_archive = pop_size
    
    generation = 0
    
    while remaining() > max_time * 0.12:
        sorted_idx = np.argsort(fitness)
        
        S_F = []
        S_CR = []
        S_delta = []
        
        for i in range(pop_size):
            if remaining() <= max_time * 0.12:
                break
            
            # Select memory index
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            # Generate CR from Normal
            CRi = np.random.normal(M_CR[ri], 0.1)
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # current-to-pbest/1 mutation
            p = max(2, int(max(0.05, 0.2 - 0.15 * generation / (generation + 100)) * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(candidates)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2_idx = np.random.randint(0, pool_size)
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(0, pool_size)
            
            if r2_idx < pop_size:
                xr2 = pop[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial = clip(trial)
            f_trial = evaluate(trial)
            
            if f_trial < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(abs(fitness[i] - f_trial))
                if len(archive) < max_archive:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial
                fitness[i] = f_trial
            elif f_trial == fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
        
        if S_F:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        generation += 1
    
    # Local search: pattern search
    if best_x is not None:
        x = best_x.copy()
        fx = best
        step = ranges * 0.005
        while remaining() > 0.1:
            improved = False
            for d in range(dim):
                if remaining() <= 0.1:
                    break
                for s in [step[d], -step[d]]:
                    xn = x.copy(); xn[d] += s
                    fn = evaluate(clip(xn))
                    if fn < fx:
                        x = clip(xn); fx = fn; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-13:
                    break
    
    return best
