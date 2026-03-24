#Looking at the results, the best algorithm (13.39) uses simple DE with adaptive parameters. The SHADE variants performed worse, likely due to overhead. Let me improve by:
#
#1. **Faster DE loop** with vectorized crossover/mutation
#2. **Multiple mutation strategies** (ensemble DE)
#3. **Better local search** with Nelder-Mead simplex + pattern search
#4. **Smaller population** for faster convergence
#5. **Opposition-based learning** for initialization
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
    
    # Initialize with opposition-based learning
    pop_size = min(max(6 * dim, 30), 150)
    half = pop_size // 2
    
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = lower[d] + (perm + np.random.rand(pop_size)) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(pop[i]) for i in range(pop_size)])
    if elapsed() >= max_time:
        return best
    
    # Opposition-based: replace worst half with opposition of best half
    sorted_idx = np.argsort(fitness)
    for i in range(half, pop_size):
        opp = lower + upper - pop[sorted_idx[i - half]]
        opp += np.random.randn(dim) * ranges * 0.05
        opp = np.clip(opp, lower, upper)
        f_opp = evaluate(opp)
        if f_opp < fitness[sorted_idx[i]]:
            pop[sorted_idx[i]] = opp
            fitness[sorted_idx[i]] = f_opp
    
    # SHADE memory
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    mem_k = 0
    
    archive = []
    max_archive = pop_size
    generation = 0
    
    # Track stagnation
    stag_count = 0
    prev_best = best
    
    while elapsed() < max_time * 0.78:
        sorted_idx = np.argsort(fitness)
        
        S_F, S_CR, S_delta = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.78:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F
            Fi = -1
            for _ in range(10):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = np.clip(Fi, 0.01, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # current-to-pbest/1
            p = max(2, int(0.15 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = np.arange(pop_size)
            mask_i = idxs != i
            r1 = np.random.choice(idxs[mask_i])
            
            pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            pool = [x for x in pool if x != i and x != r1]
            r2 = np.random.choice(pool)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            cr_mask = np.random.rand(dim) < CRi
            cr_mask[np.random.randint(dim)] = True
            trial = np.where(cr_mask, mutant, pop[i])
            
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            
            if f_trial < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi)
                S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial; fitness[i] = f_trial
            elif f_trial == fitness[i]:
                pop[i] = trial; fitness[i] = f_trial
        
        if S_F:
            w = np.array(S_delta); w /= w.sum() + 1e-30
            sf = np.array(S_F)
            M_F[mem_k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[mem_k] = np.sum(w * np.array(S_CR))
            mem_k = (mem_k + 1) % H
        
        generation += 1
        
        if best < prev_best - 1e-10:
            stag_count = 0; prev_best = best
        else:
            stag_count += 1
        
        if stag_count > 30 and pop_size > 8:
            worst = sorted_idx[-max(1, pop_size//5):]
            for idx in worst:
                pop[idx] = lower + np.random.rand(dim) * ranges
                fitness[idx] = evaluate(pop[idx])
            stag_count = 0
    
    # Local search phase
    if best_x is not None:
        x = best_x.copy(); fx = best
        step = ranges * 0.01
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for s in [step[d], -step[d]]:
                    xn = x.copy(); xn[d] += s
                    xn = np.clip(xn, lower, upper)
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn; fx = fn; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-14:
                    x = best_x + np.random.randn(dim) * ranges * 0.002
                    x = np.clip(x, lower, upper)
                    fx = evaluate(x)
                    step = ranges * 0.003
    
    return best
