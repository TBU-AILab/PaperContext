#Looking at the results, the DE+Nelder-Mead hybrid (16.49) was the best. Let me improve it with:
#
#1. **SHADE** (Success-History Adaptive DE) for better parameter adaptation
#2. **Proper Latin Hypercube Sampling** initialization
#3. **Current-to-pbest/1** mutation strategy
#4. **More aggressive local search** with multiple Nelder-Mead restarts
#5. **Vectorized crossover** and better time allocation
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # LHS initialization
    pop_size = min(max(20, 8 * dim), 200)
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, j] = lower[j] + (perm[i] + np.random.rand()) / pop_size * ranges[j]
    
    fitness = np.array([eval_f(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
        pop_size = len(fitness)

    # SHADE parameters
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size

    stagnation = 0
    prev_best = best

    while elapsed() < max_time * 0.85:
        S_F, S_CR, S_df = [], [], []
        
        # Sort for pbest selection
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            ri = np.random.randint(H)
            Fi = min(1.0, max(0.01, M_F[ri] + 0.1 * np.random.standard_cauchy()))
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = max(2, int(0.15 * pop_size))
            xpbest = population[sorted_idx[np.random.randint(p)]]
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size - 1)
            if r2 >= i: r2 += 1
            if r2 == r1: r2 = (r2 + 1) % union_size
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = clip(population[i] + Fi * (xpbest - population[i]) + Fi * (population[r1] - xr2))
            
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, population[i])
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_df.append(fitness[i] - f_trial)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            M_F[mem_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % H
        
        stagnation = stagnation + 1 if abs(best - prev_best) < 1e-14 else 0
        prev_best = best
        
        if stagnation > 15 + dim:
            half = pop_size // 2
            idx_s = np.argsort(fitness)
            for i in range(half, pop_size):
                if elapsed() >= max_time * 0.85: break
                population[idx_s[i]] = clip(best_x + 0.05 * ranges * np.random.randn(dim))
                fitness[idx_s[i]] = eval_f(population[idx_s[i]])
            stagnation = 0

    # Nelder-Mead local search
    while elapsed() < max_time * 0.98 and best_x is not None:
        n = min(dim + 1, 50)
        simplex = np.zeros((n, dim)); sf = np.zeros(n)
        simplex[0] = best_x.copy(); sf[0] = best
        sc = 0.005 * ranges * (1 + np.random.rand())
        for i in range(1, n):
            simplex[i] = clip(best_x + sc * np.random.randn(dim))
            sf[i] = eval_f(simplex[i])
        while elapsed() < max_time * 0.98:
            idx = np.argsort(sf); simplex = simplex[idx]; sf = sf[idx]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_f(xr)
            if sf[0] <= fr < sf[-2]: simplex[-1]=xr; sf[-1]=fr
            elif fr < sf[0]:
                xe = clip(c + 2*(xr-c)); fe = eval_f(xe)
                if fe<fr: simplex[-1]=xe; sf[-1]=fe
                else: simplex[-1]=xr; sf[-1]=fr
            else:
                xc = clip(c+0.5*(simplex[-1]-c)); fc = eval_f(xc)
                if fc<sf[-1]: simplex[-1]=xc; sf[-1]=fc
                else:
                    for i in range(1,n): simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0])); sf[i]=eval_f(simplex[i])
    
    return best
