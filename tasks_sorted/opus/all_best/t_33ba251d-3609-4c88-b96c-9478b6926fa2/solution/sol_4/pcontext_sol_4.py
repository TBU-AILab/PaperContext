#Looking at the results, the best approaches use DE + Nelder-Mead local search (~16.49). Let me improve with:
#
#1. **SHADE** with proper parameter adaptation
#2. **Better LHS initialization**
#3. **Multiple Nelder-Mead restarts** with shrinking scales
#4. **Vectorized crossover** for speed
#5. **Smarter time allocation** between global and local search
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
    
    evals = [0]
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # LHS initialization
    pop_size = min(max(20, 8 * dim), 150)
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

    # Main SHADE loop
    while elapsed() < max_time * 0.82:
        S_F, S_CR, S_df = [], [], []
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.82:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(0.15 * pop_size))
            xpbest = population[sorted_idx[np.random.randint(p)]]
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size - 1)
            if r2 >= i: r2 += 1
            if r2 == r1: r2 = (r2 + 1) % union_size
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (xpbest - population[i]) + Fi * (population[r1] - xr2)
            mutant = clip(mutant)
            
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
                population[i] = trial; fitness[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            M_F[mem_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % H
        
        stagnation = stagnation + 1 if abs(best - prev_best) < 1e-14 else 0
        prev_best = best
        if stagnation > 12 + dim:
            half = pop_size // 2
            si = np.argsort(fitness)
            for i in range(half, pop_size):
                if elapsed() >= max_time * 0.82: break
                population[si[i]] = clip(best_x + 0.05 * ranges * np.random.randn(dim))
                fitness[si[i]] = eval_f(population[si[i]])
            stagnation = 0

    # Multi-start Nelder-Mead local search
    nm_restart = 0
    while elapsed() < max_time * 0.99 and best_x is not None:
        n = min(dim + 1, 60)
        simplex = np.zeros((n, dim)); sf = np.zeros(n)
        simplex[0] = best_x.copy(); sf[0] = best
        sc = ranges * max(0.001, 0.02 * (0.5 ** nm_restart))
        for i in range(1, n):
            simplex[i] = clip(best_x + sc * np.random.randn(dim))
            sf[i] = eval_f(simplex[i])
        no_improve = 0
        while elapsed() < max_time * 0.99:
            idx = np.argsort(sf); simplex = simplex[idx]; sf = sf[idx]
            old_best_sf = sf[0]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_f(xr)
            if sf[0] <= fr < sf[-2]: simplex[-1]=xr; sf[-1]=fr
            elif fr < sf[0]:
                xe = clip(c + 2*(xr-c)); fe = eval_f(xe)
                if fe < fr: simplex[-1]=xe; sf[-1]=fe
                else: simplex[-1]=xr; sf[-1]=fr
            else:
                xc = clip(c+0.5*(simplex[-1]-c)); fc = eval_f(xc)
                if fc < sf[-1]: simplex[-1]=xc; sf[-1]=fc
                else:
                    for i in range(1,n): simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0])); sf[i]=eval_f(simplex[i])
            if abs(sf[0] - old_best_sf) < 1e-15: no_improve += 1
            else: no_improve = 0
            if no_improve > 5 * dim: break
        nm_restart += 1
    
    return best
