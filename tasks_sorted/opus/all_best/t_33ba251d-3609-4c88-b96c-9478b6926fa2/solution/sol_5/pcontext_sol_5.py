#Looking at the results, Algorithm 1 (SHADE + multi-start Nelder-Mead) performed best at 14.67. Let me analyze what can be improved:
#
#1. **Population size reduction** during SHADE to focus evaluations
#2. **L-SHADE** with linear population size reduction
#3. **Better local search** - combine Nelder-Mead with coordinate-wise golden section search
#4. **More diverse initialization** with opposition-based learning
#5. **Adaptive time split** between global and local phases
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

    # LHS initialization with opposition
    init_pop_size = min(max(25, 10 * dim), 200)
    population = np.zeros((init_pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(init_pop_size)
        for i in range(init_pop_size):
            population[i, j] = lower[j] + (perm[i] + np.random.rand()) / init_pop_size * ranges[j]
    
    fitness = np.zeros(init_pop_size)
    for i in range(init_pop_size):
        if elapsed() >= max_time * 0.90:
            break
        fitness[i] = eval_f(population[i])
    
    valid = min(init_pop_size, int(np.sum(elapsed() < max_time * 0.90 for _ in range(1))) + init_pop_size)
    # Just use all we evaluated
    pop_size = init_pop_size
    
    # Opposition-based population: evaluate opposites of worst half
    idx_sort = np.argsort(fitness)
    n_opp = min(pop_size // 4, 30)
    for i in range(n_opp):
        if elapsed() >= max_time * 0.85:
            break
        opp = lower + upper - population[idx_sort[-(i+1)]]
        f_opp = eval_f(opp)
        if f_opp < fitness[idx_sort[-(i+1)]]:
            population[idx_sort[-(i+1)]] = opp
            fitness[idx_sort[-(i+1)]] = f_opp

    # L-SHADE parameters
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size
    min_pop_size = max(4, dim // 2)
    init_pop = pop_size
    
    stagnation = 0
    prev_best = best
    gen = 0
    max_gen_est = max(50, int(max_time * 5))

    # Main L-SHADE loop
    while elapsed() < max_time * 0.75 and pop_size >= min_pop_size:
        gen += 1
        S_F, S_CR, S_df = [], [], []
        sorted_idx = np.argsort(fitness)
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 20:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0:
                Fi = 0.5
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            xpbest = population[sorted_idx[np.random.randint(p)]]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size - 1)
            if r2 >= i: r2 += 1
            if r2 == r1: r2 = (r2 + 1) % union_size
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (xpbest - population[i]) + Fi * (population[r1] - xr2)
            
            # Bounce-back boundary handling
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
            
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
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness = new_fit
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            sf_arr = np.array(S_F)
            M_F[mem_idx] = np.sum(w * sf_arr**2) / (np.sum(w * sf_arr) + 1e-30)
            M_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % H
        
        # Linear population size reduction
        new_size = max(min_pop_size, int(round(init_pop - (init_pop - min_pop_size) * elapsed() / (max_time * 0.75))))
        if new_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_size]]
            fitness = fitness[si[:new_size]]
            pop_size = new_size
        
        stagnation = stagnation + 1 if abs(best - prev_best) < 1e-14 else 0
        prev_best = best
        if stagnation > 15 + dim:
            half = pop_size // 2
            si = np.argsort(fitness)
            for i in range(half, pop_size):
                if elapsed() >= max_time * 0.75: break
                population[si[i]] = clip(best_x + 0.05 * ranges * np.random.randn(dim))
                fitness[si[i]] = eval_f(population[si[i]])
            stagnation = 0

    # Multi-start Nelder-Mead
    nm_restart = 0
    while elapsed() < max_time * 0.99 and best_x is not None:
        n = min(dim + 1, 60)
        simplex = np.zeros((n, dim)); sf = np.zeros(n)
        simplex[0] = best_x.copy(); sf[0] = best
        sc = ranges * max(1e-4, 0.03 * (0.5 ** nm_restart))
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
                xe = clip(c+2*(xr-c)); fe = eval_f(xe)
                if fe<fr: simplex[-1]=xe; sf[-1]=fe
                else: simplex[-1]=xr; sf[-1]=fr
            else:
                xc = clip(c+0.5*(simplex[-1]-c)); fc = eval_f(xc)
                if fc<sf[-1]: simplex[-1]=xc; sf[-1]=fc
                else:
                    for i in range(1,n): simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0])); sf[i]=eval_f(simplex[i])
            if abs(sf[0]-old_best_sf)<1e-15: no_improve+=1
            else: no_improve=0
            if no_improve > 5*dim: break
        nm_restart += 1
    
    return best
