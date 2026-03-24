#Looking at the previous results, the simpler DE + Nelder-Mead approach (24.2) outperformed the more complex SHADE + CMA-ES (59.7). I'll build on the winning approach with key improvements:
#
#1. **Better SHADE-style adaptive parameters** with success history
#2. **More aggressive time usage** - the DE phase cut off at 70% leaving time wasted
#3. **Multiple local search restarts** from top solutions
#4. **Opposition-based learning** for diversity
#5. **Better population sizing and restart logic**
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
    
    def time_left():
        return max_time - elapsed()
    
    evals = [0]
    
    def eval_f(x):
        x = np.clip(x, lower, upper)
        nonlocal best, best_x
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- LHS initialization ---
    pop_size = min(max(30, 8 * dim), 250)
    
    def lhs_population(n):
        pop = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            pop[:, d] = (perm + np.random.rand(n)) / n
        return lower + pop * ranges
    
    population = lhs_population(pop_size)
    fitness = np.array([eval_f(population[i]) for i in range(pop_size)])
    
    # SHADE memory
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    mem_k = 0
    
    archive = []
    archive_max = pop_size
    
    stagnation = 0
    prev_best = best
    
    # --- Main SHADE loop ---
    while time_left() > max_time * 0.25:
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        S_F, S_CR, S_delta = [], [], []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_left() < max_time * 0.25:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 1.0:
                    Fi = 1.0
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1 with archive
            p = max(2.0 / pop_size, np.random.uniform(0.05, 0.20))
            n_pbest = max(1, int(p * pop_size))
            pbest_idx = np.random.randint(0, n_pbest)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(0, pool_size - 1)
            if r2 >= i:
                r2 += 1
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Bounce-back bounds
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                delta = fitness[i] - f_trial
                if delta > 0:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta + 1e-30)
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update SHADE memory
        if S_F:
            w = np.array(S_delta)
            w /= w.sum()
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[mem_k] = np.sum(w * scr)
            mem_k = (mem_k + 1) % H
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 8:
            n_replace = pop_size // 2
            worst = np.argsort(fitness)[-n_replace:]
            for j in worst:
                # Opposition-based or random
                if np.random.rand() < 0.5 and best_x is not None:
                    population[j] = lower + upper - best_x + 0.1 * ranges * np.random.randn(dim)
                else:
                    population[j] = lower + np.random.rand(dim) * ranges
                population[j] = np.clip(population[j], lower, upper)
                fitness[j] = eval_f(population[j])
            stagnation = 0
    
    # --- Local search: Nelder-Mead from top solutions ---
    sort_idx = np.argsort(fitness)
    n_local = min(3, pop_size)
    
    for li in range(n_local):
        if time_left() < max_time * 0.03:
            break
        x0 = population[sort_idx[li]].copy()
        n_s = dim + 1
        simplex = np.zeros((n_s, dim))
        simplex[0] = x0
        sf = np.zeros(n_s)
        sf[0] = fitness[sort_idx[li]]
        scale = 0.02 * ranges * (1 + li)
        for j in range(1, n_s):
            simplex[j] = x0.copy()
            simplex[j][(j-1) % dim] += scale[(j-1) % dim] * (1 if np.random.rand() > 0.5 else -1)
            simplex[j] = np.clip(simplex[j], lower, upper)
            sf[j] = eval_f(simplex[j])
        
        while time_left() > max_time * 0.02:
            idx = np.argsort(sf)
            simplex, sf = simplex[idx], sf[idx]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
            fr = eval_f(xr)
            if fr < sf[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper)
                fe = eval_f(xe)
                simplex[-1], sf[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < sf[-2]:
                simplex[-1], sf[-1] = xr, fr
            else:
                xc = np.clip(centroid + 0.5*(simplex[-1] - centroid) if fr >= sf[-1] else centroid + 0.5*(xr - centroid), lower, upper)
                fc = eval_f(xc)
                if fc < sf[-1]:
                    simplex[-1], sf[-1] = xc, fc
                else:
                    for j in range(1, n_s):
                        simplex[j] = simplex[0] + 0.5*(simplex[j] - simplex[0])
                        simplex[j] = np.clip(simplex[j], lower, upper)
                        sf[j] = eval_f(simplex[j])
    
    return best
