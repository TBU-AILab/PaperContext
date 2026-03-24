#Looking at the previous algorithm, I can identify several improvements:
#
#1. **Better SHADE implementation** with proper archive usage for difference vectors
#2. **Linear population size reduction (L-SHADE)** to focus search over time
#3. **CMA-ES local search** instead of just Nelder-Mead for better high-dimensional performance
#4. **Coordinate descent** as additional local search
#5. **Better time allocation** and more restarts
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_params, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # ---- Phase 1: LHS initialization ----
    pop_size_init = min(max(12 * dim, 50), 200)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])
    
    # ---- Phase 2: L-SHADE ----
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size_init
    
    stagnation = 0
    prev_best = best
    gen = 0
    max_gen_estimate = 300
    
    while elapsed() < max_time * 0.65:
        gen += 1
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        # p for pbest
        p_min = 2.0 / pop_size
        p_max = 0.2
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
                break
            
            ri = np.random.randint(memory_size)
            # Cauchy for F
            Fi = M_F[ri]
            while True:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # pbest
            p = np.random.uniform(p_min, p_max)
            p_count = max(2, int(np.ceil(p * pop_size)))
            sorted_indices = np.argsort(fitness)
            pbest_idx = np.random.choice(sorted_indices[:p_count])
            
            # Select r1 from population (not i)
            candidates = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(candidates)
            
            # Select r2 from population + archive (not i, not r1)
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(combined_size)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Bounce-back clipping
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2.0
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2.0
            
            trial = population[i].copy()
            jrand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == jrand)
            trial[mask] = mutant[mask]
            
            f_trial = evaluate(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
            elif f_trial == fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (np.sum(weights) + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
            M_CR[k] = np.sum(weights * scr)
            k = (k + 1) % memory_size
        
        # Linear population size reduction
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * elapsed() / (max_time * 0.65))))
        if new_pop_size < pop_size:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            keep = max(pop_size // 4, 2)
            order = np.argsort(fitness)
            for i in range(keep, pop_size):
                if elapsed() >= max_time * 0.65:
                    break
                # Reinit near best with some global exploration
                if np.random.random() < 0.5:
                    population[order[i]] = best_params + np.random.randn(dim) * 0.1 * ranges
                else:
                    population[order[i]] = lower + np.random.random(dim) * ranges
                population[order[i]] = np.clip(population[order[i]], lower, upper)
                fitness[order[i]] = evaluate(population[order[i]])
            stagnation = 0
    
    # ---- Phase 3: Multi-start Nelder-Mead ----
    def nelder_mead(x0, scale_factor, time_frac):
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += scale[i] * (1 if np.random.random() > 0.5 else -1)
        simplex = np.clip(simplex, lower, upper)
        
        fs = np.array([evaluate(simplex[i]) for i in range(n + 1) if elapsed() < max_time * time_frac])
        if len(fs) < n + 1:
            return
        
        for _ in range(10000):
            if elapsed() >= max_time * time_frac:
                break
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(2 * centroid - simplex[-1], lower, upper)
            fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr; continue
            if fr < fs[0]:
                xe = np.clip(centroid + 2 * (xr - centroid), lower, upper)
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe; fs[-1] = fe
                else:
                    simplex[-1] = xr; fs[-1] = fr
                continue
            if fr < fs[-1]:
                xc = np.clip(centroid + 0.5 * (xr - centroid), lower, upper)
                fc = evaluate(xc)
                if fc <= fr:
                    simplex[-1] = xc; fs[-1] = fc; continue
            else:
                xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lower, upper)
                fc = evaluate(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc; fs[-1] = fc; continue
            for i in range(1, n + 1):
                if elapsed() >= max_time * time_frac:
                    break
                simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lower, upper)
                fs[i] = evaluate(simplex[i])
            if np.max(np.abs(fs - fs[0])) < 1e-16:
                break
    
    # Run NM at multiple scales
    for sf in [0.1, 0.02, 0.005, 0.001]:
        if elapsed() >= max_time * 0.92 or best_params is None:
            break
        nelder_mead(best_params.copy(), sf, 0.65 + (0.92 - 0.65) * (1 - [0.1, 0.02, 0.005, 0.001].index(sf) / 4.0) + 0.07)
    
    # ---- Phase 4: Coordinate descent refinement ----
    if best_params is not None:
        x = best_params.copy()
        fx = best
        for step_scale in [0.01, 0.002, 0.0005]:
            if elapsed() >= max_time * 0.98:
                break
            step = step_scale * ranges
            improved = True
            while improved and elapsed() < max_time * 0.98:
                improved = False
                for d in range(dim):
                    if elapsed() >= max_time * 0.98:
                        break
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x[d] + direction * step[d], lower[d], upper[d])
                        f_new = evaluate(x_new)
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                            improved = True
                            break
    
    return best
