#Looking at the results, algorithm #3 (SHADE with coordinate descent) performed best at 8.53. I'll improve upon it with:
#
#1. Better Nelder-Mead simplex local search instead of coordinate descent
#2. CMA-ES-like covariance adaptation during local search
#3. More aggressive population reduction
#4. Better restart strategy with diverse starting points
#5. Weighted recombination of top solutions
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_solution = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    # --- Phase 1: LHS initial population ---
    pop_size_init = min(max(15 * dim, 40), 150)
    pop_size = pop_size_init
    min_pop_size = max(4, dim)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # --- Phase 2: L-SHADE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    h_idx = 0
    archive = []
    archive_max = pop_size_init
    
    stagnation_counter = 0
    last_best = best
    
    while elapsed() < max_time * 0.55:
        sorted_idx = np.argsort(fitness)
        
        S_F, S_CR, S_w = [], [], []
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        p_min = max(2.0 / pop_size, 0.05)
        p_max = 0.2
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(0, H)
            
            Fi = -1
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0) if M_CR[ri] >= 0 else 0.0
            
            p = np.random.uniform(p_min, p_max)
            p_num = max(2, int(np.ceil(p * pop_size)))
            pbest_idx = sorted_idx[np.random.randint(0, p_num)]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(candidates)
            
            combined_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            cross_points = np.random.random(dim) < CRi
            cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            below = trial < lower; above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2.0
            trial[above] = (upper[above] + population[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            trial_f = evaluate(trial)
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(abs(fitness[i] - trial_f))
                new_population[i] = trial; new_fitness[i] = trial_f
        
        population = new_population; fitness = new_fitness
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            M_F[h_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[h_idx] = np.sum(w * np.array(S_CR))
            h_idx = (h_idx + 1) % H
        
        ratio = elapsed() / (max_time * 0.55)
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        if new_ps < pop_size:
            si = np.argsort(fitness); population = population[si[:new_ps]]; fitness = fitness[si[:new_ps]]; pop_size = new_ps
        
        if best < last_best - 1e-12: stagnation_counter = 0; last_best = best
        else: stagnation_counter += 1
        if stagnation_counter > 20 + dim:
            si = np.argsort(fitness); keep = max(2, pop_size // 4)
            for idx in si[keep:]:
                if elapsed() >= max_time * 0.55: break
                population[idx] = lower + np.random.random(dim) * ranges; fitness[idx] = evaluate(population[idx])
            stagnation_counter = 0; M_F[:] = 0.5; M_CR[:] = 0.5
    
    # --- Phase 3: Nelder-Mead simplex ---
    if best_solution is None: return best
    
    def nelder_mead(x0, scale, time_frac):
        n = dim; simplex = np.empty((n+1, n)); fs = np.empty(n+1)
        simplex[0] = x0.copy(); fs[0] = evaluate(x0)
        for i in range(n):
            p = x0.copy(); p[i] += scale * ranges[i]; p = np.clip(p, lower, upper)
            simplex[i+1] = p; fs[i+1] = evaluate(p)
        while elapsed() < max_time * time_frac:
            order = np.argsort(fs); simplex = simplex[order]; fs = fs[order]
            centroid = simplex[:-1].mean(axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper); fr = evaluate(xr)
            if fr < fs[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper); fe = evaluate(xe)
                if fe < fr: simplex[-1]=xe; fs[-1]=fe
                else: simplex[-1]=xr; fs[-1]=fr
            elif fr < fs[-2]: simplex[-1]=xr; fs[-1]=fr
            else:
                xc = np.clip(centroid + 0.5*(simplex[-1]-centroid), lower, upper); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1]=xc; fs[-1]=fc
                else:
                    for i in range(1,n+1): simplex[i]=np.clip(simplex[0]+0.5*(simplex[i]-simplex[0]),lower,upper); fs[i]=evaluate(simplex[i])
    
    nelder_mead(best_solution, 0.05, 0.80)
    nelder_mead(best_solution, 0.01, 0.92)
    
    # Phase 4: Fine coordinate search
    current = best_solution.copy(); current_f = best; step = ranges * 0.01
    while elapsed() < max_time * 0.98:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.98: return best
            for s in [1, -1]:
                t = current.copy(); t[d] = np.clip(t[d]+s*step[d], lower[d], upper[d]); ft = evaluate(t)
                if ft < current_f: current=t; current_f=ft; improved=True; break
        if not improved:
            step *= 0.5
            if np.max(step/(ranges+1e-30)) < 1e-14: break
    return best
