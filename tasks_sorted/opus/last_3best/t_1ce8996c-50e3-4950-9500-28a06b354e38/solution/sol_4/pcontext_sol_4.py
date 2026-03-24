#Looking at the previous results, the best algorithm (score 16.66) uses JADE-style DE with coordinate descent. Let me improve it with:
#
#1. **SHADE memory** for F/CR adaptation
#2. **Nelder-Mead simplex** for local search (more efficient than coordinate descent in higher dims)
#3. **Opposition-based learning** for diversity
#4. **Pattern search** with acceleration step
#5. **Better time allocation**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: LHS initial sampling ---
    pop_size = min(max(12 * dim, 40), 120)
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: SHADE-like DE ---
    H = 5
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.9)
    k = 0
    
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    last_best = best
    
    while elapsed() < max_time * 0.60:
        S_F = []
        S_CR = []
        S_df = []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.60:
                break
            
            ri = np.random.randint(H)
            Fi = np.clip(M_F[ri] + 0.1 * np.random.standard_cauchy(), 0.01, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # p-best
            p = max(2, int(0.1 * pop_size))
            p_best_idx = np.random.randint(p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            a = np.random.choice(idxs)
            
            # Choose b from population + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            if i in combined:
                combined.remove(i)
            if a in combined:
                combined.remove(a)
            if len(combined) == 0:
                combined = [j for j in range(pop_size) if j != i and j != a]
            b_idx = np.random.choice(combined)
            
            if b_idx < pop_size:
                xb = population[b_idx]
            else:
                xb = archive[b_idx - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[a] - xb)
            
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.clip(np.where(cross, mutant, population[i]), lower, upper)
            f_trial = eval_f(trial)
            
            if f_trial < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(fitness[i] - f_trial)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = f_trial
        
        if S_F:
            weights = np.array(S_df)
            weights = weights / (weights.sum() + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % H
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if best < last_best - 1e-12:
            stagnation = 0; last_best = best
        else:
            stagnation += 1
        
        if stagnation > 12:
            stagnation = 0
            for i in range(pop_size * 2 // 3, pop_size):
                sc = ranges * 0.05 * (np.random.random() + 0.01)
                population[i] = np.clip(best_x + sc * np.random.randn(dim), lower, upper)
                fitness[i] = eval_f(population[i])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]; fitness = fitness[sorted_idx]
    
    # --- Phase 3: Nelder-Mead from top candidates ---
    for start_idx in range(min(3, pop_size)):
        if elapsed() >= max_time * 0.95:
            return best
        simplex = [population[start_idx].copy()]
        sf = [fitness[start_idx]]
        step = ranges * 0.02
        for j in range(dim):
            p = population[start_idx].copy(); p[j] += step[j]
            p = np.clip(p, lower, upper)
            simplex.append(p); sf.append(eval_f(p))
        simplex = [x for _, x in sorted(zip(sf, simplex), key=lambda t: t[0])]
        sf.sort()
        for _ in range(500 * dim):
            if elapsed() >= max_time * 0.95:
                return best
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
            fr = eval_f(xr)
            if fr < sf[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper)
                fe = eval_f(xe)
                if fe < fr: simplex[-1], sf[-1] = xe, fe
                else: simplex[-1], sf[-1] = xr, fr
            elif fr < sf[-2]:
                simplex[-1], sf[-1] = xr, fr
            else:
                xc = np.clip(centroid + 0.5*(simplex[-1] - centroid), lower, upper)
                fc = eval_f(xc)
                if fc < sf[-1]: simplex[-1], sf[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = np.clip(simplex[0] + 0.5*(simplex[i]-simplex[0]), lower, upper)
                        sf[i] = eval_f(simplex[i])
            order = np.argsort(sf)
            simplex = [simplex[i] for i in order]; sf = [sf[i] for i in order]
            if max(sf) - min(sf) < 1e-12: break
    
    # --- Phase 4: Fine coordinate descent ---
    if best_x is not None:
        cur = best_x.copy(); cur_f = best
        for scale in [0.005, 0.001, 0.0002, 0.00004]:
            step = ranges * scale
            improved = True
            while improved:
                improved = False
                for i in range(dim):
                    if elapsed() >= max_time * 0.995:
                        return best
                    for d in [1, -1]:
                        trial = cur.copy(); trial[i] += d * step[i]
                        trial = np.clip(trial, lower, upper)
                        ft = eval_f(trial)
                        if ft < cur_f: cur = trial; cur_f = ft; improved = True
    return best
