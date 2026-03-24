#Looking at the results, the best algorithm (score 2.15) uses SHADE DE + Hooke-Jeeves pattern search. Let me analyze what could be improved:
#
#1. **Better time allocation** - spend less time on DE, more on local search
#2. **Nelder-Mead simplex** as additional local search method
#3. **Better DE with population size reduction (L-SHADE)**
#4. **Smarter local search** - use conjugate directions / Powell's method ideas
#5. **More aggressive exploitation** of best solutions found
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
    
    # --- Phase 2: SHADE DE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.9)
    k = 0
    archive = []
    max_archive = pop_size
    stagnation = 0
    last_best = best
    
    de_time_frac = 0.45
    
    while elapsed() < max_time * de_time_frac:
        S_F, S_CR, S_df = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_frac:
                break
            
            ri = np.random.randint(H)
            Fi = np.clip(M_F[ri] + 0.1 * np.random.standard_cauchy(), 0.05, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            p_best_idx = np.random.randint(p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            a = np.random.choice(idxs)
            
            pool_size = pop_size + len(archive)
            candidates_b = [j for j in range(pool_size) if j != i and j != a]
            b_idx = np.random.choice(candidates_b) if candidates_b else a
            xb = population[b_idx] if b_idx < pop_size else archive[b_idx - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[a] - xb)
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.clip(np.where(cross, mutant, population[i]), lower, upper)
            f_trial = eval_f(trial)
            
            if f_trial < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi); S_df.append(fitness[i] - f_trial)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                population[i] = trial; fitness[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w = w / (w.sum() + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR)); k = (k + 1) % H
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]; fitness = fitness[sorted_idx]
        
        if best < last_best - 1e-12:
            stagnation = 0; last_best = best
        else:
            stagnation += 1
        if stagnation > 6:
            stagnation = 0
            for i in range(pop_size // 2, pop_size):
                sc = ranges * 0.1
                population[i] = np.clip(best_x + sc * np.random.randn(dim), lower, upper)
                fitness[i] = eval_f(population[i])
            sorted_idx = np.argsort(fitness); population = population[sorted_idx]; fitness = fitness[sorted_idx]
    
    # --- Phase 3: Powell-like conjugate direction search from top candidates ---
    def local_search(start_x, time_limit_frac):
        cur = start_x.copy(); cur_f = eval_f(cur)
        directions = np.eye(dim)
        
        for scale in [0.05, 0.01, 0.002, 0.0004, 0.00008, 0.000016]:
            step_sizes = ranges * scale
            origin = cur.copy()
            deltas = np.zeros(dim)
            for di in range(dim):
                if elapsed() >= max_time * time_limit_frac:
                    return cur, cur_f
                d = directions[di]
                best_step = 0; best_df = cur_f
                for sign in [1, -1]:
                    t = cur + sign * step_sizes.mean() * d
                    t = np.clip(t, lower, upper)
                    ft = eval_f(t)
                    if ft < best_df:
                        best_df = ft; best_step = sign
                if best_step != 0:
                    cur = cur + best_step * step_sizes.mean() * d
                    cur = np.clip(cur, lower, upper); cur_f = best_df
                    # Accelerate
                    while elapsed() < max_time * time_limit_frac:
                        t = cur + best_step * step_sizes.mean() * d
                        t = np.clip(t, lower, upper)
                        ft = eval_f(t)
                        if ft < cur_f: cur = t; cur_f = ft
                        else: break
                    deltas[di] = best_df
            
            # Coordinate descent pass
            improved = True
            while improved and elapsed() < max_time * time_limit_frac:
                improved = False
                for i in range(dim):
                    if elapsed() >= max_time * time_limit_frac: return cur, cur_f
                    for dd in [1, -1]:
                        trial = cur.copy(); trial[i] += dd * step_sizes[i]
                        trial = np.clip(trial, lower, upper)
                        ft = eval_f(trial)
                        if ft < cur_f:
                            cur = trial; cur_f = ft; improved = True
                            while elapsed() < max_time * time_limit_frac:
                                t2 = cur.copy(); t2[i] += dd * step_sizes[i]
                                t2 = np.clip(t2, lower, upper)
                                f2 = eval_f(t2)
                                if f2 < cur_f: cur = t2; cur_f = f2
                                else: break
                            break
        return cur, cur_f
    
    cands = [population[i].copy() for i in range(min(5, pop_size))]
    for ci, cand in enumerate(cands):
        tf = 0.93 - (0.93 - 0.55) * (len(cands) - 1 - ci) / max(len(cands) - 1, 1)
        local_search(cand, min(0.93, 0.55 + (ci + 1) * 0.08))
    
    # --- Phase 4: Fine perturbation ---
    if best_x is not None:
        while elapsed() < max_time * 0.998:
            s = ranges * 0.0003 * np.random.randn(dim)
            eval_f(best_x + s)
    
    return best
