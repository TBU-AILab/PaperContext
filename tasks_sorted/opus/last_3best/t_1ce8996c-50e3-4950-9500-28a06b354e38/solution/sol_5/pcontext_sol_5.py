#Looking at the results, algorithm 1 (score 16.66) performs best with DE/current-to-best/1 and multi-scale coordinate descent. Algorithm 2 (score 16.95) with SHADE+Nelder-Mead is slightly worse - likely the Nelder-Mead in high dims is expensive. Let me focus on improving the winning approach with:
#
#1. **SHADE adaptation** (proven better than simple per-individual)
#2. **More aggressive coordinate descent with acceleration (pattern search)**
#3. **Better time allocation** - less DE time, more local search
#4. **Gaussian local search** between CD passes
#5. **Tighter restart mechanism**
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
    pop_size = min(max(10 * dim, 40), 100)
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
    
    while elapsed() < max_time * 0.55:
        S_F, S_CR, S_df = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(H)
            Fi = np.clip(M_F[ri] + 0.1 * np.random.standard_cauchy(), 0.05, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(0.15 * pop_size))
            p_best_idx = np.random.randint(p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            a = np.random.choice(idxs)
            
            pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            pool = [x for x in pool if x != i and x != a]
            if not pool:
                pool = [j for j in range(pop_size) if j != i and j != a]
            b_idx = np.random.choice(pool) if pool else a
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
        if stagnation > 8:
            stagnation = 0
            for i in range(pop_size // 2, pop_size):
                sc = ranges * 0.08 * (np.random.random() + 0.01)
                population[i] = np.clip(best_x + sc * np.random.randn(dim), lower, upper)
                fitness[i] = eval_f(population[i])
            sorted_idx = np.argsort(fitness); population = population[sorted_idx]; fitness = fitness[sorted_idx]
    
    # --- Phase 3: Multi-start pattern search with acceleration ---
    candidates = [population[i].copy() for i in range(min(5, pop_size))]
    for cand in candidates:
        cur = cand.copy(); cur_f = eval_f(cur)
        for scale in [0.05, 0.01, 0.002, 0.0004, 0.00008]:
            step = ranges * scale
            improved = True
            while improved:
                improved = False
                for i in range(dim):
                    if elapsed() >= max_time * 0.99:
                        return best
                    for d in [1, -1]:
                        trial = cur.copy(); trial[i] += d * step[i]
                        trial = np.clip(trial, lower, upper)
                        ft = eval_f(trial)
                        if ft < cur_f:
                            cur = trial; cur_f = ft; improved = True
                            # Acceleration: keep going in same direction
                            while elapsed() < max_time * 0.99:
                                trial2 = cur.copy(); trial2[i] += d * step[i]
                                trial2 = np.clip(trial2, lower, upper)
                                ft2 = eval_f(trial2)
                                if ft2 < cur_f:
                                    cur = trial2; cur_f = ft2
                                else:
                                    break
                            break
    
    return best
