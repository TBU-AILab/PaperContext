#Looking at the results, algorithm 1 (score 16.66) is the best. Algorithm 3 (score 30.71) degraded likely because the acceleration in coordinate descent with `break` after finding improvement skips checking the other direction, and running 5 candidates is too many. Let me combine the best elements:
#
#1. **SHADE DE** (proven effective)
#2. **Coordinate descent from fewer candidates** (3 not 5) without the problematic break
#3. **Linear population size reduction** (L-SHADE style)
#4. **Better time management**
#5. **Gaussian perturbation local search** in remaining time
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
    pop_size_init = min(max(15 * dim, 50), 150)
    pop_size = pop_size_init
    min_pop = max(4, dim // 2)
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
    
    # --- Phase 2: L-SHADE DE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.9)
    k = 0
    archive = []
    max_archive = pop_size_init
    stagnation = 0
    last_best = best
    total_evals_de = 0
    max_evals_estimate = pop_size_init * 80
    
    while elapsed() < max_time * 0.65:
        S_F, S_CR, S_df = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
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
            b_raw = np.random.randint(pool_size - 2)
            candidates_b = [j for j in range(pool_size) if j != i and j != a]
            b_idx = candidates_b[b_raw % len(candidates_b)] if candidates_b else a
            xb = population[b_idx] if b_idx < pop_size else archive[b_idx - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[a] - xb)
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.clip(np.where(cross, mutant, population[i]), lower, upper)
            f_trial = eval_f(trial)
            total_evals_de += 1
            
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
        
        # L-SHADE population reduction
        new_pop_size = max(min_pop, int(round(pop_size_init - (pop_size_init - min_pop) * total_evals_de / max_evals_estimate)))
        if new_pop_size < pop_size:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        else:
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]; fitness = fitness[sorted_idx]
        
        if best < last_best - 1e-12:
            stagnation = 0; last_best = best
        else:
            stagnation += 1
        if stagnation > 10:
            stagnation = 0
            n_replace = min(pop_size // 2, pop_size - 1)
            for i in range(pop_size - n_replace, pop_size):
                sc = ranges * 0.06
                population[i] = np.clip(best_x + sc * np.random.randn(dim), lower, upper)
                fitness[i] = eval_f(population[i])
            sorted_idx = np.argsort(fitness); population = population[sorted_idx]; fitness = fitness[sorted_idx]
    
    # --- Phase 3: Multi-scale coordinate descent from top 3 ---
    candidates = [population[i].copy() for i in range(min(3, pop_size))]
    for cand in candidates:
        cur = cand.copy(); cur_f = eval_f(cur)
        for scale in [0.05, 0.01, 0.002, 0.0004, 0.00008]:
            step = ranges * scale
            improved = True
            while improved:
                improved = False
                for i in range(dim):
                    if elapsed() >= max_time * 0.98:
                        return best
                    for d in [1, -1]:
                        trial = cur.copy(); trial[i] += d * step[i]
                        trial = np.clip(trial, lower, upper)
                        ft = eval_f(trial)
                        if ft < cur_f:
                            cur = trial; cur_f = ft; improved = True
    
    # --- Phase 4: Gaussian perturbation ---
    if best_x is not None:
        while elapsed() < max_time * 0.995:
            sigma = ranges * 0.001 * np.random.random()
            trial = np.clip(best_x + sigma * np.random.randn(dim), lower, upper)
            eval_f(trial)
    
    return best
