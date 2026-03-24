#Looking at the results, algorithm #7 (score 11.83) is the best. I'll analyze what worked and make targeted improvements:
#
#1. **L-SHADE population reduction** (time-based) was beneficial in algorithms 1-3
#2. **Bounce-back boundary** in mutation was better than simple clipping
#3. **Pattern moves** in coordinate descent help exploit momentum
#4. **Faster r1/r2 selection** avoiding expensive list operations
#5. **Multiple local search scales** and random direction escape
#6. **Weighted Lehmer mean for CR** (not arithmetic mean)
#
#Key changes from #7: Add L-SHADE reduction, bounce-back boundaries, pattern moves, faster index selection, better CR update, and smarter local search restarts.
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
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

    # --- Phase 1: L-SHADE ---
    N_init = max(20, min(80, dim * 5))
    N_min = 4
    pop_size = N_init
    H = 20

    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            pop[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
    
    fit = np.array([eval_f(p) for p in pop])
    
    opp_pop = lower + upper - pop
    opp_fit = np.array([eval_f(p) for p in opp_pop])
    
    all_pop = np.vstack([pop, opp_pop])
    all_fit = np.concatenate([fit, opp_fit])
    idx_sort = np.argsort(all_fit)[:pop_size]
    pop = all_pop[idx_sort].copy()
    fit = all_fit[idx_sort].copy()
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = N_init
    
    generation = 0
    stagnation = 0
    prev_best = best
    de_start = elapsed()

    while remaining() > max_time * 0.25:
        if remaining() <= 0:
            return best
        
        S_F, S_CR, delta_f = [], [], []
        sorted_idx = np.argsort(fit)
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            ri = np.random.randint(0, H)
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0: break
            else: Fi = 0.5
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = np.random.uniform(max(2.0/pop_size, 0.05), 0.2)
            n_pbest = max(1, int(p * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, n_pbest)]
            
            r1 = np.random.randint(0, pop_size - 1)
            if r1 >= i: r1 += 1
            
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pool_size)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            for dd in range(dim):
                if mutant[dd] < lower[dd]: mutant[dd] = (lower[dd] + pop[i][dd]) / 2.0
                elif mutant[dd] > upper[dd]: mutant[dd] = (upper[dd] + pop[i][dd]) / 2.0
            
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi; mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            trial_fit = eval_f(trial)
            
            if trial_fit <= fit[i]:
                if trial_fit < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); delta_f.append(abs(fit[i] - trial_fit))
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max: archive.pop(np.random.randint(len(archive)))
                pop[i] = trial; fit[i] = trial_fit
        
        if S_F:
            w = np.array(delta_f); w /= w.sum() + 1e-30
            sf, scr = np.array(S_F), np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr**2) / (np.sum(w * scr) + 1e-30)
            k = (k+1) % H
        
        generation += 1
        if best < prev_best - 1e-10: stagnation = 0; prev_best = best
        else: stagnation += 1
        if stagnation > 25:
            order = np.argsort(fit)
            for idx in order[pop_size//3:]:
                pop[idx] = best_x + np.random.randn(dim) * ranges * 0.1; pop[idx] = clip(pop[idx]); fit[idx] = eval_f(pop[idx])
            stagnation = 0
        
        ratio = min(1.0, (elapsed() - de_start) / (max_time * 0.75))
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))
        if new_size < pop_size:
            order = np.argsort(fit)[:new_size]; pop = pop[order]; fit = fit[order]; pop_size = new_size

    if best_x is None: return best
    x = best_x.copy(); fx = best; step = ranges * 0.1; prev_x = x.copy()
    
    while remaining() > 0.05:
        improved = False
        for d in range(dim):
            if remaining() <= 0: return best
            for sign in [1, -1]:
                xt = x.copy(); xt[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                ft = eval_f(xt)
                if ft < fx: x = xt; fx = ft; step[d] *= 1.2; improved = True; break
            else: step[d] *= 0.5
        if improved:
            dv = x - prev_x
            if np.any(dv != 0):
                xp = clip(x + dv); fp = eval_f(xp)
                if fp < fx: x = xp; fx = fp
            prev_x = x.copy()
        if not improved and np.max(step / ranges) < 1e-11:
            x = best_x + np.random.randn(dim) * ranges * 0.005
            x = clip(x); fx = eval_f(x); step = ranges * 0.008; prev_x = x.copy()
    
    return best
