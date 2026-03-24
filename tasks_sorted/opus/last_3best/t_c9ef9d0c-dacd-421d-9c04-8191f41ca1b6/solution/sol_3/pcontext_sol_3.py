#Looking at the results, algorithm 1 (DE with multiple strategies + Nelder-Mead) scored best at 16.43, while algorithm 3 (L-SHADE + Nelder-Mead) scored worst at 209.4. The L-SHADE had issues with `max_evals_approx=100000` being a fixed constant that doesn't adapt to actual budget, and population reduction was too aggressive.
#
#Let me build an improved version combining the best elements: SHADE adaptation (proven effective), multiple mutation strategies, better time management, and enhanced local search with restarts.
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    pop_size = min(max(15 * dim, 50), 200)
    H = 50
    
    # LHS initialization
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + pop[:, d]) / pop_size
    pop = lower + pop * ranges
    
    fit = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fit[i] = evaluate(pop[i])
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.9)
    k = 0
    archive = []
    stagnation = 0
    prev_best = best
    
    # --- Main DE loop with SHADE adaptation + multi-strategy ---
    while elapsed() < max_time * 0.88:
        S_F, S_CR, S_w = [], [], []
        gen_improved = False
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.88:
                break
            
            ri = np.random.randint(H)
            Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            strategy = np.random.randint(3)
            
            if strategy == 0:
                # current-to-pbest/1
                p = max(2, int(0.1 * pop_size))
                top_p = np.argpartition(fit, p)[:p]
                xp = pop[top_p[np.random.randint(len(top_p))]]
                
                r1 = np.random.randint(pop_size - 1)
                if r1 >= i: r1 += 1
                
                union_size = pop_size + len(archive)
                if union_size > 1:
                    r2 = np.random.randint(union_size - 1)
                    if r2 >= i: r2 += 1
                    xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                else:
                    xr2 = pop[r1]
                
                mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
            
            elif strategy == 1:
                # DE/best/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mutant = best_params + Fi * (pop[idxs[0]] - pop[idxs[1]])
            
            else:
                # DE/rand/1
                idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                mutant = pop[idxs[0]] + Fi * (pop[idxs[1]] - pop[idxs[2]])
            
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            below = trial < lower; above = trial > upper
            trial[below] = lower[below] + np.random.random(np.sum(below)) * (pop[i][below] - lower[below])
            trial[above] = upper[above] - np.random.random(np.sum(above)) * (upper[above] - pop[i][above])
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(fit[i] - f_trial)
                    if len(archive) < pop_size: archive.append(pop[i].copy())
                    elif archive: archive[np.random.randint(len(archive))] = pop[i].copy()
                    gen_improved = True
                pop[i] = trial; fit[i] = f_trial
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        if not gen_improved:
            stagnation += 1
        else:
            stagnation = 0
        
        if stagnation > 15:
            n_replace = pop_size // 4
            worst_idx = np.argsort(fit)[-n_replace:]
            for idx in worst_idx:
                sc = 0.1 * ranges * (0.5 ** (stagnation // 15))
                pop[idx] = best_params + np.random.normal(0, 1, dim) * sc
                pop[idx] = np.clip(pop[idx], lower, upper)
                if elapsed() >= max_time * 0.88: break
                fit[idx] = evaluate(pop[idx])
            stagnation = 0
    
    # --- Nelder-Mead local search ---
    if best_params is not None:
        n = dim; scale = 0.05 * ranges
        simplex = np.zeros((n+1, n)); sf = np.zeros(n+1)
        simplex[0] = best_params.copy(); sf[0] = best
        for i in range(n):
            if elapsed() >= max_time * 0.98: return best
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i] if best_params[i] + scale[i] <= upper[i] else -scale[i]
            sf[i+1] = evaluate(simplex[i+1])
        while elapsed() < max_time * 0.98:
            o = np.argsort(sf); simplex = simplex[o]; sf = sf[o]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
            fr = evaluate(xr)
            if elapsed() >= max_time * 0.98: break
            if sf[0] <= fr < sf[-2]: simplex[-1] = xr; sf[-1] = fr
            elif fr < sf[0]:
                xe = np.clip(centroid + 2*(xr - centroid), lower, upper); fe = evaluate(xe)
                if fe < fr: simplex[-1] = xe; sf[-1] = fe
                else: simplex[-1] = xr; sf[-1] = fr
            else:
                xc = np.clip(centroid + 0.5*(simplex[-1] - centroid), lower, upper); fc = evaluate(xc)
                if fc < sf[-1]: simplex[-1] = xc; sf[-1] = fc
                else:
                    for i in range(1, n+1):
                        if elapsed() >= max_time*0.98: break
                        simplex[i] = simplex[0] + 0.5*(simplex[i]-simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper); sf[i] = evaluate(simplex[i])
    return best
