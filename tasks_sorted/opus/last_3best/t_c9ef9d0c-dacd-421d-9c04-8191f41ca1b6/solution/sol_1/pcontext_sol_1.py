#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Slow index generation** with list comprehensions for `np.random.choice`
#2. **Element-wise crossover/boundary loops** are slow in Python
#3. **Population size might be suboptimal**
#4. **SHADE-style adaptation** would be better than random F/CR
#5. **More restarts with diverse strategies** could help escape local minima
#6. **CMA-ES-like local search** could be more effective than Nelder-Mead
#
#Let me implement an improved version with SHADE (Success-History based Adaptive DE) + multiple restarts + vectorized operations:
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
    
    def time_left(fraction=0.98):
        return elapsed() < max_time * fraction
    
    def clip_bounds(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip_bounds(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    pop_size = min(max(10 * dim, 40), 150)
    H = 50  # memory size for SHADE
    
    def run_shade():
        nonlocal best, best_params
        
        # LHS initialization
        pop = np.random.uniform(0, 1, (pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, d] = (perm + pop[:, d]) / pop_size
        pop = lower + pop * ranges
        
        fit = np.array([evaluate(pop[i]) for i in range(pop_size) if time_left(0.95)] + [float('inf')] * max(0, pop_size - sum(1 for _ in range(pop_size))))
        
        # Recompute properly
        fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if not time_left(0.95): return
            fit[i] = evaluate(pop[i])
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        
        while time_left(0.90):
            S_F, S_CR, S_w = [], [], []
            
            for i in range(pop_size):
                if not time_left(0.90): return
                
                ri = np.random.randint(H)
                Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.1, 1.0)
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
                
                # current-to-pbest/1
                p = max(2, int(0.1 * pop_size))
                pbest_idx = np.random.randint(p) if p > 0 else 0
                sorted_idx = np.argpartition(fit, p)[:p]
                xp = pop[sorted_idx[np.random.randint(len(sorted_idx))]]
                
                candidates = [j for j in range(pop_size) if j != i]
                r1 = candidates[np.random.randint(len(candidates))]
                pool = list(range(pop_size)) + list(range(len(archive)))
                r2v = np.random.randint(len(pool))
                xr2 = pop[pool[r2v]] if pool[r2v] < pop_size else archive[pool[r2v] - pop_size]
                
                mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
                
                mask = np.random.random(dim) < CRi
                mask[np.random.randint(dim)] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Bounce-back
                below = trial < lower
                above = trial > upper
                trial[below] = (lower[below] + pop[i][below]) / 2
                trial[above] = (upper[above] + pop[i][above]) / 2
                trial = clip_bounds(trial)
                
                f_trial = evaluate(trial)
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi); S_CR.append(CRi); S_w.append(abs(fit[i] - f_trial))
                        if len(archive) < pop_size: archive.append(pop[i].copy())
                        else: archive[np.random.randint(len(archive))] = pop[i].copy()
                    pop[i] = trial; fit[i] = f_trial
            
            if S_F:
                w = np.array(S_w); w /= w.sum() + 1e-30
                M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[k] = np.sum(w * np.array(S_CR))
                k = (k + 1) % H
    
    run_shade()
    
    # Local search: coordinate descent
    if best_params is not None:
        x = best_params.copy()
        fx = best
        step = 0.01 * ranges
        while time_left(0.98):
            for d in range(dim):
                if not time_left(0.98): break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn; fx = fn; step[d] *= 1.2; break
                else:
                    step[d] *= 0.5
    
    return best
#
#Wait, let me clean this up properly:
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

    pop_size = min(max(10 * dim, 40), 150)
    H = 50

    # LHS init
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + pop[:, d]) / pop_size
    pop = lower + pop * ranges
    
    fit = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95: return best
        fit[i] = evaluate(pop[i])
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    
    while elapsed() < max_time * 0.88:
        S_F, S_CR, S_w = [], [], []
        for i in range(pop_size):
            if elapsed() >= max_time * 0.88: break
            ri = np.random.randint(H)
            Fi = min(1.0, max(0.1, M_F[ri] + 0.1 * np.random.standard_cauchy()))
            CRi = min(1.0, max(0.0, np.random.normal(M_CR[ri], 0.1)))
            
            p = max(2, int(0.1 * pop_size))
            top_p = np.argpartition(fit, p)[:p]
            xp = pop[top_p[np.random.randint(len(top_p))]]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = idxs[np.random.randint(len(idxs))]
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size - 1)
            if r2 >= i: r2 += 1
            if r2 >= pop_size:
                xr2 = archive[r2 - pop_size]
            else:
                xr2 = pop[r2]
            
            mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
            
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2.0
            trial[above] = (upper[above] + pop[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_w.append(abs(fit[i] - f_trial))
                    if len(archive) < pop_size:
                        archive.append(pop[i].copy())
                    else:
                        archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial
                fit[i] = f_trial
        
        if S_F:
            w = np.array(S_w)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
    
    # Local search phase - pattern search
    if best_params is not None:
        x = best_params.copy()
        fx = best
        step = 0.02 * ranges
        min_step = 1e-8 * ranges
        
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98: break
                for sign in [1.0, -1.0]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x = xn
                        fx = fn
                        step[d] *= 1.5
                        improved = True
                        break
                else:
                    step[d] *= 0.5
            
            if not improved and np.all(step < min_step):
                # Restart local search with larger step
                step = 0.005 * ranges
                if np.all(step < min_step):
                    break
    
    return best
