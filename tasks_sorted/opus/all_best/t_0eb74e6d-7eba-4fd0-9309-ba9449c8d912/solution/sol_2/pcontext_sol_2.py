#Looking at the best algorithm (score 11.83), I can identify several improvements:
#
#1. **L-SHADE with linear population size reduction** - more efficient than fixed population
#2. **Better time allocation** - spend more time on DE, less rigid phase splitting
#3. **Improved local search** - combine coordinate descent with pattern search (accelerated moves)
#4. **Multiple local search restarts** from top solutions
#5. **Faster inner loop** by vectorizing where possible
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
    
    evals = 0
    def eval_f(x):
        nonlocal best, best_x, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: L-SHADE ---
    N_init = max(25, min(100, dim * 6))
    N_min = 4
    pop_size = N_init
    H = 30
    max_evals_shade = None  # we use time-based
    
    # Latin Hypercube Sampling
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            pop[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
    
    fit = np.array([eval_f(p) for p in pop])
    
    # Opposition-based initialization
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
    evals_at_start = evals
    stagnation = 0
    prev_best = best
    
    while remaining() > max_time * 0.20:
        if remaining() <= 0:
            return best
        
        S_F = []
        S_CR = []
        delta_f = []
        
        new_pop = []
        new_fit_list = []
        
        # Sorted indices for pbest selection
        sorted_idx = np.argsort(fit)
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            else:
                Fi = 0.5
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = np.random.uniform(max(2.0/pop_size, 0.05), 0.25)
            n_pbest = max(1, int(p * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, n_pbest)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pool_size)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            # Bounce-back clipping
            for dd in range(dim):
                if mutant[dd] < lower[dd]:
                    mutant[dd] = (lower[dd] + pop[i][dd]) / 2.0
                elif mutant[dd] > upper[dd]:
                    mutant[dd] = (upper[dd] + pop[i][dd]) / 2.0
            
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            
            trial_fit = eval_f(trial)
            
            if trial_fit <= fit[i]:
                if trial_fit < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fit[i] - trial_fit))
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                new_pop.append(trial)
                new_fit_list.append(trial_fit)
            else:
                new_pop.append(pop[i])
                new_fit_list.append(fit[i])
        
        pop = np.array(new_pop)
        fit = np.array(new_fit_list)
        
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        generation += 1
        
        # L-SHADE population reduction
        ratio = min(1.0, (evals - evals_at_start) / max(1, dim * 1000))
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))
        if new_size < pop_size:
            order = np.argsort(fit)[:new_size]
            pop = pop[order]
            fit = fit[order]
            pop_size = new_size
    
    # --- Phase 2: Pattern search + coordinate descent from best solutions ---
    if best_x is None:
        return best
    
    x = best_x.copy()
    fx = best
    step = ranges * 0.05
    
    while remaining() > 0.05:
        improved = False
        for d in range(dim):
            if remaining() <= 0:
                return best
            for sign in [1, -1]:
                x_trial = x.copy()
                x_trial[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                f_trial = eval_f(x_trial)
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    step[d] *= 1.3
                    improved = True
                    break
            else:
                step[d] *= 0.5
        
        if not improved and np.max(step / ranges) < 1e-12:
            x = best_x + np.random.randn(dim) * ranges * 0.005
            x = clip(x)
            fx = eval_f(x)
            step = ranges * 0.01
    
    return best
