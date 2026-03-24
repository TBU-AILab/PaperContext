#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Better DE strategy**: Use SHADE-like adaptive DE with success-history based parameter adaptation
#2. **Multiple restarts**: Restart the search from different regions to avoid local optima
#3. **Better local search**: Use a coordinate descent with adaptive step sizes instead of Nelder-Mead (which struggles in high dimensions)
#4. **Opposition-based learning** for better exploration
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

    # --- Phase 1: SHADE (Success-History based Adaptive DE) ---
    pop_size = max(20, min(80, dim * 5))
    H = 20  # memory size
    
    # Latin Hypercube Sampling for initial population
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            pop[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size * ranges[d]
    
    fit = np.array([eval_f(p) for p in pop])
    
    # Opposition-based population initialization
    opp_pop = lower + upper - pop
    opp_fit = np.array([eval_f(p) for p in opp_pop])
    
    all_pop = np.vstack([pop, opp_pop])
    all_fit = np.concatenate([fit, opp_fit])
    idx_sort = np.argsort(all_fit)[:pop_size]
    pop = all_pop[idx_sort].copy()
    fit = all_fit[idx_sort].copy()
    
    # SHADE memory
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    # Archive
    archive = []
    archive_max = pop_size
    
    p_min = 2.0 / pop_size
    p_max = 0.2
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while remaining() > max_time * 0.25:
        if remaining() <= 0:
            return best
        
        S_F = []
        S_CR = []
        delta_f = []
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            # Select from memory
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            # Generate CR from Normal
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = np.random.uniform(p_min, p_max)
            n_pbest = max(1, int(p * pop_size))
            pbest_idx = np.random.choice(np.argsort(fit)[:n_pbest])
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            r2_idx = np.random.choice(combined)
            if r2_idx < pop_size:
                xr2 = pop[r2_idx]
            else:
                xr2 = archive[r2_idx - pop_size]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            mutant = clip(mutant)
            
            # Binomial crossover
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi)
            mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            trial = clip(trial)
            
            trial_fit = eval_f(trial)
            
            if trial_fit <= fit[i]:
                if trial_fit < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fit[i] - trial_fit))
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fit[i] = trial_fit
        
        pop = new_pop
        fit = new_fit
        
        # Update memory
        if S_F:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[k] = mean_F
            M_CR[k] = mean_CR
            k = (k + 1) % H
        
        generation += 1
        
        if best < prev_best - 1e-10:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # Restart worst half if stagnating
        if stagnation > 30:
            order = np.argsort(fit)
            for idx in order[pop_size // 2:]:
                pop[idx] = np.random.uniform(lower, upper)
                fit[idx] = eval_f(pop[idx])
            stagnation = 0
    
    # --- Phase 2: Adaptive coordinate descent local search ---
    if best_x is None:
        return best
    
    x = best_x.copy()
    fx = best
    step = ranges * 0.1
    
    while remaining() > 0.05:
        improved = False
        for d in range(dim):
            if remaining() <= 0:
                return best
            
            # Try positive step
            x_trial = x.copy()
            x_trial[d] = min(x[d] + step[d], upper[d])
            f_trial = eval_f(x_trial)
            
            if f_trial < fx:
                x = x_trial
                fx = f_trial
                step[d] *= 1.2
                improved = True
                continue
            
            # Try negative step
            x_trial = x.copy()
            x_trial[d] = max(x[d] - step[d], lower[d])
            f_trial = eval_f(x_trial)
            
            if f_trial < fx:
                x = x_trial
                fx = f_trial
                step[d] *= 1.2
                improved = True
                continue
            
            step[d] *= 0.5
        
        if not improved and np.max(step / ranges) < 1e-10:
            # Restart local search with perturbation
            x = best_x + np.random.randn(dim) * ranges * 0.01
            x = clip(x)
            fx = eval_f(x)
            step = ranges * 0.01
    
    return best
