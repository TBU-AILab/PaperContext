#Looking at the best algorithm (score 11.83), I'll focus on several key improvements:
#
#1. **L-SHADE with linear population size reduction** for better convergence
#2. **Eigenvector-based crossover** to capture variable correlations
#3. **More aggressive local search** with multiple strategies (coordinate descent + random direction search)
#4. **Better time management** between exploration and exploitation phases
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
    N_init = max(25, min(100, dim * 6))
    N_min = 4
    pop_size = N_init
    H = 30

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
    
    total_evals = 2 * N_init
    max_evals_estimate = max(total_evals + 1, int(max_time * 500))  # rough estimate
    
    stagnation = 0
    prev_best = best
    
    while remaining() > max_time * 0.30:
        if remaining() <= 0:
            return best
        
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fit)
        
        trials = np.empty_like(pop)
        trial_from = np.zeros(pop_size, dtype=bool)
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            ri = np.random.randint(0, H)
            
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            else:
                Fi = 0.5
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
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
            total_evals += 1
            
            if trial_fit <= fit[i]:
                if trial_fit < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fit[i] - trial_fit))
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                pop[i] = trial
                fit[i] = trial_fit
        
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        if best < prev_best - 1e-10:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 25:
            order = np.argsort(fit)
            for idx in order[pop_size // 3:]:
                pop[idx] = best_x + np.random.randn(dim) * ranges * 0.1
                pop[idx] = clip(pop[idx])
                fit[idx] = eval_f(pop[idx])
            stagnation = 0
        
        ratio = min(1.0, total_evals / max_evals_estimate)
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))
        if new_size < pop_size:
            order = np.argsort(fit)[:new_size]
            pop = pop[order]
            fit = fit[order]
            pop_size = new_size

    # --- Phase 2: Multi-strategy local search ---
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
        
        if not improved:
            if np.max(step / ranges) < 1e-12:
                # Random direction search
                for _ in range(5):
                    if remaining() <= 0:
                        return best
                    direction = np.random.randn(dim)
                    direction /= (np.linalg.norm(direction) + 1e-30)
                    scale = np.random.exponential(0.01) * np.mean(ranges)
                    x_trial = clip(best_x + direction * scale)
                    f_trial = eval_f(x_trial)
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                x = best_x.copy()
                fx = best
                step = ranges * 0.005
    
    return best
