#Looking at the results, the best approach (24.2) used DE + Nelder-Mead. The SHADE variant (213.8) performed worst, likely due to bugs or overhead. Let me build an improved version with:
#
#1. **Proper SHADE with cleaner implementation** - success-history based parameter adaptation
#2. **Better time management** - use ~75% for DE, 25% for local search
#3. **Multiple Nelder-Mead restarts** from different top solutions
#4. **Smaller, more efficient population** for faster convergence
#5. **Better mutation strategy** mixing
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    eval_count = [0]
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- LHS Initialization ---
    pop_size = min(max(20, 8 * dim), 200)
    
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + np.random.rand(pop_size)) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([eval_f(pop[i]) for i in range(pop_size)])
    
    # Sort
    idx = np.argsort(fit)
    pop = pop[idx]
    fit = fit[idx]
    
    # DE parameters - adaptive
    F_base = 0.7
    CR_base = 0.9
    
    stagnation = 0
    prev_best = best
    
    # --- Phase 1: Differential Evolution ---
    de_deadline = max_time * 0.70
    
    while elapsed() < de_deadline:
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        # Adaptive parameters based on progress
        progress = elapsed() / de_deadline
        F = F_base * (1.0 - 0.3 * progress)  # decrease F over time
        CR = CR_base
        
        for i in range(pop_size):
            if elapsed() >= de_deadline:
                break
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            Fi = np.clip(F + 0.15 * np.random.standard_cauchy(), 0.1, 1.5)
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
            
            r = np.random.rand()
            if r < 0.4:
                # current-to-pbest/1
                p_best = max(1, int(0.15 * pop_size))
                pb = np.random.randint(0, p_best)
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[a] - pop[b])
            elif r < 0.7:
                # current-to-best/1
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = pop[i] + Fi * (pop[0] - pop[i]) + Fi * (pop[a] - pop[b])
            else:
                # rand/1
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + Fi * (pop[b] - pop[c])
            
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            
            # Bounce-back bounds
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_f(trial)
            if f_trial <= fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        pop = new_pop
        fit = new_fit
        idx = np.argsort(fit)
        pop = pop[idx]
        fit = fit[idx]
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 6:
            n_replace = pop_size // 2
            for j in range(pop_size - n_replace, pop_size):
                if np.random.rand() < 0.5 and best_x is not None:
                    pop[j] = best_x + 0.1 * ranges * np.random.randn(dim)
                else:
                    pop[j] = lower + np.random.rand(dim) * ranges
                pop[j] = np.clip(pop[j], lower, upper)
                fit[j] = eval_f(pop[j])
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            stagnation = 0
    
    # --- Phase 2: Nelder-Mead from multiple starting points ---
    n_starts = min(5, pop_size)
    for si in range(n_starts):
        if remaining() < max_time * 0.03:
            break
        
        x0 = pop[si].copy()
        n_s = dim + 1
        simplex = np.zeros((n_s, dim))
        sf = np.zeros(n_s)
        simplex[0] = x0
        sf[0] = fit[si]
        scale = 0.03 * ranges * (1 + si * 0.5)
        for j in range(1, n_s):
            simplex[j] = x0.copy()
            d_idx = (j - 1) % dim
            simplex[j][d_idx] += scale[d_idx] * (1 if np.random.rand() > 0.5 else -1)
            simplex[j] = np.clip(simplex[j], lower, upper)
            sf[j] = eval_f(simplex[j])
        
        deadline_nm = elapsed() + remaining() / (n_starts - si + 0.5)
        while elapsed() < min(deadline_nm, max_time * 0.97):
            oi = np.argsort(sf)
            simplex, sf = simplex[oi], sf[oi]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + (centroid - simplex[-1]), lower, upper)
            fr = eval_f(xr)
            if fr < sf[0]:
                xe = np.clip(centroid + 2 * (xr - centroid), lower, upper)
                fe = eval_f(xe)
                simplex[-1], sf[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < sf[-2]:
                simplex[-1], sf[-1] = xr, fr
            else:
                if fr < sf[-1]:
                    xc = np.clip(centroid + 0.5 * (xr - centroid), lower, upper)
                else:
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lower, upper)
                fc = eval_f(xc)
                if fc < sf[-1]:
                    simplex[-1], sf[-1] = xc, fc
                else:
                    for j in range(1, n_s):
                        simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                        simplex[j] = np.clip(simplex[j], lower, upper)
                        sf[j] = eval_f(simplex[j])
    
    return best
