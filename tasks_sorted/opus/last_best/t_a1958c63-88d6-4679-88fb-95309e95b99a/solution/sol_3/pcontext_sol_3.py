#Looking at the previous algorithms, I can see opportunities for improvement:
#
#1. **Better local search** - CMA-ES style covariance adaptation instead of simple coordinate descent
#2. **Multiple restarts** with different strategies
#3. **Better time management** - more aggressive local search when converging
#4. **Hybrid approach** combining DE with proper Nelder-Mead and pattern search
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def rand_in_bounds():
        return lower + ranges * np.random.rand(dim)
    
    # Nelder-Mead
    def nelder_mead(x0, max_evals=2000, initial_step=0.1):
        if time_left() < 0.05 * max_time:
            return
        n = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        step = initial_step * ranges
        simplex = [x0.copy()]
        simplex_f = [evaluate(x0)]
        for i in range(n):
            xi = x0.copy()
            xi[i] += step[i]
            xi = clip(xi)
            simplex.append(xi)
            simplex_f.append(evaluate(xi))
            if time_left() < 0.02 * max_time:
                return
        
        ev = 0
        stagnation = 0
        while ev < max_evals and time_left() > 0.02 * max_time:
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            old_best = simplex_f[0]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr); ev += 1
            
            if fr < simplex_f[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe); ev += 1
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                if fr < simplex_f[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc); ev += 1
                    if fc <= fr:
                        simplex[-1], simplex_f[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            simplex_f[i] = evaluate(simplex[i]); ev += 1
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc); ev += 1
                    if fc < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            simplex_f[i] = evaluate(simplex[i]); ev += 1
            
            if abs(simplex_f[0] - old_best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation > 10 * n:
                break
    
    # --- Phase 1: SHADE ---
    pop_size = min(max(8 * dim, 40), 200)
    n_init = pop_size * 2
    
    all_x, all_f = [], []
    for _ in range(n_init):
        if time_left() < max_time * 0.5:
            break
        x = rand_in_bounds()
        f = evaluate(x)
        all_x.append(x); all_f.append(f)
    
    all_x = np.array(all_x); all_f = np.array(all_f)
    idx = np.argsort(all_f)[:pop_size]
    pop = all_x[idx].copy(); pop_fit = all_f[idx].copy()
    
    H = 100
    M_F = np.full(H, 0.5); M_CR = np.full(H, 0.8)
    ki = 0; archive = []
    N_init = pop_size; N_min = max(4, dim)
    generation = 0; no_improve = 0; prev_best = best
    
    while time_left() > max_time * 0.25:
        generation += 1
        S_F, S_CR, S_df = [], [], []
        
        r_idx = np.random.randint(0, H, size=pop_size)
        F_vals = np.empty(pop_size)
        for i in range(pop_size):
            while True:
                f_val = 0.1 * np.random.standard_cauchy() + M_F[r_idx[i]]
                if f_val > 0:
                    F_vals[i] = min(f_val, 1.0)
                    break
        CR_vals = np.clip(np.random.randn(pop_size) * 0.1 + M_CR[r_idx], 0.0, 1.0)
        
        n_pbest = max(2, int(0.11 * pop_size))
        pbest_idx = np.argsort(pop_fit)[:n_pbest]
        
        for i in range(pop_size):
            if time_left() < max_time * 0.25:
                break
            Fi, CRi = F_vals[i], CR_vals[i]
            pi = pbest_idx[np.random.randint(n_pbest)]
            
            r1 = np.random.randint(pop_size)
            while r1 == i: r1 = np.random.randint(pop_size)
            
            union_sz = pop_size + len(archive)
            r2 = np.random.randint(union_sz)
            while r2 == i or r2 == r1: r2 = np.random.randint(union_sz)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pi] - pop[i]) + Fi * (pop[r1] - xr2)
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi; mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            if f_trial <= pop_fit[i]:
                if f_trial < pop_fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_df.append(pop_fit[i] - f_trial)
                    archive.append(pop[i].copy())
                pop[i] = trial; pop_fit[i] = f_trial
        
        if len(archive) > pop_size:
            archive = [archive[j] for j in np.random.choice(len(archive), pop_size, replace=False)]
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            M_F[ki] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[ki] = np.sum(w * np.array(S_CR)); ki = (ki + 1) % H
        
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * generation / max(1, max_time * 8))))
        if new_size < pop_size:
            sidx = np.argsort(pop_fit)[:new_size]
            pop = pop[sidx]; pop_fit = pop_fit[sidx]; pop_size = new_size
        
        if best < prev_best - 1e-14: no_improve = 0
        else: no_improve += 1
        prev_best = best
        
        if no_improve > 15 + dim:
            n_keep = max(2, pop_size // 4)
            kidx = np.argsort(pop_fit)[:n_keep]
            for i in range(pop_size):
                if i not in kidx:
                    pop[i] = best_params + 0.3 * ranges * (np.random.rand(dim) - 0.5)
                    pop[i] = clip(pop[i])
                    pop_fit[i] = evaluate(pop[i])
            no_improve = 0
    
    # Phase 2: Multiple Nelder-Mead restarts
    if best_params is not None:
        for scale in [0.05, 0.02, 0.005, 0.001]:
            if time_left() < 0.03 * max_time:
                break
            nelder_mead(best_params.copy(), max_evals=3000, initial_step=scale)
    
    return best
#
#Key improvements over last version:
#1. **Proper Nelder-Mead** with both inside/outside contraction and stagnation detection
#2. **Multiple NM restarts** at decreasing scales for progressive refinement
#3. **Cauchy regeneration** ensuring positive F values
#4. **Smarter restart**: perturbs around best rather than fully random
#5. **Better time allocation**: 75% DE, 25% local search
#6. **Higher initial CR** (0.8) which tends to work better for many problems
