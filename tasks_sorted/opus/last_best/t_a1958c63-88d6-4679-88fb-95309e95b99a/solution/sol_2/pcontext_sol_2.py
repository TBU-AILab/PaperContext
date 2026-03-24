#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Too much time on initialization** (3x pop_size random samples)
#2. **Local search is too simple** - coordinate descent is weak; Nelder-Mead or pattern search would be better
#3. **Population size may be too large**, wasting evaluations
#4. **No population size reduction** (L-SHADE style)
#5. **Need stronger local search** at end of budget
#
#Let me implement L-SHADE with linear population size reduction plus a more aggressive local search phase.
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
    
    # Nelder-Mead local search
    def nelder_mead(x0, f0, max_evals=500):
        n = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        # Initial simplex
        step = 0.05 * ranges
        simplex = [x0.copy()]
        simplex_f = [f0]
        for i in range(n):
            xi = x0.copy()
            xi[i] += step[i]
            xi = clip(xi)
            fi = evaluate(xi)
            simplex.append(xi)
            simplex_f.append(fi)
            if time_left() < max_time * 0.01:
                return
        
        evals_used = n
        while evals_used < max_evals and time_left() > max_time * 0.01:
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr); evals_used += 1
            
            if fr < simplex_f[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe); evals_used += 1
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                if fr < simplex_f[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = evaluate(xc); evals_used += 1
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1], simplex_f[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        simplex_f[i] = evaluate(simplex[i]); evals_used += 1
                        if time_left() < max_time * 0.01:
                            return
    
    # --- L-SHADE ---
    N_init = min(max(10 * dim, 50), 200)
    N_min = 4
    pop_size = N_init
    
    pop = np.array([rand_in_bounds() for _ in range(pop_size)])
    pop_fit = np.array([evaluate(pop[i]) for i in range(pop_size)])
    
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    ki = 0
    archive = []
    generation = 0
    no_improve = 0
    prev_best = best
    
    while time_left() > max_time * 0.15:
        generation += 1
        S_F, S_CR, S_df = [], [], []
        
        r_idx = np.random.randint(0, H, size=pop_size)
        F_vals = np.clip(np.random.standard_cauchy(pop_size) * 0.1 + M_F[r_idx], 0.01, 1.0)
        CR_vals = np.clip(np.random.randn(pop_size) * 0.1 + M_CR[r_idx], 0.0, 1.0)
        
        n_pbest = max(2, int(0.11 * pop_size))
        pbest_idx = np.argsort(pop_fit)[:n_pbest]
        
        new_pop = pop.copy()
        new_fit = pop_fit.copy()
        
        for i in range(pop_size):
            if time_left() < max_time * 0.15:
                break
            Fi, CRi = F_vals[i], CR_vals[i]
            pi = pbest_idx[np.random.randint(n_pbest)]
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(union_size)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[pi] - pop[i]) + Fi * (pop[r1] - xr2)
            
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            if f_trial <= pop_fit[i]:
                if f_trial < pop_fit[i]:
                    S_F.append(Fi); S_CR.append(CRi)
                    S_df.append(pop_fit[i] - f_trial)
                    archive.append(pop[i].copy())
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        pop, pop_fit = new_pop, new_fit
        
        if len(archive) > pop_size:
            archive = [archive[j] for j in np.random.choice(len(archive), pop_size, replace=False)]
        
        if S_F:
            w = np.array(S_df); w = w / (w.sum() + 1e-30)
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[ki] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[ki] = np.sum(w * scr)
            ki = (ki + 1) % H
        
        # L-SHADE population reduction
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * generation / max(1, max_time * 10))))
        if new_size < pop_size:
            idx = np.argsort(pop_fit)[:new_size]
            pop = pop[idx]
            pop_fit = pop_fit[idx]
            pop_size = new_size
        
        if best < prev_best - 1e-12:
            no_improve = 0
        else:
            no_improve += 1
        prev_best = best
        
        if no_improve > 25 + dim:
            n_keep = max(2, pop_size // 4)
            idx = np.argsort(pop_fit)[:n_keep]
            for i in range(pop_size):
                if i not in idx:
                    pop[i] = rand_in_bounds()
                    pop_fit[i] = evaluate(pop[i])
            no_improve = 0
    
    # Final local search phase
    if best_params is not None and time_left() > max_time * 0.02:
        nelder_mead(best_params.copy(), best, max_evals=5000)
    
    return best
#
#Key improvements:
#1. **L-SHADE** with population size reduction for better convergence
#2. **Nelder-Mead** local search at the end for fine-tuning (much stronger than coordinate descent)
#3. **Better time allocation**: 85% for L-SHADE, 15% for Nelder-Mead refinement
#4. **Less wasteful initialization** - direct population instead of 3x oversampling
#5. **Proper archive management** with random subsampling
