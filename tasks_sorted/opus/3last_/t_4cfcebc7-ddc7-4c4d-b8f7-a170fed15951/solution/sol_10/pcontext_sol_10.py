#Looking at the results, algorithm 1 (score 11.44) performed best. It used L-SHADE + Nelder-Mead + Powell + coordinate descent but was cut off mid-code. Algorithm 3 (37.39) had good structure but spent too much time on restarts. Let me improve by:
#
#1. **Better L-SHADE with larger population and more time**
#2. **CMA-ES as secondary global search**
#3. **Tighter local search with multiple methods**
#4. **Better time management**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    def time_left():
        return max_time - elapsed()
    def clip(x):
        return np.clip(x, lower, upper)
    
    top_solutions = []
    eval_count = [0]
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def add_top(x, f):
        top_solutions.append((x.copy(), f))
        top_solutions.sort(key=lambda t: t[1])
        if len(top_solutions) > 80:
            top_solutions[:] = top_solutions[:80]
    
    # --- Phase 1: Latin Hypercube Sampling + Opposition ---
    n_init = min(max(30 * dim, 400), 1500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    for i in range(n_init):
        if time_left() < max_time * 0.05:
            return best
        f = eval_func(init_pop[i])
        add_top(init_pop[i], f)
    
    # Opposition-based
    n_opp = min(n_init // 3, 300)
    for i in range(n_opp):
        if time_left() < max_time * 0.05:
            break
        opp = lower + upper - init_pop[i]
        f_opp = eval_func(opp)
        add_top(opp, f_opp)
    
    # Midpoints from top pairs
    top_n = min(20, len(top_solutions))
    for i in range(top_n):
        for j in range(i+1, min(i+5, top_n)):
            if time_left() < max_time * 0.04:
                break
            alpha = np.random.uniform(0.2, 0.8)
            mid = alpha * top_solutions[i][0] + (1 - alpha) * top_solutions[j][0]
            f_mid = eval_func(mid)
            add_top(mid, f_mid)
    
    # --- Phase 2: L-SHADE ---
    def lshade(time_frac_stop=0.35):
        nonlocal best, best_params
        pop_size_init = min(max(12 * dim, 60), 250)
        pop_size_min = max(4, dim // 2)
        pop_size = pop_size_init
        
        n_elite = min(pop_size // 2, len(top_solutions))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = top_solutions[i][0].copy()
            fit[i] = top_solutions[i][1]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if time_left() < max_time * time_frac_stop:
                for j in range(pop_size):
                    add_top(pop[j], fit[j])
                return
            fit[i] = eval_func(pop[i])
            add_top(pop[i], fit[i])
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = pop_size_init
        nfe_count = 0
        max_nfe = 200000
        
        for gen in range(50000):
            if time_left() < max_time * time_frac_stop:
                break
            S_F, S_CR, S_delta = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            fit_order = np.argsort(fit)
            p_max = max(2, int(pop_size * 0.2))
            p_min = max(2, int(pop_size * 0.05))
            
            for i in range(pop_size):
                if time_left() < max_time * time_frac_stop:
                    break
                ri = np.random.randint(H)
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
                
                p = np.random.randint(p_min, p_max + 1)
                pbest_idx = fit_order[np.random.randint(p)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i, d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i, d]) / 2
                
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                f_trial = eval_func(trial)
                nfe_count += 1
                add_top(trial, f_trial)
                
                if f_trial <= fit[i]:
                    delta = fit[i] - f_trial
                    if delta > 0:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_delta.append(delta)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            if S_F:
                w = np.array(S_delta)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
            
            ratio = min(nfe_count / max_nfe, 1.0)
            new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
            if new_size < pop_size:
                order = np.argsort(fit)
                pop = pop[order[:new_size]]
                fit = fit[order[:new_size]]
                pop_size = new_size
        
        for i in range(pop_size):
            add_top(pop[i], fit[i])
    
    lshade(time_frac_stop=0.30)
    
    # --- Phase 3: Nelder-Mead ---
    def nelder_mead(x0, scale=0.03, max_iter=10000, time_stop=0.01):
        nonlocal best, best_params
        n = dim
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += ranges[i] * scale * (1 if np.random.random() > 0.5 else -1)
            simplex[i+1] = clip(p)
        f_s = np.array([eval_func(simplex[i]) for i in range(n+1)])
        for si in range(n+1):
            add_top(simplex[si], f_s[si])
        no_improve = 0
        for it in range(max_iter):
            if time_left() < max_time * time_stop:
                return
            o = np.argsort(f_s); simplex = simplex[o]; f_s = f_s[o]
            if np.max(np.abs(simplex[-1]-simplex[0])) < 1e-15:
                return
            old_best_f = f_s[0]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_func(xr); add_top(xr, fr)
            if fr < f_s[0]:
                xe = clip(c + 2*(xr-c)); fe = eval_func(xe); add_top(xe, fe)
                simplex[-1], f_s[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_s[-2]:
                simplex[-1], f_s[-1] = xr, fr
            else:
                if fr < f_s[-1]:
                    xc = clip(c + 0.5*(xr-c)); fc = eval_func(xc); add_top(xc, fc)
                    if fc <= fr: simplex[-1], f_s[-1] = xc, fc; continue
                else:
                    xc = clip(c + 0.5*(simplex[-1]-c)); fc = eval_func(xc); add_top(xc, fc)
                    if fc < f_s[-1]: simplex[-1], f_s[-1] = xc, fc; continue
                for i in range(1, n+1):
                    if time_left() < max_time * time_stop: return
                    simplex[i] = clip(simplex[0] + 0.5*(simplex[i]-simplex[0]))
                    f_s[i] = eval_func(simplex[i]); add_top(simplex[i], f_s[i])
            if f_s[0] >= old_best_f - 1e-15:
                no_improve += 1
                if no_improve > 80: return
            else:
                no_improve = 0
    
    def coordinate_search(x0, initial_step=0.01, min_step=1e-14, time_stop=0.01):
        nonlocal best, best_params
        x = x0.copy(); fx = eval_func(x); step = initial_step
        while step > min_step:
            if time_left() < max_time * time_stop: return
            improved = False
            for d in range(dim):
                if time_left() < max_time * time_stop: return
                delta = ranges[d] * step
                for sign in [1, -1]:
                    x_try = x.copy()
                    x_try[d] = np.clip(x[d] + sign * delta, lower[d], upper[d])
                    f_try = eval_func(x_try)
                    if f_try < fx:
                        x = x_try; fx = f_try; improved = True; break
            if not improved:
                step *= 0.5
    
    # Multiple NM from diverse top solutions
    if best_params is not None and time_left() > 2:
        nelder_mead(best_params, scale=0.05, time_stop=0.05)
    
    visited = [best_params.copy()] if best_params is not None else []
    for idx in range(min(15, len(top_solutions))):
        if time_left() < max_time * 0.12: break
        sol = top_solutions[idx][0]
        if all(np.linalg.norm(sol - v) > np.mean(ranges) * 0.02 for v in visited):
            visited.append(sol.copy())
            nelder_mead(sol, scale=0.03, time_stop=0.06)
    
    for _ in range(8):
        if time_left() < max_time * 0.08: break
        if best_params is not None:
            perturbed = best_params + np.random.randn(dim) * ranges * 0.08
            nelder_mead(clip(perturbed), scale=0.02, time_stop=0.04)
    
    if best_params is not None and time_left() > 1.5:
        nelder_mead(best_params, scale=0.005, time_stop=0.03)
    if best_params is not None and time_left() > 0.8:
        coordinate_search(best_params, initial_step=0.005, time_stop=0.02)
    if best_params is not None and time_left() > 0.4:
        nelder_mead(best_params, scale=0.001, time_stop=0.01)
    if best_params is not None and time_left() > 0.2:
        coordinate_search(best_params, initial_step=0.0002, time_stop=0.005)
    
    return best
