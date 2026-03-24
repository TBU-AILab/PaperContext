#Looking at the results, algorithm 1 (score 37.4) performed terribly - likely due to time management issues. Algorithms 2 and 3 (scores ~12.8-12.9) performed similarly. The L-SHADE + Nelder-Mead combo (alg 2) slightly edged out L-SHADE + CMA-ES (alg 3). Let me improve with:
#
#1. **Better time management** - more aggressive global search
#2. **Hybrid approach**: L-SHADE with restarts + Nelder-Mead + coordinate descent
#3. **Multiple L-SHADE restarts** with smaller populations for diversity
#4. **Smarter local search scheduling**
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    top_solutions = []
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def add_top(x, f):
        top_solutions.append((x.copy(), f))
        top_solutions.sort(key=lambda t: t[1])
        while len(top_solutions) > 60:
            top_solutions.pop()
    
    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(20 * dim, 200), 600)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    for i in range(n_init):
        if time_left() < max_time * 0.85:
            break
        f = eval_func(init_pop[i])
        add_top(init_pop[i], f)
    
    # Opposition
    n_opp = min(n_init // 3, 150)
    for i in range(n_opp):
        if time_left() < max_time * 0.80:
            break
        opp = lower + upper - init_pop[i]
        f_opp = eval_func(opp)
        add_top(opp, f_opp)
    
    # Midpoints from top pairs
    top_n = min(10, len(top_solutions))
    for i in range(top_n):
        for j in range(i+1, min(i+3, top_n)):
            if time_left() < max_time * 0.78:
                break
            mid = 0.5 * (top_solutions[i][0] + top_solutions[j][0])
            f_mid = eval_func(mid)
            add_top(mid, f_mid)
    
    # --- Phase 2: L-SHADE ---
    def lshade(pop_init=None, time_frac_stop=0.30, pop_size_init=None):
        nonlocal best, best_params
        
        if pop_size_init is None:
            pop_size_init = min(max(10 * dim, 50), 200)
        pop_size_min = max(4, dim // 2)
        pop_size = pop_size_init
        
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        if pop_init is not None:
            n_seed = min(len(pop_init), pop_size)
            for i in range(n_seed):
                pop[i] = pop_init[i][0].copy()
                fit[i] = pop_init[i][1]
            for i in range(n_seed, pop_size):
                pop[i] = lower + np.random.random(dim) * ranges
                if time_left() < max_time * time_frac_stop:
                    return
                fit[i] = eval_func(pop[i])
                add_top(pop[i], fit[i])
        else:
            n_elite = min(pop_size // 2, len(top_solutions))
            for i in range(n_elite):
                pop[i] = top_solutions[i][0].copy()
                fit[i] = top_solutions[i][1]
            for i in range(n_elite, pop_size):
                pop[i] = lower + np.random.random(dim) * ranges
                if time_left() < max_time * time_frac_stop:
                    return
                fit[i] = eval_func(pop[i])
                add_top(pop[i], fit[i])
        
        H = 80
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.8)
        k = 0
        archive = []
        archive_max = pop_size_init
        nfe_count = 0
        max_nfe = 150000
        stag_count = 0
        prev_best_fit = np.min(fit)
        
        for gen in range(10000):
            if time_left() < max_time * time_frac_stop:
                break
            
            S_F, S_CR, S_delta = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            fit_order = np.argsort(fit)
            p_max = max(2, int(pop_size * 0.15))
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
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr)
                k = (k + 1) % H
            
            cur_best_fit = np.min(fit)
            if cur_best_fit >= prev_best_fit - 1e-15:
                stag_count += 1
            else:
                stag_count = 0
                prev_best_fit = cur_best_fit
            
            if stag_count > 30 + dim:
                break
            
            ratio = min(nfe_count / max_nfe, 1.0)
            new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
            if new_size < pop_size:
                order = np.argsort(fit)
                pop = pop[order[:new_size]]
                fit = fit[order[:new_size]]
                pop_size = new_size
        
        for i in range(pop_size):
            add_top(pop[i], fit[i])
    
    # First L-SHADE run
    lshade(time_frac_stop=0.35)
    
    # Second L-SHADE run with restart if time allows
    if time_left() > max_time * 0.45:
        # Create diverse population: mix top solutions with random
        restart_pop = []
        for i in range(min(10, len(top_solutions))):
            restart_pop.append(top_solutions[i])
        # Add perturbed versions
        for i in range(min(5, len(top_solutions))):
            px = top_solutions[i][0] + np.random.randn(dim) * ranges * 0.15
            px = clip(px)
            pf = eval_func(px)
            add_top(px, pf)
            restart_pop.append((px, pf))
        lshade(pop_init=restart_pop, time_frac_stop=0.25, pop_size_init=min(max(8*dim, 40), 150))
    
    # --- Phase 3: Nelder-Mead ---
    def nelder_mead(x0, scale=0.03, max_iter=8000, time_stop=0.02):
        nonlocal best, best_params
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += ranges[i] * scale * (1 if np.random.random() > 0.5 else -1)
            simplex[i + 1] = clip(p)
        f_s = np.array([eval_func(simplex[i]) for i in range(n + 1)])
        for si in range(n+1):
            add_top(simplex[si], f_s[si])
        no_improve = 0
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        for it in range(max_iter):
            if time_left() < max_time * time_stop:
                return
            o = np.argsort(f_s); simplex = simplex[o]; f_s = f_s[o]
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                return
            old_best = f_s[0]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c + alpha * (c - simplex[-1])); fr = eval_func(xr); add_top(xr, fr)
            if fr < f_s[0]:
                xe = clip(c + gamma * (xr - c)); fe = eval_func(xe); add_top(xe, fe)
                simplex[-1], f_s[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_s[-2]:
                simplex[-1], f_s[-1] = xr, fr
            else:
                if fr < f_s[-1]:
                    xc = clip(c + rho * (xr - c)); fc = eval_func(xc); add_top(xc, fc)
                    if fc <= fr:
                        simplex[-1], f_s[-1] = xc, fc; continue
                else:
                    xc = clip(c + rho * (simplex[-1] - c)); fc = eval_func(xc); add_top(xc, fc)
                    if fc < f_s[-1]:
                        simplex[-1], f_s[-1] = xc, fc; continue
                for i in range(1, n+1):
                    if time_left() < max_time * time_stop: return
                    simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                    f_s[i] = eval_func(simplex[i]); add_top(simplex[i], f_s[i])
            if f_s[0] >= old_best - 1e-15:
                no_improve += 1
                if no_improve > 60: return
            else:
                no_improve = 0
    
    def coordinate_search(x0, initial_step=0.01, min_step=1e-13, time_stop=0.01):
        nonlocal best, best_params
        x = x0.copy(); fx = eval_func(x); step = initial_step
        while step > min_step:
            if time_left() < max_time * time_stop: return
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if time_left() < max_time * time_stop: return
                delta = ranges[d] * step
                for sign in [1, -1]:
                    x_try = x.copy(); x_try[d] = np.clip(x[d] + sign * delta, lower[d], upper[d])
                    f_try = eval_func(x_try)
                    if f_try < fx:
                        x = x_try; fx = f_try; improved = True; break
            if not improved: step *= 0.5
    
    # NM from best
    if best_params is not None and time_left() > 1.5:
        nelder_mead(best_params, scale=0.04, time_stop=0.04)
    
    # NM from diverse top solutions
    visited = [best_params.copy()] if best_params is not None else []
    for idx in
