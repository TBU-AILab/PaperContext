#Looking at the results, algorithm 3 (score 10.84) performed best with L-SHADE + Nelder-Mead. Algorithm 2 (11.00) added coordinate descent. Algorithm 1 (12.87) used CMA-ES which was slower. Let me combine the best elements and improve:
#
#1. **More efficient L-SHADE** with better time management
#2. **Hybrid local search**: Nelder-Mead + coordinate descent + pattern search
#3. **Multiple diverse restarts** from top solutions
#4. **Better population initialization** using Sobol-like stratification
#5. **Adaptive time allocation** based on dimension
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
        while len(top_solutions) > 40:
            top_solutions.pop()
    
    # --- Phase 1: Latin Hypercube Sampling + Opposition ---
    n_init = min(max(20 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.05:
            return best
        init_fitness[i] = eval_func(init_pop[i])
        add_top(init_pop[i], init_fitness[i])
    
    # Opposition-based candidates
    n_opp = min(n_init // 2, 300)
    for i in range(n_opp):
        if time_left() < max_time * 0.05:
            break
        opp = lower + upper - init_pop[i]
        f_opp = eval_func(opp)
        add_top(opp, f_opp)
    
    # Midpoint candidates from top pairs
    top_n = min(20, len(top_solutions))
    for i in range(top_n):
        for j in range(i+1, min(i+3, top_n)):
            if time_left() < max_time * 0.05:
                break
            mid = 0.5 * (top_solutions[i][0] + top_solutions[j][0])
            f_mid = eval_func(mid)
            add_top(mid, f_mid)
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: L-SHADE ---
    def lshade(time_frac_stop=0.25):
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
        max_nfe = 120000
        
        for gen in range(10000):
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
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
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
    
    lshade(time_frac_stop=0.25)
    
    # --- Phase 3: Nelder-Mead from multiple diverse starts ---
    def nelder_mead(x0, scale=0.03, max_iter=5000, time_stop=0.01):
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
        for it in range(max_iter):
            if time_left() < max_time * time_stop:
                return
            o = np.argsort(f_s); simplex = simplex[o]; f_s = f_s[o]
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                return
            old_best = f_s[0]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_func(xr)
            add_top(xr, fr)
            if fr < f_s[0]:
                xe = clip(c + 2*(xr - c)); fe = eval_func(xe)
                add_top(xe, fe)
                simplex[-1], f_s[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_s[-2]:
                simplex[-1], f_s[-1] = xr, fr
            else:
                if fr < f_s[-1]:
                    xc = clip(c + 0.5*(xr - c)); fc = eval_func(xc)
                    add_top(xc, fc)
                    if fc <= fr:
                        simplex[-1], f_s[-1] = xc, fc
                        continue
                else:
                    xc = clip(c - 0.5*(c - simplex[-1])); fc = eval_func(xc)
                    add_top(xc, fc)
                    if fc < f_s[-1]:
                        simplex[-1], f_s[-1] = xc, fc
                        continue
                for i in range(1, n+1):
                    if time_left() < max_time * time_stop:
                        return
                    simplex[i] = clip(simplex[0] + 0.5*(simplex[i] - simplex[0]))
                    f_s[i] = eval_func(simplex[i])
                    add_top(simplex[i], f_s[i])
            if f_s[0] >= old_best - 1e-15:
                no_improve += 1
                if no_improve > 50:
                    return
            else:
                no_improve = 0
    
    def coordinate_search(x0, initial_step=0.01, min_step=1e-12, time_stop=0.01):
        nonlocal best, best_params
        x = x0.copy()
        fx = eval_func(x)
        step = initial_step
        while step > min_step:
            if time_left() < max_time * time_stop:
                return
            improved = False
            for d in range(dim):
                if time_left() < max_time * time_stop:
                    return
                delta = ranges[d] * step
                x_try = x.copy()
                x_try[d] = min(x[d] + delta, upper[d])
                f_try = eval_func(x_try)
                if f_try < fx:
                    x = x_try; fx = f_try; improved = True; continue
                x_try = x.copy()
                x_try[d] = max(x[d] - delta, lower[d])
                f_try = eval_func(x_try)
                if f_try < fx:
                    x = x_try; fx = f_try; improved = True; continue
            if not improved:
                step *= 0.5
    
    # NM from best
    if best_params is not None and time_left() > 1.5:
        nelder_mead(best_params, scale=0.03, time_stop=0.03)
    
    # NM from diverse top solutions
    visited = [best_params.copy()] if best_params is not None else []
    for idx in range(min(15, len(top_solutions))):
        if time_left() < max_time * 0.12:
            break
        sol = top_solutions[idx][0]
        is_diverse = all(np.linalg.norm(sol - v) > np.mean(ranges) * 0.015 for v in visited)
        if is_diverse:
            visited.append(sol.copy())
            nelder_mead(sol, scale=0.02, time_stop=0.05)
    
    # Fine NM polish
    if best_params is not None and time_left() > 0.8:
        nelder_mead(best_params, scale=0.005, time_stop=0.02)
    
    # Coordinate descent
    if best_params is not None and time_left() > 0.5:
        coordinate_search(best_params, initial_step=0.005, time_stop=0.02)
    
    # Ultra-fine NM
    if best_params is not None and time_left() > 0.3:
        nelder_mead(best_params, scale=0.001, time_stop=0.01)
    
    # Final coordinate descent
    if best_params is not None and time_left() > 0.15:
        coordinate_search(best_params, initial_step=0.0005, time_stop=0.01)
    
    # Last ultra-fine coordinate
    if best_params is not None and time_left() > 0.05:
        coordinate_search(best_params, initial_step=0.00005, time_stop=0.005)
    
    return best
