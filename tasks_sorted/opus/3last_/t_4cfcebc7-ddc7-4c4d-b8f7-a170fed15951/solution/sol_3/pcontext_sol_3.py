#Looking at the results, algorithm 1 (score 10.84) performed best with L-SHADE + multiple Nelder-Mead polishing rounds. Let me improve upon it with:
#
#1. **Better time allocation** - more time for local search
#2. **Multiple restarts from diverse good solutions** (not just the best)
#3. **Coordinate-wise local search** for fine-tuning
#4. **Improved L-SHADE with better archive management**
#5. **Powell-like pattern search** as additional local optimizer
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
    
    eval_cache = {}
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        key = x.tobytes()
        if key in eval_cache:
            return eval_cache[key]
        f = func(x)
        eval_cache[key] = f
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling + Opposition ---
    n_init = min(max(20 * dim, 200), 1000)
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
    
    # Opposition-based candidates
    n_opp = min(n_init // 2, 300)
    opp_pop = []
    opp_fit = []
    for i in range(n_opp):
        if time_left() < max_time * 0.05:
            break
        opp = lower + upper - init_pop[i]
        f_opp = eval_func(opp)
        opp_pop.append(opp)
        opp_fit.append(f_opp)
    
    # Combine all evaluated points
    all_pop = list(init_pop) + opp_pop
    all_fit = list(init_fitness) + opp_fit
    all_pop = np.array(all_pop)
    all_fit = np.array(all_fit)
    sorted_idx = np.argsort(all_fit)
    
    # Store top solutions for later restarts
    n_top = min(20, len(sorted_idx))
    top_solutions = [(all_pop[sorted_idx[i]].copy(), all_fit[sorted_idx[i]]) for i in range(n_top)]
    
    # --- Phase 2: L-SHADE ---
    def lshade():
        nonlocal best, best_params
        
        pop_size_init = min(max(12 * dim, 60), 300)
        pop_size_min = max(4, dim // 2)
        pop_size = pop_size_init
        
        n_elite = min(pop_size // 3, len(sorted_idx))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = all_pop[sorted_idx[i]].copy()
            fit[i] = all_fit[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if time_left() < max_time * 0.15:
                return
            fit[i] = eval_func(pop[i])
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = pop_size_init
        
        nfe_start_count = 0
        max_nfe = 150000
        
        for gen in range(10000):
            if time_left() < max_time * 0.25:
                return
            
            S_F, S_CR, S_delta = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            fit_order = np.argsort(fit)
            p_max = max(2, int(pop_size * 0.2))
            p_min = max(2, int(pop_size * 0.05))
            
            for i in range(pop_size):
                if time_left() < max_time * 0.20:
                    return
                
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
                nfe_start_count += 1
                
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
            
            ratio = min(nfe_start_count / max_nfe, 1.0)
            new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
            if new_size < pop_size:
                order = np.argsort(fit)
                pop = pop[order[:new_size]]
                fit = fit[order[:new_size]]
                pop_size = new_size
        
        # Update top solutions
        for i in range(pop_size):
            top_solutions.append((pop[i].copy(), fit[i]))
        top_solutions.sort(key=lambda x: x[1])
        while len(top_solutions) > 20:
            top_solutions.pop()
    
    lshade()
    
    # --- Phase 3: Nelder-Mead from multiple good starts ---
    def nelder_mead(x0, scale=0.03, max_iter=5000):
        nonlocal best, best_params
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += ranges[i] * scale * (1 if np.random.random() > 0.5 else -1)
            simplex[i + 1] = clip(p)
        f_s = np.array([eval_func(simplex[i]) for i in range(n + 1)])
        for it in range(max_iter):
            if time_left() < max_time * 0.02: return
            o = np.argsort(f_s); simplex = simplex[o]; f_s = f_s[o]
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15: return
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_func(xr)
            if fr < f_s[0]:
                xe = clip(c + 2*(xr - c)); fe = eval_func(xe)
                simplex[-1], f_s[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_s[-2]: simplex[-1], f_s[-1] = xr, fr
            else:
                if fr < f_s[-1]:
                    xc = clip(c + 0.5*(xr - c)); fc = eval_func(xc)
                    if fc <= fr: simplex[-1], f_s[-1] = xc, fc; continue
                else:
                    xc = clip(c - 0.5*(c - simplex[-1])); fc = eval_func(xc)
                    if fc < f_s[-1]: simplex[-1], f_s[-1] = xc, fc; continue
                for i in range(1, n+1):
                    if time_left() < 0.02: return
                    simplex[i] = clip(simplex[0] + 0.5*(simplex[i] - simplex[0]))
                    f_s[i] = eval_func(simplex[i])
    
    # --- Phase 4: Coordinate descent for fine-tuning ---
    def coordinate_search(x0, initial_step=0.01, min_step=1e-10, max_iter=5000):
        nonlocal best, best_params
        x = x0.copy()
        fx = eval_func(x)
        step = initial_step
        it = 0
        while step > min_step and it < max_iter:
            improved = False
            for d in range(dim):
                if time_left() < max_time * 0.01:
                    return
                it += 1
                delta = ranges[d] * step
                # Try positive direction
                x_try = x.copy()
                x_try[d] = min(x[d] + delta, upper[d])
                f_try = eval_func(x_try)
                if f_try < fx:
                    x = x_try; fx = f_try; improved = True; continue
                # Try negative direction
                x_try = x.copy()
                x_try[d] = max(x[d] - delta, lower[d])
                f_try = eval_func(x_try)
                if f_try < fx:
                    x = x_try; fx = f_try; improved = True; continue
            if not improved:
                step *= 0.5
    
    # Run NM from best point
    if best_params is not None and time_left() > 1.0:
        nelder_mead(best_params, scale=0.03)
    
    # Run NM from diverse top solutions
    for idx in range(1, min(5, len(top_solutions))):
        if time_left() < max_time * 0.10:
            break
        sol = top_solutions[idx][0]
        if np.linalg.norm(sol - best_params) > np.mean(ranges) * 0.01:
            nelder_mead(sol, scale=0.02)
    
    # Fine NM polish
    if best_params is not None and time_left() > 0.5:
        nelder_mead(best_params, scale=0.005)
    
    # Coordinate descent polish
    if best_params is not None and time_left() > 0.3:
        coordinate_search(best_params, initial_step=0.005)
    
    # Ultra-fine NM
    if best_params is not None and time_left() > 0.15:
        nelder_mead(best_params, scale=0.001)
    
    # Final coordinate descent
    if best_params is not None and time_left() > 0.05:
        coordinate_search(best_params, initial_step=0.0005)
    
    return best
#
#Key improvements over algorithm 1:
#- **Caching** to avoid re-evaluating identical points
#- **Coordinate descent** as additional local optimizer for fine-tuning
#- **Multiple NM restarts** from diverse top solutions (not just the best)
#- **Better time allocation**: more time reserved for local search phases
#- **More top solutions tracked** through the optimization for diverse restarts
#- **Progressively finer local search**: large→medium→small→tiny scale refinements
