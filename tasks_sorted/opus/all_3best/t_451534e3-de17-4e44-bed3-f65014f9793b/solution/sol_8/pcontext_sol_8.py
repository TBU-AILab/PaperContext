#Looking at the results, algorithm 1 (score 18.68) is the best, using L-SHADE with aggressive coordinate descent. I'll improve it by:
#
#1. **Better initial sampling** with Latin Hypercube Sampling
#2. **Smarter time allocation** - more restarts with shorter DE phases, more local search
#3. **Golden section search** for 1D optimization within coordinate descent
#4. **Random permutation** of dimensions each CD pass
#5. **Perturbation-based escape** from local minima between CD passes
#6. **Smaller population** for faster DE generations
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_sol, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_sol = x.copy()
        return f

    def golden_section_1d(x_cur, f_cur, d, a_bound, b_bound, tol=1e-6, max_evals_gs=12):
        """Golden section search along dimension d between a_bound and b_bound."""
        gr = (np.sqrt(5) + 1) / 2
        c = b_bound - (b_bound - a_bound) / gr
        d_val = a_bound + (b_bound - a_bound) / gr
        n_ev = 0
        
        x_c = x_cur.copy(); x_c[d] = c; x_c = clip(x_c)
        fc = eval_func(x_c); n_ev += 1
        x_d = x_cur.copy(); x_d[d] = d_val; x_d = clip(x_d)
        fd = eval_func(x_d); n_ev += 1
        
        best_x = x_cur.copy()
        best_f = f_cur
        if fc < best_f: best_f = fc; best_x = x_c.copy()
        if fd < best_f: best_f = fd; best_x = x_d.copy()
        
        while abs(b_bound - a_bound) > tol * ranges[d] and n_ev < max_evals_gs and remaining() > 0.03:
            if fc < fd:
                b_bound = d_val
                d_val = c
                fd = fc
                c = b_bound - (b_bound - a_bound) / gr
                x_c = x_cur.copy(); x_c[d] = c; x_c = clip(x_c)
                fc = eval_func(x_c); n_ev += 1
                if fc < best_f: best_f = fc; best_x = x_c.copy()
            else:
                a_bound = c
                c = d_val
                fc = fd
                d_val = a_bound + (b_bound - a_bound) / gr
                x_d = x_cur.copy(); x_d[d] = d_val; x_d = clip(x_d)
                fd = eval_func(x_d); n_ev += 1
                if fd < best_f: best_f = fd; best_x = x_d.copy()
        
        return best_x, best_f, n_ev

    def coordinate_descent(x0, f0, init_step=0.05, min_step=1e-15, max_evals=None, time_limit=None, use_golden=False):
        if max_evals is None:
            max_evals = dim * 50
        if time_limit is None:
            time_limit = remaining() * 0.3
        t_start = elapsed()
        x_cur = x0.copy()
        f_cur = f0
        step = init_step * ranges.copy()
        n_evals = 0
        
        while n_evals < max_evals and (elapsed() - t_start) < time_limit and remaining() > 0.05:
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if remaining() <= 0.05 or n_evals >= max_evals:
                    return x_cur, f_cur
                
                # Try positive step
                x_try = x_cur.copy()
                x_try[d] += step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_prev = x_cur.copy()
                    x_cur = x_try
                    f_cur = f_try
                    # Accelerate
                    while n_evals < max_evals and remaining() > 0.05:
                        step[d] *= 2.0
                        x_try2 = x_cur.copy()
                        x_try2[d] += step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2
                            f_cur = f_try2
                        else:
                            step[d] *= 0.5
                            # Optional golden section refinement
                            if use_golden and n_evals + 8 < max_evals and remaining() > 0.1:
                                a_b = min(x_prev[d], x_try2[d])
                                b_b = max(x_prev[d], x_try2[d])
                                a_b = max(a_b, lower[d])
                                b_b = min(b_b, upper[d])
                                if b_b - a_b > 1e-14 * ranges[d]:
                                    xg, fg, ne = golden_section_1d(x_cur, f_cur, d, a_b, b_b, tol=1e-4, max_evals_gs=8)
                                    n_evals += ne
                                    if fg < f_cur:
                                        x_cur = xg; f_cur = fg
                            break
                    improved = True
                    continue
                
                # Try negative step
                x_try = x_cur.copy()
                x_try[d] -= step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_prev = x_cur.copy()
                    x_cur = x_try
                    f_cur = f_try
                    while n_evals < max_evals and remaining() > 0.05:
                        step[d] *= 2.0
                        x_try2 = x_cur.copy()
                        x_try2[d] -= step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2
                            f_cur = f_try2
                        else:
                            step[d] *= 0.5
                            if use_golden and n_evals + 8 < max_evals and remaining() > 0.1:
                                a_b = min(x_try2[d], x_prev[d])
                                b_b = max(x_try2[d], x_prev[d])
                                a_b = max(a_b, lower[d])
                                b_b = min(b_b, upper[d])
                                if b_b - a_b > 1e-14 * ranges[d]:
                                    xg, fg, ne = golden_section_1d(x_cur, f_cur, d, a_b, b_b, tol=1e-4, max_evals_gs=8)
                                    n_evals += ne
                                    if fg < f_cur:
                                        x_cur = xg; f_cur = fg
                            break
                    improved = True
                else:
                    step[d] *= 0.5
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < min_step:
                    break
        
        return x_cur, f_cur

    restart_count = 0
    
    while remaining() > 0.3:
        restart_count += 1
        time_for_de = remaining() * 0.40
        
        N_init = min(max(16, 5 * dim), 150)
        N_min = max(4, dim // 2 + 1)
        pop_size = N_init
        max_nfe_estimate = max(1, int(time_for_de * 800))
        nfe_at_start = evals
        
        H = 80
        memory_F = np.full(H, 0.5 if restart_count == 1 else 0.1 + 0.8 * np.random.rand())
        memory_CR = np.full(H, 0.5 if restart_count == 1 else 0.1 + 0.8 * np.random.rand())
        k_idx = 0
        archive = []; archive_max = N_init
        
        population = np.random.uniform(lower, upper, (N_init, dim))
        if restart_count > 1 and best_sol is not None:
            n_local = max(1, pop_size // 3)
            scale = max(0.005, 0.5 / restart_count)
            for j in range(n_local):
                population[j] = clip(best_sol + scale * ranges * np.random.randn(dim))
        
        fitness = np.array([eval_func(ind) for ind in population])
        if remaining() <= 0.3: break
        
        generation = 0; stagnation = 0; prev_best = best; de_start = elapsed()
        
        while remaining() > 0.3 and (elapsed() - de_start) < time_for_de:
            generation += 1
            nfe_since = evals - nfe_at_start
            ratio = min(1.0, nfe_since / max(1, max_nfe_estimate))
            new_ps = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            if new_ps < pop_size:
                si = np.argsort(fitness); population = population[si[:new_ps]]; fitness = fitness[si[:new_ps]]; pop_size = new_ps
            p_best_size = max(2, int((0.25 - 0.23 * ratio) * pop_size))
            ri = np.random.randint(0, H, pop_size)
            Fs = np.empty(pop_size)
            for idx in range(pop_size):
                for _ in range(20):
                    f_val = memory_F[ri[idx]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0: Fs[idx] = min(f_val, 1.0); break
                else: Fs[idx] = 0.5
            CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0, 1)
            S_F, S_CR, S_delta = [], [], []; sorted_idx = np.argsort(fitness)
            new_pop = population.copy(); new_fit = fitness.copy()
            for i in range(pop_size):
                if remaining() <= 0.2: break
                pi = sorted_idx[np.random.randint(0, p_best_size)]
                r1 = i
                while r1 == i: r1 = np.random.randint(pop_size)
                cs = pop_size + len(archive); r2 = i
                while r2 == i or r2 == r1: r2 = np.random.randint(cs)
                x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                mutant = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - x_r2)
                jrand = np.random.randint(dim); mask = np.random.rand(dim) < CRs[i]; mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                bl = trial < lower; ab = trial > upper
                trial[bl] = (lower[bl] + population[i][bl]) / 2; trial[ab] = (upper[ab] + population[i][ab]) / 2
                tf = eval_func(trial)
                if tf <= fitness[i]:
                    d = fitness[i] - tf
                    if tf < fitness[i]:
                        archive.append(population[i].copy())
                        if len(archive) > archive_max: archive.pop(np.random.randint(len(archive)))
                        S_F.append(Fs[i]); S_CR.append(CRs[i]); S_delta.append(d + 1e-30)
                    new_pop[i] = trial; new_fit[i] = tf
            population = new_pop; fitness = new_fit
            if S_F:
                w = np.array(S_delta); w /= w.sum(); sf = np.array(S_F); sc = np.array(S_CR)
                memory_F[k_idx % H] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
                memory_CR[k_idx % H] = np.sum(w * sc); k_idx += 1
            if abs(prev_best - best) < 1e-15: stagnation += 1
            else: stagnation = 0
            prev_best = best
            if stagnation > 40 or pop_size <= N_min: break
        
        if best_sol is not None and remaining() > 0.3:
            coordinate_descent(best_sol, best, init_step=0.1, max_evals=dim*35, time_limit=remaining()*0.15, use_golden=True)
        if best_sol is not None and remaining() > 0.3:
            coordinate_descent(best_sol, best, init_step=0.005, max_evals=dim*30, time_limit=remaining()*0.15, use_golden=False)
        if best_sol is not None and remaining() > 0.3:
            coordinate_descent(best_sol, best, init_step=0.0002, max_evals=dim*25, time_limit=remaining()*0.15)
    
    if best_sol is not None and remaining() > 0.05:
        coordinate_descent(best_sol, best, init_step=0.01, min_step=1e-16, max_evals=dim*200, time_limit=remaining()*0.45, use_golden=True)
    if best_sol is not None and remaining() > 0.05:
        coordinate_descent(best_sol, best, init_step=0.0001, min_step=1e-17, max_evals=dim*200, time_limit=remaining()*0.95)
    
    return best
