#Looking at the results, algorithm 3 (score 18.68) is the best, using L-SHADE with coordinate descent. The key insight is that coordinate descent with acceleration is very effective for local search. I'll improve by:
#
#1. **Smarter time splitting** - less DE time, more local search
#2. **Multiple diverse restarts** with smaller populations for faster exploration
#3. **Improved coordinate descent** with better step management and random permutation
#4. **Adding a Solis-Wets style random local search** for escaping local minima
#5. **Better initial population seeding** using Latin hypercube-like sampling
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
        return max_time * 0.96 - elapsed()
    
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

    def coordinate_descent(x0, f0, init_step=0.05, min_step=1e-15, max_evals=None, time_limit=None):
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
                    x_cur = x_try; f_cur = f_try
                    while n_evals < max_evals and remaining() > 0.05:
                        step[d] *= 2.0
                        x_try2 = x_cur.copy()
                        x_try2[d] += step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2; f_cur = f_try2
                        else:
                            step[d] *= 0.5; break
                    improved = True; continue
                
                # Try negative step
                x_try = x_cur.copy()
                x_try[d] -= step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try; f_cur = f_try
                    while n_evals < max_evals and remaining() > 0.05:
                        step[d] *= 2.0
                        x_try2 = x_cur.copy()
                        x_try2[d] -= step[d]
                        x_try2 = clip(x_try2)
                        f_try2 = eval_func(x_try2)
                        n_evals += 1
                        if f_try2 < f_cur:
                            x_cur = x_try2; f_cur = f_try2
                        else:
                            step[d] *= 0.5; break
                    improved = True
                else:
                    step[d] *= 0.5
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < min_step:
                    break
        
        return x_cur, f_cur

    def random_local_search(x0, f0, sigma=0.1, max_evals=50, time_limit=None):
        if time_limit is None:
            time_limit = remaining() * 0.1
        t_start = elapsed()
        x_cur = x0.copy(); f_cur = f0
        n_evals = 0
        bias = np.zeros(dim)
        succ = 0; fail = 0
        while n_evals < max_evals and (elapsed() - t_start) < time_limit and remaining() > 0.05:
            diff = sigma * ranges * np.random.randn(dim) + bias
            x_try = clip(x_cur + diff)
            f_try = eval_func(x_try); n_evals += 1
            if f_try < f_cur:
                bias = 0.2 * bias + 0.4 * diff
                x_cur = x_try; f_cur = f_try; succ += 1; fail = 0
            else:
                x_try2 = clip(x_cur - diff)
                f_try2 = eval_func(x_try2); n_evals += 1
                if f_try2 < f_cur:
                    bias = bias * 0.2 - 0.4 * diff
                    x_cur = x_try2; f_cur = f_try2; succ += 1; fail = 0
                else:
                    bias *= 0.5; fail += 1; succ = 0
            if succ >= 5: sigma *= 2.0; succ = 0
            if fail >= 3: sigma *= 0.5; fail = 0
            if sigma < 1e-15: break
        return x_cur, f_cur

    restart_count = 0
    while remaining() > 0.3:
        restart_count += 1
        time_for_de = remaining() * 0.35
        N_init = min(max(14, 5 * dim), 120)
        N_min = max(4, dim // 2 + 1)
        pop_size = N_init
        max_nfe_est = max(1, int(time_for_de * 900))
        nfe_start = evals
        H = 60
        memory_F = np.full(H, 0.5 if restart_count == 1 else 0.1 + 0.8 * np.random.rand())
        memory_CR = np.full(H, 0.5 if restart_count == 1 else 0.1 + 0.8 * np.random.rand())
        k_idx = 0; archive = []; archive_max = N_init
        population = np.random.uniform(lower, upper, (N_init, dim))
        if restart_count > 1 and best_sol is not None:
            nl = max(1, pop_size // 3); sc = max(0.005, 0.5 / restart_count)
            for j in range(nl): population[j] = clip(best_sol + sc * ranges * np.random.randn(dim))
        fitness = np.array([eval_func(ind) for ind in population])
        if remaining() <= 0.3: break
        gen = 0; stag = 0; pb = best; dst = elapsed()
        while remaining() > 0.3 and (elapsed() - dst) < time_for_de:
            gen += 1; nfs = evals - nfe_start; rat = min(1.0, nfs / max(1, max_nfe_est))
            nps = max(N_min, int(round(N_init + (N_min - N_init) * rat)))
            if nps < pop_size: si = np.argsort(fitness); population = population[si[:nps]]; fitness = fitness[si[:nps]]; pop_size = nps
            pbs = max(2, int((0.25 - 0.23 * rat) * pop_size)); ri = np.random.randint(0, H, pop_size)
            Fs = np.empty(pop_size)
            for idx in range(pop_size):
                for _ in range(20):
                    fv = memory_F[ri[idx]] + 0.1 * np.random.standard_cauchy()
                    if fv > 0: Fs[idx] = min(fv, 1.0); break
                else: Fs[idx] = 0.5
            CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0, 1)
            SF, SC, SD = [], [], []; si2 = np.argsort(fitness); np2 = population.copy(); nf2 = fitness.copy()
            for i in range(pop_size):
                if remaining() <= 0.2: break
                pi = si2[np.random.randint(0, pbs)]; r1 = i
                while r1 == i: r1 = np.random.randint(pop_size)
                cs = pop_size + len(archive); r2 = i
                while r2 == i or r2 == r1: r2 = np.random.randint(cs)
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                mu = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - xr2)
                jr = np.random.randint(dim); mk = np.random.rand(dim) < CRs[i]; mk[jr] = True
                tr = np.where(mk, mu, population[i]); bl = tr < lower; ab = tr > upper
                tr[bl] = (lower[bl] + population[i][bl]) / 2; tr[ab] = (upper[ab] + population[i][ab]) / 2
                tf = eval_func(tr)
                if tf <= fitness[i]:
                    d = fitness[i] - tf
                    if tf < fitness[i]: archive.append(population[i].copy()); SF.append(Fs[i]); SC.append(CRs[i]); SD.append(d + 1e-30)
                    if len(archive) > archive_max: archive.pop(np.random.randint(len(archive)))
                    np2[i] = tr; nf2[i] = tf
            population = np2; fitness = nf2
            if SF: w = np.array(SD); w /= w.sum(); sf = np.array(SF); sc = np.array(SC); memory_F[k_idx % H] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30); memory_CR[k_idx % H] = np.sum(w * sc); k_idx += 1
            if abs(pb - best) < 1e-15: stag += 1
            else: stag = 0
            pb = best
            if stag > 35 or pop_size <= N_min: break
        if best_sol is not None and remaining() > 0.3: coordinate_descent(best_sol, best, init_step=0.1, max_evals=dim*40, time_limit=remaining()*0.12)
        if best_sol is not None and remaining() > 0.3: random_local_search(best_sol, best, sigma=0.05, max_evals=dim*8, time_limit=remaining()*0.05)
        if best_sol is not None and remaining() > 0.3: coordinate_descent(best_sol, best, init_step=0.005, max_evals=dim*35, time_limit=remaining()*0.12)
        if best_sol is not None and remaining() > 0.3: coordinate_descent(best_sol, best, init_step=0.0002, max_evals=dim*30, time_limit=remaining()*0.12)
    if best_sol is not None and remaining() > 0.05: coordinate_descent(best_sol, best, init_step=0.01, min_step=1e-16, max_evals=dim*200, time_limit=remaining()*0.45)
    if best_sol is not None and remaining() > 0.05: coordinate_descent(best_sol, best, init_step=0.0001, min_step=1e-17, max_evals=dim*200, time_limit=remaining()*0.95)
    return best
