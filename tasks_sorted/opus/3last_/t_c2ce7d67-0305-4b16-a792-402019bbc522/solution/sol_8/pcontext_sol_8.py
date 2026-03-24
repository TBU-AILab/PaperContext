#Looking at the results, algorithm 1 performed best (28.24) with its multi-phase approach: LHS + multiple DE strategies + coordinate descent. Algorithm 3 (simple DE + coordinate descent) got 103.8, and algorithm 2 (SHADE + CMA-ES) got 207.2.
#
#Key improvements to make:
#1. Better SHADE DE with proper parameter adaptation
#2. CMA-ES restarts from best solutions found
#3. More aggressive local search with multiple scales
#4. Better time management
#5. Combine the best elements from all three
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Latin Hypercube Sampling initialization
    n_init = min(500, max(100, dim * 12))
    perms = np.zeros((n_init, dim))
    for d in range(dim):
        perms[:, d] = np.random.permutation(n_init)
    init_samples = (perms + np.random.rand(n_init, dim)) / n_init
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.04:
            break
        x = lower + init_samples[i] * ranges
        eval_f(x)

    # Phase 2: SHADE DE
    def run_shade(time_frac):
        nonlocal best, best_x
        pop_size = min(max(7 * dim, 50), 200)
        pop = lower + np.random.rand(pop_size, dim) * ranges
        fit = np.zeros(pop_size)
        
        if best_x is not None:
            pop[0] = best_x.copy()
            for j in range(1, min(pop_size // 5, 8)):
                scale = 0.05 * (j + 1)
                pop[j] = np.clip(best_x + scale * ranges * np.random.randn(dim), lower, upper)
        
        for i in range(pop_size):
            if remaining() < max_time * 0.5:
                pop = pop[:max(i,1)]
                fit = fit[:max(i,1)]
                pop_size = max(i,1)
                break
            fit[i] = eval_f(pop[i])
        
        if pop_size < 4:
            return
            
        H = 6
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.85)
        mem_idx = 0
        archive = []
        
        end_time = elapsed() + max_time * time_frac
        stagnation = 0
        local_best = best
        gen = 0
        
        while elapsed() < end_time and remaining() > max_time * 0.15:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            
            sort_idx = np.argsort(fit[:pop_size])
            
            for i in range(pop_size):
                if elapsed() >= end_time or remaining() < max_time * 0.15:
                    break
                
                ri = np.random.randint(H)
                while True:
                    F_i = np.random.standard_cauchy() * 0.1 + memory_F[ri]
                    if F_i > 0:
                        break
                F_i = min(F_i, 1.0)
                CR_i = np.clip(np.random.randn() * 0.1 + memory_CR[ri], 0.0, 1.0)
                
                # DE/current-to-pbest/1 with archive
                p = max(2, int(0.11 * pop_size))
                pb = sort_idx[np.random.randint(p)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = pop_size + len(archive)
                r2 = np.random.randint(combined)
                attempts = 0
                while (r2 == i or r2 == r1) and attempts < 20:
                    r2 = np.random.randint(combined)
                    attempts += 1
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + F_i * (pop[pb] - pop[i]) + F_i * (pop[r1] - xr2)
                
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i][d]) / 2.0
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i][d]) / 2.0
                
                cross_points = np.random.rand(dim) < CR_i
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = eval_f(trial)
                if f_trial <= fit[i]:
                    df = fit[i] - f_trial
                    if df > 0:
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        S_df.append(df)
                    archive.append(pop[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial
                    fit[i] = f_trial
            
            if S_F:
                w = np.array(S_df)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                memory_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                memory_CR[mem_idx] = np.sum(w * scr)
                mem_idx = (mem_idx + 1) % H
            
            if best < local_best - 1e-12:
                local_best = best
                stagnation = 0
            else:
                stagnation += 1
            
            if stagnation > 25:
                half = pop_size // 2
                for j in range(half, pop_size):
                    if best_x is not None:
                        sc = 0.1 * (1 + stagnation * 0.03)
                        pop[j] = np.clip(best_x + sc * ranges * np.random.randn(dim), lower, upper)
                    else:
                        pop[j] = lower + np.random.rand(dim) * ranges
                    if remaining() < max_time * 0.15:
                        break
                    fit[j] = eval_f(pop[j])
                stagnation = 0

    run_shade(0.55)

    # Phase 3: Multi-scale coordinate descent with random perturbations
    if best_x is not None:
        for scale_idx in range(15):
            if remaining() < 0.08:
                break
            step = ranges * (0.05 * (0.4 ** scale_idx))
            improved = True
            while improved and remaining() > 0.05:
                improved = False
                perm = np.random.permutation(dim)
                for i in perm:
                    if remaining() < 0.03:
                        break
                    for d_val in [step[i], -step[i]]:
                        xt = best_x.copy()
                        xt[i] = np.clip(best_x[i] + d_val, lower[i], upper[i])
                        if eval_f(xt) < best:
                            improved = True
                            break
            # Random restarts near best
            if remaining() > 0.1:
                sc = 0.02 * (0.5 ** min(scale_idx, 5))
                for _ in range(5):
                    if remaining() < 0.05:
                        break
                    xr = np.clip(best_x + sc * ranges * np.random.randn(dim), lower, upper)
                    eval_f(xr)

    return best
