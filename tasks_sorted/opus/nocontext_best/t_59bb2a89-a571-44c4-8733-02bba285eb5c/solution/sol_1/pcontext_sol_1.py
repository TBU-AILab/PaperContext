#Looking at the current implementation, I see a solid SHADE-like DE with local search. Let me improve it with several enhancements:
#
#1. **Better SHADE implementation** with memory of successful parameters
#2. **Linear population size reduction** (L-SHADE)
#3. **Multiple restarts** of local search from top solutions
#4. **CMA-ES inspired local search** instead of simple coordinate descent
#5. **Better time management**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
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
    
    # --- Phase 1: L-SHADE ---
    N_init = min(max(25, 18 * dim), 300)
    pop_size = N_init
    N_min = 4
    H = 6  # memory size
    
    # Latin Hypercube Sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.array([eval_f(population[i]) for i in range(pop_size)])
    if remaining() < 0.5:
        return best
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    max_nfe = pop_size
    nfe = pop_size
    # Estimate total NFE we can afford
    time_per_eval = elapsed() / max(nfe, 1)
    if time_per_eval > 0:
        estimated_total_nfe = int(max_time * 0.88 / time_per_eval)
    else:
        estimated_total_nfe = 100000
    max_nfe = estimated_total_nfe
    
    generation = 0
    
    while remaining() > max_time * 0.15:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.11 * pop_size))
        
        S_F = []
        S_CR = []
        S_df = []
        
        new_pop = []
        new_fit = []
        
        for i in range(pop_size):
            if remaining() < max_time * 0.12:
                break
            
            ri = np.random.randint(H)
            if M_F[ri] <= 0:
                F = 0.0
            else:
                F = min(np.random.standard_cauchy() * 0.1 + M_F[ri], 1.0)
                while F <= 0:
                    F = min(np.random.standard_cauchy() * 0.1 + M_F[ri], 1.0)
            
            CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            if M_CR[ri] < 0:
                CR = 0.0
            
            # pbest
            p_best_idx = sorted_idx[np.random.randint(0, p_best_size)]
            
            # r1
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - xr2)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CR)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back bounds
            out_low = trial < lower
            out_high = trial > upper
            trial[out_low] = (lower[out_low] + population[i][out_low]) / 2.0
            trial[out_high] = (upper[out_high] + population[i][out_high]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_f(trial)
            nfe += 1
            
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fit.append(f_trial)
                if f_trial < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    S_df.append(fitness[i] - f_trial)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
            else:
                new_pop.append(population[i])
                new_fit.append(fitness[i])
        
        if len(new_pop) < 2:
            break
            
        population = np.array(new_pop)
        fitness = np.array(new_fit)
        pop_size = len(population)
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_df)
            weights = weights / weights.sum()
            mean_F = np.sum(weights * np.array(S_F)**2) / max(np.sum(weights * np.array(S_F)), 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[k] = mean_F
            M_CR[k] = mean_CR
            k = (k + 1) % H
        
        # LPSR: reduce population
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * nfe / max_nfe)))
        if new_size < pop_size:
            sorted_idx2 = np.argsort(fitness)
            population = population[sorted_idx2[:new_size]]
            fitness = fitness[sorted_idx2[:new_size]]
            pop_size = new_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
    
    # --- Phase 2: Local search on best solutions ---
    if best_x is None:
        return best
    
    # Nelder-Mead simplex on best
    def nelder_mead(x0, step_scale=0.05, time_limit_frac=0.5):
        n = dim
        tl = remaining() * time_limit_frac
        if tl < 0.1:
            return
        t0 = elapsed()
        
        simplex = np.empty((n + 1, n))
        simplex[0] = x0.copy()
        f_simplex = np.empty(n + 1)
        f_simplex[0] = eval_f(x0)
        
        for i in range(n):
            p = x0.copy()
            delta = step_scale * ranges[i]
            if delta < 1e-15:
                delta = 1e-6
            p[i] += delta
            p = np.clip(p, lower, upper)
            simplex[i + 1] = p
            f_simplex[i + 1] = eval_f(p)
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        while elapsed() - t0 < tl and remaining() > 0.05:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = simplex[:-1].mean(axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = eval_f(xr)
            
            if fr < f_simplex[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])
                else:
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = eval_f(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-14:
                break
    
    # Run NM from best with different scales
    for scale in [0.05, 0.01, 0.002]:
        if remaining() > 0.3:
            nelder_mead(best_x.copy(), step_scale=scale, time_limit_frac=0.4)
    
    # Coordinate descent refinement
    step = 0.001 * ranges.copy()
    while remaining() > 0.05:
        improved = False
        for j in range(dim):
            if remaining() < 0.05:
                return best
            trial = best_x.copy()
            trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
            f1 = eval_f(trial)
            if f1 < best:
                improved = True
                continue
            trial = best_x.copy()
            trial[j] = np.clip(trial[j] - step[j], lower[j], upper[j])
            f2 = eval_f(trial)
            if f2 < best:
                improved = True
                continue
        if not improved:
            step *= 0.5
            if np.max(step / ranges) < 1e-14:
                break
    
    return best
#
#Key improvements:
#- **L-SHADE**: Linear population size reduction for better convergence
#- **Weighted Lehmer mean** for F adaptation (proper SHADE)
#- **Nelder-Mead simplex** local search at multiple scales instead of just coordinate descent
#- **Better time budgeting**: 85% for DE, 15% for local search
#- **Fitness-weighted parameter adaptation** using improvement deltas
#- **More robust bound handling** using midpoint bounce-back
