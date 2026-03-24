#Looking at the current algorithm, I can identify several areas for improvement:
#
#1. **Better initial sampling** (Sobol-like quasi-random)
#2. **Improved L-SHADE with rank-based weighting**
#3. **CMA-ES local search** instead of just coordinate descent
#4. **Better time allocation** and more aggressive local refinement
#5. **Opposition-based learning** for diversity
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
    
    def time_left():
        return max_time - elapsed()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_x, evals
        x = clip_to_bounds(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Initialize with LHS + opposition ---
    init_pop_size = min(max(40, 15 * dim), 400)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.random(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.85:
            return best
        fitness[i] = eval_func(population[i])
    
    # Opposition-based: evaluate opposite points for worst half
    sidx = np.argsort(fitness)
    n_opp = pop_size // 3
    for i in range(n_opp):
        if elapsed() >= max_time * 0.80:
            break
        idx = sidx[pop_size - 1 - i]
        opp = lower + upper - population[idx]
        opp = clip_to_bounds(opp)
        opp_f = eval_func(opp)
        if opp_f < fitness[idx]:
            population[idx] = opp
            fitness[idx] = opp_f

    # --- Phase 2: L-SHADE with improvements ---
    H = 8
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    max_archive = pop_size
    
    no_improve_count = 0
    prev_best = best
    
    de_time_budget = max_time * 0.78
    
    while elapsed() < de_time_budget:
        sorted_idx = np.argsort(fitness)
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Adaptive p range
        fraction_elapsed = min(elapsed() / de_time_budget, 1.0)
        
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            F = -1
            for _ in range(20):
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F > 0:
                    break
            if F <= 0:
                F = 0.01
            F = min(F, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CR = 0.0
            else:
                CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # Adaptive p
            p_rate = max(0.05, 0.25 - 0.20 * fraction_elapsed)
            pi = max(2, int(p_rate * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, pi)]
            
            # r1
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            
            # r2 from pop + archive
            combined_size = pop_size + len(archive)
            r2 = i
            for _ in range(30):
                r2 = np.random.randint(0, combined_size)
                if r2 != i and r2 != r1:
                    break
            
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            # Current-to-pbest/1
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back
            out_low = trial < lower
            out_high = trial > upper
            if np.any(out_low):
                trial[out_low] = (lower[out_low] + population[i][out_low]) / 2.0
            if np.any(out_high):
                trial[out_high] = (upper[out_high] + population[i][out_high]) / 2.0
            trial = clip_to_bounds(trial)
            
            trial_f = eval_func(trial)
            
            if trial_f <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_f
                if trial_f < fitness[i]:
                    delta = abs(fitness[i] - trial_f)
                    S_F.append(F)
                    S_CR.append(CR)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
        
        population = new_population
        fitness = new_fitness
        
        # Update memory with weighted Lehmer mean
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (np.sum(weights) + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            
            M_F[k] = np.sum(weights * sf ** 2) / (np.sum(weights * sf) + 1e-30)
            if np.max(scr) == 0:
                M_CR[k] = -1
            else:
                M_CR[k] = np.sum(weights * scr)
            k = (k + 1) % H
        
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # Population reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * fraction_elapsed)))
        if new_pop_size < pop_size:
            sidx2 = np.argsort(fitness)
            population = population[sidx2[:new_pop_size]]
            fitness = fitness[sidx2[:new_pop_size]]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation: partial restart
        if no_improve_count > 40:
            no_improve_count = 0
            sidx2 = np.argsort(fitness)
            n_restart = max(1, pop_size // 3)
            for kk in range(n_restart):
                idx = sidx2[pop_size - 1 - kk]
                if np.random.random() < 0.3 and best_x is not None:
                    scale = 0.1 * (1 - fraction_elapsed) + 0.001
                    population[idx] = best_x + np.random.randn(dim) * ranges * scale
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = eval_func(population[idx])

    # --- Phase 3: CMA-ES-like local search ---
    if best_x is not None and time_left() > max_time * 0.02:
        # Simple (1+1)-CMA-ES style with step-size adaptation
        sigma = 0.01 * np.mean(ranges)
        x = best_x.copy()
        fx = best
        
        # Covariance as diagonal initially
        C_diag = (0.01 * ranges) ** 2
        success_rate = 0.2
        p_target = 0.2
        c_p = 0.1
        d_factor = 2.0
        
        while time_left() > max_time * 0.005:
            z = np.random.randn(dim)
            trial = x + sigma * np.sqrt(C_diag) * z
            trial = clip_to_bounds(trial)
            ft = eval_func(trial)
            
            if ft < fx:
                x = trial
                fx = ft
                success_rate = (1 - c_p) * success_rate + c_p
                # Update diagonal covariance towards successful direction
                C_diag = 0.95 * C_diag + 0.05 * (sigma * np.sqrt(C_diag) * z) ** 2
            else:
                success_rate = (1 - c_p) * success_rate
            
            # 1/5th rule
            if success_rate > p_target:
                sigma *= np.exp(1.0 / d_factor)
            else:
                sigma *= np.exp(-0.4 / d_factor)
            
            sigma = max(sigma, 1e-15 * np.mean(ranges))
            sigma = min(sigma, 0.5 * np.mean(ranges))

    # --- Phase 4: Nelder-Mead refinement ---
    if best_x is not None and time_left() > max_time * 0.01:
        n = dim
        x = best_x.copy()
        fx = best
        
        simplex = np.empty((n + 1, n))
        simplex[0] = x.copy()
        simplex_f = np.empty(n + 1)
        simplex_f[0] = fx
        
        for j in range(n):
            p = x.copy()
            delta = max(0.002 * ranges[j], 1e-10)
            p[j] += delta
            p = clip_to_bounds(p)
            simplex[j + 1] = p
            simplex_f[j + 1] = eval_func(p)
        
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while time_left() > max_time * 0.002:
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30))
            if spread < 1e-15:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip_to_bounds(centroid + alpha_nm * (centroid - simplex[-1]))
            fr = eval_func(xr)
            
            if fr < simplex_f[0]:
                xe = clip_to_bounds(centroid + gamma_nm * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe; simplex_f[-1] = fe
                else:
                    simplex[-1] = xr; simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr; simplex_f[-1] = fr
            else:
                if fr < simplex_f[-1]:
                    xc = clip_to_bounds(centroid + rho_nm * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc; simplex_f[-1] = fc
                    else:
                        for j in range(1, n + 1):
                            simplex[j] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[j] - simplex[0]))
                            simplex_f[j] = eval_func(simplex[j])
                else:
                    xc = clip_to_bounds(centroid - rho_nm * (centroid - simplex[-1]))
                    fc = eval_func(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc; simplex_f[-1] = fc
                    else:
                        for j in range(1, n + 1):
                            simplex[j] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[j] - simplex[0]))
                            simplex_f[j] = eval_func(simplex[j])
    
    return best
#
#Key improvements over the previous version:
#- **Opposition-based learning** during initialization to better explore the space
#- **Adaptive p-rate** that decreases over time (wider exploration early, focused later)
#- **(1+1)-CMA-ES style local search** with diagonal covariance adaptation and 1/5th success rule for step-size control
#- **Better time budget allocation**: 78% DE, then CMA-ES local search, then Nelder-Mead
#- **Improved bounce-back** using midpoint strategy
#- **Larger initial population** for better coverage
#- **Terminal CR memory** handling (set to -1 when all successful CRs are 0)
