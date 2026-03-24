#Looking at the current algorithm, I see it's already a solid L-SHADE with local search. Let me analyze potential weaknesses and improve:
#
#1. **Better initialization** with opposition-based learning
#2. **Improved DE with current-to-pbest/1 and rank-based adaptation**
#3. **CMA-ES local search** instead of coordinate descent for better multi-dimensional refinement
#4. **Smarter time allocation** and more aggressive local search
#5. **Eigenvector-based crossover** for correlated variables
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

    # --- Phase 1: Opposition-based LHS initialization ---
    init_pop_size = min(max(30, 10 * dim), 250)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
    # LHS
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.85:
            return best
        fitness[i] = eval_func(population[i])
    
    # Opposition-based: evaluate opposition and keep best
    opp_pop = lower + upper - population
    opp_fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.85:
            break
        opp_fitness[i] = eval_func(opp_pop[i])
    
    combined = np.vstack([population, opp_pop])
    combined_f = np.concatenate([fitness, opp_fitness])
    sidx = np.argsort(combined_f)[:pop_size]
    population = combined[sidx]
    fitness = combined_f[sidx]

    # --- Phase 2: L-SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
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
        
        # Adaptive p value based on progress
        fraction_elapsed = min(elapsed() / de_time_budget, 1.0)
        p_rate = max(0.05, 0.25 - 0.20 * fraction_elapsed)
        
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            F = -1
            for _ in range(10):
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
            
            # pbest index
            p_num = max(2, int(p_rate * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p_num)]
            
            # r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            # r2 from pop + archive, != i, != r1
            combined_size = pop_size + len(archive)
            r2 = i
            for _ in range(30):
                r2 = np.random.randint(0, combined_size)
                if r2 != i and r2 != r1:
                    break
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # current-to-pbest/1 mutation
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back bounds handling
            out_low = trial < lower
            out_high = trial > upper
            if np.any(out_low):
                trial[out_low] = lower[out_low] + np.random.random(np.sum(out_low)) * (population[i][out_low] - lower[out_low])
            if np.any(out_high):
                trial[out_high] = upper[out_high] - np.random.random(np.sum(out_high)) * (upper[out_high] - population[i][out_high])
            trial = clip_to_bounds(trial)
            
            trial_f = eval_func(trial)
            
            if trial_f <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_f
                if trial_f < fitness[i]:
                    delta = fitness[i] - trial_f
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
            mean_cr = np.sum(weights * scr)
            if np.max(scr) == 0:
                M_CR[k] = -1
            else:
                M_CR[k] = mean_cr
            k = (k + 1) % H
        
        # Track improvement
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # Population size reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * fraction_elapsed)))
        
        if new_pop_size < pop_size:
            sidx2 = np.argsort(fitness)
            keep = sidx2[:new_pop_size]
            population = population[keep]
            fitness = fitness[keep]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation: partial restart
        if no_improve_count > 25 and pop_size > min_pop_size + 2:
            no_improve_count = 0
            sidx2 = np.argsort(fitness)
            n_restart = max(1, pop_size // 3)
            for kk in range(n_restart):
                idx = sidx2[pop_size - 1 - kk]
                if np.random.random() < 0.3:
                    population[idx] = lower + np.random.random(dim) * ranges
                else:
                    scale = 0.1 * (1.0 - 0.9 * fraction_elapsed)
                    population[idx] = best_x + np.random.randn(dim) * ranges * scale
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = eval_func(population[idx])

    # --- Phase 3: CMA-ES-like local search ---
    def cma_local_search(x0, sigma0, time_budget):
        nonlocal best, best_x
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_cma = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        
        t_end = elapsed() + time_budget
        
        gen = 0
        while elapsed() < t_end:
            gen += 1
            
            # Update eigen decomposition periodically
            if gen % max(1, int(lam / (10 * n))) == 0 or gen == 1:
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            # Sample offspring
            arz = np.random.randn(lam, n)
            arx = np.empty((lam, n))
            for ki in range(lam):
                arx[ki] = mean + sigma * (B @ (D * arz[ki]))
                arx[ki] = clip_to_bounds(arx[ki])
            
            # Evaluate
            arfitness = np.empty(lam)
            for ki in range(lam):
                if elapsed() >= t_end:
                    return
                arfitness[ki] = eval_func(arx[ki])
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = np.sum(weights[:, None] * selected, axis=0)
            
            # CSA
            zmean = np.sum(weights[:, None] * arz[arindex[:mu]], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2.0 / (n + 1))
            
            # CPC
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance matrix adaptation
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu_cma) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_cma * (artmp.T @ np.diag(weights) @ artmp)
            
            # Sigma adaptation
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            # Break if converged
            if sigma * np.max(D) < 1e-14 * np.max(ranges):
                break
            
            # Fix C symmetry
            C = np.triu(C) + np.triu(C, 1).T

    # Run CMA-ES from multiple starting points
    sidx_final = np.argsort(fitness)
    n_starts = min(5, pop_size)
    time_per_start = time_left() * 0.85 / max(n_starts, 1)
    
    visited = set()
    for si in range(n_starts):
        if time_left() < max_time * 0.02:
            break
        x0 = population[sidx_final[si]].copy()
        sigma0 = 0.05 * np.max(ranges) * max(0.01, 1.0 - 0.8 * si / max(n_starts - 1, 1))
        cma_local_search(x0, sigma0, min(time_per_start, time_left() * 0.8))
    
    # Final CMA from global best
    if time_left() > max_time * 0.02 and best_x is not None:
        cma_local_search(best_x.copy(), 0.01 * np.max(ranges), time_left() * 0.9)
    
    # --- Phase 4: Final Nelder-Mead polish ---
    if time_left() > max_time * 0.01 and best_x is not None:
        n = dim
        x = best_x.copy()
        fx = best
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
        simplex = np.empty((n + 1, n))
        simplex[0] = x.copy()
        simplex_f = np.empty(n + 1)
        simplex_f[0] = fx
        for j in range(n):
            p = x.copy()
            delta = max(0.001 * ranges[j], 1e-10)
            p[j] += delta
            p = clip_to_bounds(p)
            simplex[j + 1] = p
            simplex_f[j + 1] = eval_func(p)
        
        while time_left() > max_time * 0.002:
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            if np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30)) < 1e-15:
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
#- **Opposition-based learning** during initialization doubles initial coverage
#- **CMA-ES local search** replaces coordinate descent - much better for correlated landscapes
#- **Adaptive p-rate** that decreases over time for more exploitation
#- **Multi-start CMA-ES** from top candidates with decreasing sigma
#- **Better time allocation**: 78% DE, ~18% CMA-ES, ~4% Nelder-Mead polish
#- **Final CMA from global best** with small sigma for fine-tuning
#- **Eigendecomposition-based sampling** captures variable correlations that coordinate descent misses
