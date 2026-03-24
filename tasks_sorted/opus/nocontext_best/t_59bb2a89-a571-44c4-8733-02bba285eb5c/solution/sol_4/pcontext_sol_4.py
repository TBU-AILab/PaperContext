#Looking at the current algorithm, I see it's already a solid L-SHADE with local search. Let me analyze potential improvements:
#
#1. **Better initialization** using opposition-based learning
#2. **Improved DE strategy** with rank-based mutation weighting
#3. **CMA-ES local search** instead of coordinate descent for better exploitation
#4. **Adaptive time budgeting** based on dimension
#5. **More aggressive restarts** and population diversity management
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
    
    # Latin Hypercube Sampling
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.random(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = eval_func(population[i])
    
    # Opposition-based population
    if elapsed() < max_time * 0.15:
        opp_pop = lower + upper - population
        opp_fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= max_time * 0.20:
                break
            opp_fitness[i] = eval_func(opp_pop[i])
        
        combined = np.vstack([population, opp_pop])
        combined_f = np.concatenate([fitness, opp_fitness])
        sidx = np.argsort(combined_f)[:pop_size]
        population = combined[sidx]
        fitness = combined_f[sidx]

    # --- Phase 2: L-SHADE with improvements ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    max_archive = pop_size
    
    generation = 0
    no_improve_count = 0
    prev_best = best
    
    # Adaptive time budget: more for DE in higher dims
    de_fraction = min(0.82, 0.70 + 0.01 * dim)
    de_time_budget = max_time * de_fraction
    
    while elapsed() < de_time_budget:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Pre-generate random numbers for speed
        ri_all = np.random.randint(0, H, pop_size)
        cauchy_all = np.random.standard_cauchy(pop_size)
        normal_all = np.random.normal(0, 0.1, pop_size)
        
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            ri = ri_all[i]
            
            # Generate F via Cauchy
            F = M_F[ri] + 0.1 * cauchy_all[i]
            while F <= 0:
                F = M_F[ri] + 0.1 * np.random.standard_cauchy()
            F = min(F, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CR = 0.0
            else:
                CR = np.clip(M_CR[ri] + 0.1 * normal_all[i], 0, 1)
            
            # pbest with rank-weighted selection
            weights_p = 1.0 / (1.0 + np.arange(p_best_size))
            weights_p /= weights_p.sum()
            p_best_idx = sorted_idx[np.random.choice(p_best_size, p=weights_p)]
            
            # r1 != i
            r1 = np.random.randint(0, pop_size - 1)
            if r1 >= i:
                r1 += 1
            
            # r2 from pop + archive, != i, r1
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            attempts = 0
            while (r2 == i or r2 == r1) and attempts < 25:
                r2 = np.random.randint(0, combined_size)
                attempts += 1
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # current-to-pbest/1 mutation
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - x_r2)
            
            # Binomial crossover
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Bounce-back bounds handling
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
                    delta = fitness[i] - trial_f
                    S_F.append(F)
                    S_CR.append(CR)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
        
        population = new_population
        fitness = new_fitness
        
        # Update memory with Lehmer mean
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (np.sum(weights) + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            
            M_F[k] = np.sum(weights * sf ** 2) / (np.sum(weights * sf) + 1e-30)
            if np.max(scr) == 0:
                M_CR[k] = -1.0
            else:
                M_CR[k] = np.sum(weights * scr)
            k = (k + 1) % H
        
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # Linear population size reduction
        fraction_elapsed = elapsed() / de_time_budget
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * fraction_elapsed)))
        
        if new_pop_size < pop_size:
            sidx = np.argsort(fitness)[:new_pop_size]
            population = population[sidx]
            fitness = fitness[sidx]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation restart with diversity injection
        if no_improve_count > 25 and pop_size > min_pop_size + 4:
            no_improve_count = 0
            sidx = np.argsort(fitness)
            n_restart = max(1, pop_size // 3)
            for kk in range(n_restart):
                idx = sidx[pop_size - 1 - kk]
                r = np.random.random()
                if r < 0.4 and best_x is not None:
                    # Near best with adaptive scale
                    scale = 0.05 * (1.0 + np.random.random())
                    population[idx] = best_x + np.random.randn(dim) * ranges * scale
                elif r < 0.7 and best_x is not None:
                    # Gaussian around best with larger scale
                    population[idx] = best_x + np.random.randn(dim) * ranges * 0.2
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = eval_func(population[idx])

    # --- Phase 3: CMA-ES-like local search ---
    def mini_cma(x0, sigma0, time_budget):
        nonlocal best, best_x
        t_start = elapsed()
        n = len(x0)
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for efficiency in high dim
        if n > 20:
            diagC = np.ones(n)
            use_full = False
        else:
            C = np.eye(n)
            use_full = True
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        
        gen = 0
        while elapsed() - t_start < time_budget and time_left() > max_time * 0.005:
            gen += 1
            
            # Sample
            arz = np.random.randn(lam, n)
            if use_full:
                try:
                    sqrtC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.diag(np.diag(C))
                    try:
                        sqrtC = np.linalg.cholesky(C)
                    except:
                        break
                arx = xmean + sigma * (arz @ sqrtC.T)
            else:
                sqrtD = np.sqrt(np.maximum(diagC, 1e-20))
                arx = xmean + sigma * (arz * sqrtD)
            
            arx = np.array([clip_to_bounds(x) for x in arx])
            
            arfitness = np.array([eval_func(x) for x in arx])
            
            idx_sort = np.argsort(arfitness)
            arx = arx[idx_sort]
            arz = arz[idx_sort]
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2.0 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / (sigma + 1e-30)
            
            if use_full:
                artmp = (arx[:mu] - xold) / (sigma + 1e-30)
                C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (weights[:, None] * artmp).T @ artmp
                C = np.triu(C) + np.triu(C, 1).T
            else:
                artmp = (arx[:mu] - xold) / (sigma + 1e-30)
                diagC = (1 - c1 - cmu_val) * diagC + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diagC) + cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if sigma < 1e-16 * np.max(ranges):
                break
            
            if arfitness[0] >= arfitness[min(lam-1, int(0.7*lam))] - 1e-20:
                if gen > 10:
                    break

    # Run CMA-ES from best and top candidates
    sidx = np.argsort(fitness)
    starts = []
    if best_x is not None:
        starts.append(best_x.copy())
    for ii in range(min(3, pop_size)):
        c = population[sidx[ii]].copy()
        if not any(np.allclose(c, s, rtol=1e-6) for s in starts):
            starts.append(c)
    
    remaining = time_left()
    time_per_start = remaining * 0.85 / max(len(starts), 1)
    
    for sx in starts:
        if time_left() < max_time * 0.02:
            break
        sigma0 = 0.05 * np.max(ranges)
        mini_cma(sx, sigma0, min(time_per_start, time_left() * 0.8))
    
    # --- Phase 4: Final coordinate-wise refinement ---
    if best_x is not None and time_left() > max_time * 0.01:
        x = best_x.copy()
        fx = best
        step = 0.001 * ranges.copy()
        
        while time_left() > max_time * 0.002:
            improved = False
            for j in range(dim):
                if time_left() < max_time * 0.002:
                    return best
                trial = x.copy()
                trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
                ft = eval_func(trial)
                if ft < fx:
                    fx = ft; x = trial; step[j] *= 1.5; improved = True; continue
                trial = x.copy()
                trial[j] = np.clip(trial[j] - step[j], lower[j], upper[j])
                ft = eval_func(trial)
                if ft < fx:
                    fx = ft; x = trial; step[j] *= 1.5; improved = True; continue
                step[j] *= 0.5
            if not improved:
                if np.max(step / ranges) < 1e-15:
                    break
    
    return best
