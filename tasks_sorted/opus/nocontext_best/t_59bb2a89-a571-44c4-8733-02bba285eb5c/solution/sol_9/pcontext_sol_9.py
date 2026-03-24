#Looking at the current algorithm, it's already quite sophisticated with L-SHADE + local search + Nelder-Mead. Let me analyze potential improvements:
#
#1. **Better initial sampling** with Sobol-like quasi-random sequences
#2. **CMA-ES as the local search** instead of coordinate descent
#3. **Improved DE with rank-based weighting and better current-to-pbest**
#4. **More aggressive restarts on stagnation**
#5. **Better time allocation and adaptive phase switching**
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
    
    evals = [0]
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip_to_bounds(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Quasi-random initialization ---
    def sobol_like_init(n, d):
        # Use stratified sampling as a simple quasi-random approach
        pop = np.zeros((n, d))
        for j in range(d):
            perm = np.random.permutation(n)
            pop[:, j] = (perm + np.random.uniform(0, 1, n)) / n
        return pop

    init_pop_size = min(max(40, 15 * dim), 400)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
    population = sobol_like_init(pop_size, dim)
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.92:
            return best
        fitness[i] = eval_func(population[i])
    
    # Sort initially
    sidx = np.argsort(fitness)
    population = population[sidx]
    fitness = fitness[sidx]

    # --- L-SHADE with improved adaptation ---
    H = 10
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    max_archive = pop_size
    
    no_improve_count = 0
    prev_best = best
    generation = 0
    
    de_time_budget = max_time * 0.78
    
    while elapsed() < de_time_budget:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.11 * pop_size))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Pre-generate random parameters for speed
        ri_all = np.random.randint(0, H, pop_size)
        
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            ri = ri_all[i]
            
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
            
            # current-to-pbest/1
            p_best_idx = sorted_idx[np.random.randint(0, p_best_size)]
            
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            
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
            
            # Mutation: current-to-pbest/1
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
                M_CR[k] = -1.0
            else:
                M_CR[k] = mean_cr
            k = (k + 1) % H
        
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # Population size reduction
        fraction = elapsed() / de_time_budget
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * fraction)))
        
        if new_pop_size < pop_size:
            sidx2 = np.argsort(fitness)
            population = population[sidx2[:new_pop_size]]
            fitness = fitness[sidx2[:new_pop_size]]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation restart
        if no_improve_count > 40 and pop_size > min_pop_size + 4:
            no_improve_count = 0
            sidx2 = np.argsort(fitness)
            n_restart = pop_size // 3
            for kk in range(n_restart):
                idx = sidx2[pop_size - 1 - kk]
                if np.random.random() < 0.3:
                    population[idx] = lower + np.random.random(dim) * ranges
                else:
                    scale = np.random.exponential(0.1)
                    population[idx] = best_x + np.random.randn(dim) * ranges * scale
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = eval_func(population[idx])

    # --- Phase 3: CMA-ES local search from best ---
    def mini_cma_es(x0, sigma0, budget_frac):
        nonlocal best, best_x
        deadline = elapsed() + max(time_left() * budget_frac, 0.05)
        n = len(x0)
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        eigeneval = 0
        
        fmean = eval_func(xmean)
        
        while elapsed() < deadline:
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for ki in range(lam):
                arx[ki] = xmean + sigma * (B @ (D * arz[ki]))
                arx[ki] = clip_to_bounds(arx[ki])
            
            arfitness = np.zeros(lam)
            for ki in range(lam):
                if elapsed() >= deadline:
                    return
                arfitness[ki] = eval_func(arx[ki])
            
            idx_sort = np.argsort(arfitness)
            arx = arx[idx_sort]
            arz = arz[idx_sort]
            arfitness = arfitness[idx_sort]
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[:mu], axis=0)
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (eigeneval + 1))) / chiN < 1.4 + 2.0 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
            
            artmp = (arx[:mu] - xold) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges))
            
            eigeneval += 1
            if eigeneval % (lam / (c1 + cmu_val + 1e-30) / n / 10 + 1) < 1 or True:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            
            if sigma * np.max(D) < 1e-15 * np.max(ranges):
                break

    # Run CMA-ES from best
    if best_x is not None and time_left() > 0.1:
        sigma0 = 0.1 * np.max(ranges)
        mini_cma_es(best_x.copy(), sigma0, 0.6)
    
    # Run CMA-ES again with smaller sigma
    if best_x is not None and time_left() > 0.1:
        sigma0 = 0.01 * np.max(ranges)
        mini_cma_es(best_x.copy(), sigma0, 0.7)
    
    # --- Final coordinate-wise refinement ---
    if best_x is not None and time_left() > 0.05:
        x = best_x.copy()
        fx = best
        step = 0.001 * ranges
        while time_left() > 0.01:
            improved = False
            for j in range(dim):
                if time_left() < 0.01:
                    break
                for sign in [1, -1]:
                    trial = x.copy()
                    trial[j] = np.clip(trial[j] + sign * step[j], lower[j], upper[j])
                    ft = eval_func(trial)
                    if ft < fx:
                        fx = ft
                        x = trial
                        step[j] *= 1.5
                        improved = True
                        break
                else:
                    step[j] *= 0.5
            if not improved and np.max(step / ranges) < 1e-15:
                break
    
    return best
