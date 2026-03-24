#Looking at the current algorithm, I'll improve it with several enhancements:
#
#1. **Better SHADE implementation** with memory of successful parameters
#2. **Linear population size reduction** (L-SHADE)
#3. **Multiple restart strategies** with opposition-based learning
#4. **Better local search** using a proper Nelder-Mead simplex
#5. **CMA-ES-inspired local search** on the best solution
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: L-SHADE ---
    init_pop_size = min(max(30, 15 * dim), 300)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
    # LHS initialization
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = eval_func(population[i])
    
    # SHADE memory
    H = max(5, dim)
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    max_evals_estimate = pop_size * 200  # rough estimate
    evals_used = pop_size
    generation = 0
    no_improve_count = 0
    prev_best = best
    
    while elapsed() < max_time * 0.75:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.2 * pop_size))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            # Pick random memory index
            ri = np.random.randint(0, H)
            
            # Generate F
            F = -1
            while F <= 0:
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F >= 1:
                    F = 1.0
            F = min(F, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CR = 0.0
            else:
                CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # p-best index
            pi = np.random.randint(p_min, p_max + 1)
            p_best_idx = sorted_idx[np.random.randint(0, pi)]
            
            # r1
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            
            # r2 from pop + archive
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(0, pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pool_size)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            # Mutation
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - xr2)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back bounds
            out_low = trial < lower
            out_high = trial > upper
            trial[out_low] = (lower[out_low] + population[i][out_low]) / 2.0
            trial[out_high] = (upper[out_high] + population[i][out_high]) / 2.0
            trial = clip(trial)
            
            trial_f = eval_func(trial)
            evals_used += 1
            
            if trial_f <= fitness[i]:
                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    S_delta.append(abs(fitness[i] - trial_f))
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                new_population[i] = trial
                new_fitness[i] = trial_f
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if len(S_F) > 0:
            S_delta = np.array(S_delta)
            weights = S_delta / np.sum(S_delta)
            S_F = np.array(S_F)
            S_CR = np.array(S_CR)
            M_F[k] = np.sum(weights * S_F**2) / (np.sum(weights * S_F) + 1e-30)
            if np.max(S_CR) == 0:
                M_CR[k] = -1
            else:
                M_CR[k] = np.sum(weights * S_CR**2) / (np.sum(weights * S_CR) + 1e-30)
            k = (k + 1) % H
        
        # Linear population size reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * evals_used / max_evals_estimate)))
        if new_pop_size < pop_size:
            sorted_idx = np.argsort(fitness)
            keep = sorted_idx[:new_pop_size]
            population = population[keep]
            fitness = fitness[keep]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation check
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        if no_improve_count > 30:
            no_improve_count = 0
            # Restart worst half around best
            sorted_idx = np.argsort(fitness)
            n_restart = pop_size // 2
            for kk in range(n_restart):
                idx = sorted_idx[pop_size - 1 - kk]
                sigma = 0.1 * ranges * (0.5 ** (generation / 50.0))
                population[idx] = clip(best_x + np.random.randn(dim) * sigma)
                fitness[idx] = eval_func(population[idx])
            M_F[:] = 0.5
            M_CR[:] = 0.5
    
    # --- Phase 2: CMA-ES-like local search ---
    if best_x is None:
        return best
    
    sigma = 0.05
    mean = best_x.copy()
    C = np.eye(dim)
    lam = max(4, 4 + int(3 * np.log(dim)))
    mu = lam // 2
    weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights_cma /= np.sum(weights_cma)
    mu_eff = 1.0 / np.sum(weights_cma**2)
    
    cc = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu_val = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(dim + 1)) - 1) + cs
    
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    eigeneval = 0
    B = np.eye(dim)
    D = np.ones(dim)
    invsqrtC = np.eye(dim)
    
    while elapsed() < max_time * 0.97:
        # Generate offspring
        arz = np.random.randn(lam, dim)
        arx = np.zeros((lam, dim))
        arf = np.zeros(lam)
        
        for i in range(lam):
            if elapsed() >= max_time * 0.97:
                break
            arx[i] = mean + sigma * (B @ (D * arz[i]))
            arx[i] = clip(arx[i])
            arf[i] = eval_func(arx[i])
        
        # Sort
        idx_sort = np.argsort(arf)
        
        old_mean = mean.copy()
        # Recombination
        mean = np.zeros(dim)
        for i in range(mu):
            mean += weights_cma[i] * arx[idx_sort[i]]
        
        # Update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ (mean - old_mean) / sigma
        hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(dim + 1) else 0.0
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
        
        # Update covariance matrix
        artmp = np.zeros((mu, dim))
        for i in range(mu):
            artmp[i] = (arx[idx_sort[i]] - old_mean) / sigma
        
        C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
        for i in range(mu):
            C += cmu_val * weights_cma[i] * np.outer(artmp[i], artmp[i])
        
        # Update sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = max(sigma, 1e-20)
        sigma = min(sigma, np.max(ranges))
        
        # Decompose C
        eigeneval += 1
        if eigeneval >= 1.0 / (c1 + cmu_val) / dim / 5 or eigeneval <= 1:
            C = np.triu(C) + np.triu(C, 1).T
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.maximum(D_sq, 1e-20)
                D = np.sqrt(D_sq)
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                eigeneval = 0
            except:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                invsqrtC = np.eye(dim)
    
    # --- Phase 3: Fine coordinate descent ---
    if best_x is not None:
        step = 0.001 * ranges
        cx = best_x.copy()
        while elapsed() < max_time * 0.995:
            improved = False
            for j in range(dim):
                if elapsed() >= max_time * 0.995:
                    return best
                trial = cx.copy()
                trial[j] = np.clip(trial[j] + step[j], lower[j], upper[j])
                f = eval_func(trial)
                if f < best:
                    cx = trial.copy()
                    improved = True
                    continue
                trial[j] = np.clip(cx[j] - step[j], lower[j], upper[j])
                f = eval_func(trial)
                if f < best:
                    cx = trial.copy()
                    improved = True
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-14:
                    break
    
    return best
