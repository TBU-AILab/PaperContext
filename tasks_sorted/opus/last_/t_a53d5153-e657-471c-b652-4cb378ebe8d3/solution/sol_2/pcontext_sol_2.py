#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better DE strategy**: Use L-SHADE with population size reduction
#2. **More aggressive local search**: Use CMA-ES-like covariance adaptation instead of Nelder-Mead
#3. **Better time allocation**: More time for exploitation
#4. **Multiple restarts of local search** with adaptive step sizes
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
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS initialization ---
    pop_size = min(max(10 * dim, 40), 200)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: L-SHADE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    archive_max = pop_size
    init_pop_size = pop_size
    min_pop_size = max(4, dim)
    n_evals = pop_size
    max_evals_estimate = pop_size * 500  # rough estimate
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    while elapsed() < max_time * 0.65:
        generation += 1
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F from Cauchy
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CRi = 0.0
            else:
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # current-to-pbest/1
            p_rate = max(2.0/pop_size, 0.05 + 0.15 * (1 - n_evals/max_evals_estimate))
            p = max(2, int(np.ceil(p_rate * pop_size)))
            pbest_idx = np.random.randint(0, p)
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(0, pool_size - 1)
            r2_candidates = list(range(pool_size))
            if i in r2_candidates:
                r2_candidates.remove(i)
            if r1 in r2_candidates:
                r2_candidates.remove(r1)
            if len(r2_candidates) == 0:
                r2_candidates = [r1]
            r2_idx = np.random.choice(r2_candidates)
            
            xr2 = population[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = eval_f(trial)
            n_evals += 1
            
            if trial_fitness < new_fitness[i]:
                archive.append(population[i].copy())
                if len(archive) > archive_max:
                    archive.pop(np.random.randint(len(archive)))
                delta = fitness[i] - trial_fitness
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(delta)
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_cr = np.sum(weights * np.array(S_CR))
            M_CR[k] = mean_cr if max(S_CR) > 0 else -1
            k = (k + 1) % H
        
        # LPSR: Linear population size reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * (elapsed() / (max_time * 0.65)))))
        if new_pop_size < pop_size:
            population = population[:new_pop_size]
            fitness = fitness[:new_pop_size]
            pop_size = new_pop_size
            archive_max = pop_size
        
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 30:
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = best_params + 0.1 * ranges * np.random.randn(dim)
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = eval_f(population[i])
                n_evals += 1
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: CMA-ES-like local search ---
    sigma = 0.02 * np.mean(ranges)
    mean = best_params.copy()
    C = np.diag((0.02 * ranges) ** 2)
    lam = max(4 + int(3 * np.log(dim)), 8)
    mu = lam // 2
    weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights_cma /= np.sum(weights_cma)
    mu_eff = 1.0 / np.sum(weights_cma ** 2)
    
    c_cov = min(1.0, 2.0 / (dim ** 2))
    
    while elapsed() < max_time * 0.92:
        try:
            L = np.linalg.cholesky(C)
        except:
            C = np.diag(np.diag(C)) + 1e-10 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.diag((0.02 * ranges) ** 2)
                L = np.linalg.cholesky(C)
        
        offspring = np.zeros((lam, dim))
        f_off = np.zeros(lam)
        for i in range(lam):
            if elapsed() >= max_time * 0.92:
                break
            z = np.random.randn(dim)
            offspring[i] = mean + sigma * L @ z
            offspring[i] = np.clip(offspring[i], lower, upper)
            f_off[i] = eval_f(offspring[i])
        
        order = np.argsort(f_off)
        selected = offspring[order[:mu]]
        
        old_mean = mean.copy()
        mean = np.sum(weights_cma[:, None] * selected, axis=0)
        
        diff = mean - old_mean
        C = (1 - c_cov) * C + c_cov * np.outer(diff, diff) / (sigma ** 2 + 1e-30)
        
        # Adapt sigma
        if f_off[order[0]] < best:
            sigma *= 1.1
        else:
            sigma *= 0.85
        sigma = max(sigma, 1e-15 * np.mean(ranges))
        sigma = min(sigma, 0.5 * np.mean(ranges))
        
        mean = np.clip(mean, lower, upper)
    
    # --- Phase 4: Fine-grained random local search ---
    scales = [0.01, 0.001, 0.0001]
    while elapsed() < max_time * 0.998:
        for s in scales:
            if elapsed() >= max_time * 0.998:
                break
            candidate = best_params + s * ranges * np.random.randn(dim)
            candidate = np.clip(candidate, lower, upper)
            eval_f(candidate)
    
    return best
