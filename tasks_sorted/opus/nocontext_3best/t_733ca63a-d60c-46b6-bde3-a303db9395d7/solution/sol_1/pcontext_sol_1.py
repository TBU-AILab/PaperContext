

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    range_width = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok():
        return elapsed() < max_time * 0.93
    
    def eval_f(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    best_solution = None
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    pop_size = min(max(15 * dim, 40), 150)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * range_width
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if not time_ok():
            return best
        fitness[i] = eval_f(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: Hybrid DE with CMA-ES local search ---
    
    # DE parameters with self-adaptation (SHADE-like)
    memory_size = 5
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    memory_idx = 0
    
    archive = []
    archive_max = pop_size
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    # CMA-ES local search function
    def cma_local_search(x0, sigma0, max_evals):
        nonlocal best
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for efficiency
        C_diag = np.ones(n)
        evals_used = 0
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        best_local = func(xmean)
        evals_used += 1
        if best_local < best:
            best = best_local
        best_local_x = xmean.copy()
        
        while evals_used < max_evals and time_ok():
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fit_local = np.zeros(lam)
            
            for k in range(lam):
                if not time_ok():
                    return best_local_x, best_local, evals_used
                arx[k] = xmean + sigma * np.sqrt(C_diag) * arz[k]
                arx[k] = np.clip(arx[k], lower, upper)
                fit_local[k] = eval_f(arx[k])
                evals_used += 1
                if fit_local[k] < best_local:
                    best_local = fit_local[k]
                    best_local_x = arx[k].copy()
            
            idx_sort = np.argsort(fit_local)
            
            # Recombination
            xold = xmean.copy()
            xmean = np.zeros(n)
            zmean = np.zeros(n)
            for k in range(mu):
                xmean += weights[k] * arx[idx_sort[k]]
                zmean += weights[k] * arz[idx_sort[k]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(evals_used/lam+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
            
            # Diagonal covariance update
            C_diag = ((1 - c1 - cmu_val) * C_diag +
                      c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag))
            for k in range(mu):
                diff = (arx[idx_sort[k]] - xold) / sigma
                C_diag += cmu_val * weights[k] * diff**2
            
            C_diag = np.maximum(C_diag, 1e-20)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(range_width))
            
            if sigma < 1e-16:
                break
        
        return best_local_x, best_local, evals_used
    
    while time_ok():
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Select parameters from memory
        ri = np.random.randint(0, memory_size)
        
        for i in range(pop_size):
            if not time_ok():
                return best
            
            # Generate F and CR from memory (Cauchy and Normal respectively)
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # DE/current-to-pbest/1 with archive
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.randint(0, p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            sorted_by_fit = np.argsort(fitness)
            x_pbest = population[sorted_by_fit[pbest_idx]]
            
            mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Boundary handling - midpoint
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2.0
            trial[above] = (upper[above] + population[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            trial_fitness = eval_f(trial)
            
            if trial_fitness <= fitness[i]:
                delta = fitness[i] - trial_fitness
                if delta > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                
                # Add old to archive
                if len(archive) < archive_max:
                    archive.append(population[i].copy())
                else:
                    replace_idx = np.random.randint(0, archive_max)
                    archive[replace_idx] = population[i].copy()
                
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if len(S_F) > 0:
            S_delta_arr = np.array(S_delta)
            w = S_delta_arr / S_delta_arr.sum()
            M_F[memory_idx] = np.sum(w * np.array(S_F)**2) / np.sum(w * np.array(S_F))
            M_CR[memory_idx] = np.sum(w * np.array(S_CR))
            memory_idx = (memory_idx + 1) % memory_size
        
        # Stagnation check
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Periodic CMA-ES local search
        if generation % 10 == 0 and time_ok():
            remaining_time = max_time * 0.93 - elapsed()
            if remaining_time > max_time * 0.05:
                sigma0 = 0.1 * np.max(range_width)
                max_cma_evals = max(pop_size * 2, int(remaining_time / (elapsed() / max(1, generation * pop_size)) * 0.3))
                max_cma_evals = min(max_cma_evals, 500 * dim)
                cma_x, cma_f, _ = cma_local_search(best_solution.copy(), sigma0, max_cma_evals)
        
        # Restart on stagnation
        if stagnation > 20 + dim:
            n_keep = max(3, pop_size // 5)
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            for i in range(n_keep, pop_size):
                if not time_ok():
                    return best
                if np.random.random() < 0.4:
                    # Local around best
                    scale = 0.05 * range_width * (0.5 + 0.5 * np.random.random())
                    population[i] = best_solution + scale * np.random.randn(dim)
                    population[i] = np.clip(population[i], lower, upper)
                else:
                    population[i] = lower + np.random.random(dim) * range_width
                
                fitness[i] = eval_f(population[i])
            
            # Reset memory
            M_F = np.full(memory_size, 0.5)
            M_CR = np.full(memory_size, 0.5)
            archive = []
            stagnation = 0
    
    # Final local refinement with CMA-ES
    if time_ok() and best_solution is not None:
        remaining = max_time * 0.93 - elapsed()
        if remaining > 0.5:
            sigma0 = 0.01 * np.max(range_width)
            max_evals = max(100, int(remaining / max(1e-6, elapsed() / max(1, generation * pop_size)) * 0.8))
            max_evals = min(max_evals, 1000 * dim)
            cma_local_search(best_solution.copy(), sigma0, max_evals)
    
    return best