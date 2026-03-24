#Looking at the results, algorithm #1 (score 13.38) is the best. I'll improve it with:
#
#1. **Proper CMA-ES** with sep-CMA for high dimensions
#2. **Better L-SHADE** with weighted Lehmer mean
#3. **Multi-restart CMA-ES** with decreasing sigma
#4. **Adaptive local search** with coordinate-wise refinement
#5. **Better time budgeting**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size_init = min(max(40, 15 * dim), 350)
    pop_size = pop_size_init
    min_pop_size = max(6, dim)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: L-SHADE ---
    memory_size = 25
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size_init
    stagnation = 0
    prev_best = best
    generation = 0
    t_before_de = elapsed()
    de_time_frac = 0.45
    
    while elapsed() < max_time * de_time_frac:
        generation += 1
        S_F, S_CR, S_delta = [], [], []
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_frac:
                break
            
            ri = np.random.randint(0, memory_size)
            mu_F, mu_CR = M_F[ri], M_CR[ri]
            
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(pop_size * np.random.uniform(0.05, 0.2)))
            p_best_idx = np.random.randint(0, p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial_fitness = evaluate(trial)
            
            if trial_fitness <= fitness[i]:
                if trial_fitness < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(abs(fitness[i] - trial_fitness))
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        if S_F:
            wts = np.array(S_delta)
            wts = wts / (wts.sum() + 1e-30)
            M_F[mem_idx] = np.sum(wts * np.array(S_F)**2) / (np.sum(wts * np.array(S_F)) + 1e-30)
            M_CR[mem_idx] = np.sum(wts * np.array(S_CR))
            mem_idx = (mem_idx + 1) % memory_size
        
        ratio = min(1.0, (elapsed() - t_before_de) / (max_time * de_time_frac - t_before_de + 1e-10))
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        sorted_idx = np.argsort(fitness)
        if new_pop_size < pop_size:
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        else:
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        if stagnation > 30:
            n_replace = max(1, pop_size // 3)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0

    # --- Phase 3: Sep-CMA-ES with restarts ---
    for restart in range(5):
        if elapsed() >= max_time * 0.93 or best_params is None:
            break
        sigma = 0.1 * (0.5 ** restart)
        mean = best_params.copy() if restart == 0 else clip(best_params + 0.05 * ranges * np.random.randn(dim))
        n = dim
        lam = max(10, 4 + int(3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1))-1) + c_s
        chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        diag_C = np.ones(n)
        ps = np.zeros(n)
        no_imp = 0
        phase_limit = max_time * (0.93 - 0.12*restart)
        while elapsed() < phase_limit:
            samps, fs = [], []
            for _ in range(lam):
                if elapsed() >= phase_limit: break
                z = np.random.randn(n)
                x = clip(mean + sigma * np.sqrt(diag_C) * ranges * z)
                f = evaluate(x)
                samps.append(x); fs.append(f)
            if len(samps) < mu: break
            idx_s = np.argsort(fs)
            sel = np.array([samps[idx_s[j]] for j in range(mu)])
            old_mean = mean.copy()
            mean = clip(np.dot(w, sel))
            diff = (mean - old_mean) / (sigma * np.sqrt(diag_C) * ranges + 1e-30)
            ps = (1-c_s)*ps + np.sqrt(c_s*(2-c_s)*mu_eff)*diff
            sigma *= np.exp((c_s/d_s)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-14, 1.0)
            artmp = (sel - old_mean) / (sigma * ranges + 1e-30)
            diag_C = 0.8*diag_C + 0.2*np.sum(w[:,None]*artmp**2, axis=0)
            diag_C = np.clip(diag_C, 1e-20, 1e6)
            if best < prev_best - 1e-12: no_imp=0; prev_best=best
            else: no_imp += 1
            if sigma < 1e-12 or no_imp > 40+5*n: break

    # --- Phase 4: Fine local search ---
    if best_params is not None:
        scale = 0.0005
        while elapsed() < max_time * 0.998:
            candidate = clip(best_params + scale * ranges * np.random.randn(dim))
            old_b = best
            evaluate(candidate)
            if best >= old_b: scale *= 0.97
            else: scale *= 1.2
            scale = np.clip(scale, 1e-10, 0.005)
    
    return best
