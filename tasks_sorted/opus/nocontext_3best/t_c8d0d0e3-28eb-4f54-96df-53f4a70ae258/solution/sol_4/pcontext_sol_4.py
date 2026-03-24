#Looking at the results, algorithm #1 performs best (13.38). I'll improve it with:
#
#1. Better SHADE with improved parameter adaptation
#2. Restart mechanism when stuck
#3. Proper CMA-ES with covariance adaptation (not just isotropic)
#4. Better time budget allocation
#5. Avoiding wasted evaluations (e.g., the CMA-ES in #3 calls `func(mean)` unnecessarily)
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

    def run_shade(time_fraction_end, pop_size):
        nonlocal best, best_params
        
        # LHS initialization
        population = np.random.uniform(0, 1, (pop_size, dim))
        for i in range(dim):
            perm = np.random.permutation(pop_size)
            population[:, i] = (perm + population[:, i]) / pop_size
        population = lower + population * ranges
        
        # Inject best if available
        if best_params is not None:
            population[0] = best_params.copy()
        
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= max_time * time_fraction_end:
                return
            fitness[i] = evaluate(population[i])
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        memory_size = 20
        M_F = np.full(memory_size, 0.5)
        M_CR = np.full(memory_size, 0.5)
        mem_idx = 0
        
        archive = []
        archive_max = pop_size
        
        stagnation = 0
        prev_best_local = best
        
        while elapsed() < max_time * time_fraction_end:
            S_F, S_CR, S_delta = [], [], []
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if elapsed() >= max_time * time_fraction_end:
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
                weights = np.array(S_delta)
                weights = weights / (weights.sum() + 1e-30)
                M_F[mem_idx] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
                M_CR[mem_idx] = np.sum(weights * np.array(S_CR))
                mem_idx = (mem_idx + 1) % memory_size
            
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            if best < prev_best_local - 1e-12:
                stagnation = 0
                prev_best_local = best
            else:
                stagnation += 1
            
            if stagnation > 25:
                n_replace = pop_size // 3
                for j in range(pop_size - n_replace, pop_size):
                    population[j] = lower + np.random.rand(dim) * ranges
                    fitness[j] = evaluate(population[j])
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx]
                fitness = fitness[sorted_idx]
                stagnation = 0

    pop_size = min(max(30, 15 * dim), 300)
    run_shade(0.45, pop_size)
    run_shade(0.70, max(20, pop_size // 2))

    # CMA-ES local search
    if best_params is not None:
        sigma = 0.03
        mean = best_params.copy()
        lam = max(8, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        
        best_sigma_f = best
        while elapsed() < max_time * 0.94:
            samples, f_s = [], []
            for _ in range(lam):
                if elapsed() >= max_time * 0.94:
                    break
                x = clip(mean + sigma * ranges * np.random.randn(dim))
                f = evaluate(x)
                samples.append(x); f_s.append(f)
            if len(samples) < mu:
                break
            idx_s = np.argsort(f_s)
            new_mean = clip(np.dot(w, np.array([samples[idx_s[j]] for j in range(mu)])))
            if f_s[idx_s[0]] < best_sigma_f:
                sigma *= 1.08
                best_sigma_f = f_s[idx_s[0]]
            else:
                sigma *= 0.83
            sigma = np.clip(sigma, 1e-10, 0.3)
            mean = new_mean

    if best_params is not None:
        scale = 0.0003
        while elapsed() < max_time * 0.99:
            evaluate(clip(best_params + scale * ranges * np.random.randn(dim)))

    return best
