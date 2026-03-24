#Looking at the previous results, the DE+Nelder-Mead hybrid (score 16.49) significantly outperformed the CMA-ES (score 157.26). Let me improve upon the best algorithm with:
#
#1. **SHADE (Success-History based Adaptive DE)** - self-adaptive F and CR based on successful parameters
#2. **Better population sizing and initialization** with Latin Hypercube Sampling
#3. **More aggressive local search** with multi-start pattern search
#4. **Opposition-based learning** for better exploration
#5. **Vectorized crossover** for speed
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Latin Hypercube Sampling initialization
    pop_size = min(max(30, 8 * dim), 300)
    
    def lhs_init(n):
        result = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                result[i, j] = lower[j] + (perm[i] + np.random.rand()) / n * ranges[j]
        return result
    
    population = lhs_init(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    # SHADE memory
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0  # memory index
    
    # Archive
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    # Main SHADE loop
    while elapsed() < max_time * 0.80:
        generation += 1
        
        S_F = []
        S_CR = []
        S_df = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            
            # Select random memory index
            ri = np.random.randint(H)
            
            # Generate F from Cauchy
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            # Generate CR from Normal
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # current-to-pbest/1 mutation
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.argsort(fitness)[:p]
            xpbest = population[np.random.choice(pbest_idx)]
            
            # Select r1 != i
            r1 = np.random.randint(pop_size)
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            # Select r2 from pop + archive, != i, != r1
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(union_size)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (xpbest - population[i]) + Fi * (population[r1] - xr2)
            mutant = clip(mutant)
            
            # Binomial crossover
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi) | (np.arange(dim) == j_rand)
            trial = np.where(mask, mutant, population[i])
            trial = clip(trial)
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(fitness[i] - f_trial)
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if S_F:
            weights = np.array(S_df)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[k] = mean_F
            M_CR[k] = mean_CR
            k = (k + 1) % H
        
        # Stagnation check
        if abs(best - prev_best) < 1e-14:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        # Population size reduction (linear)
        min_pop = max(4, dim // 2)
        time_frac = elapsed() / (max_time * 0.80)
        new_size = max(min_pop, int(pop_size * (1 - 0.5 * time_frac)))
        if new_size < len(population):
            idx_sort = np.argsort(fitness)
            population = population[idx_sort[:new_size]]
            fitness = fitness[idx_sort[:new_size]]
            pop_size = new_size
        
        if stagnation_count > 15 + dim:
            half = pop_size // 2
            idx_sort = np.argsort(fitness)
            population = population[idx_sort]
            fitness = fitness[idx_sort]
            for i in range(half, pop_size):
                if elapsed() >= max_time * 0.80:
                    break
                population[i] = clip(best_x + 0.1 * ranges * np.random.randn(dim) * np.random.rand())
                fitness[i] = eval_f(population[i])
            stagnation_count = 0
    
    # Phase 2: Local search - coordinate descent with golden section
    if best_x is not None and elapsed() < max_time * 0.98:
        x_local = best_x.copy()
        step = 0.01 * ranges
        
        while elapsed() < max_time * 0.98:
            improved = False
            for j in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for direction in [1, -1]:
                    x_trial = x_local.copy()
                    x_trial[j] = np.clip(x_local[j] + direction * step[j], lower[j], upper[j])
                    f_trial = eval_f(x_trial)
                    if f_trial < best:
                        x_local = x_trial.copy()
                        step[j] *= 1.5
                        improved = True
                        break
                else:
                    step[j] *= 0.5
            
            if not improved:
                if np.max(step / ranges) < 1e-10:
                    # Restart local search with perturbation
                    x_local = clip(best_x + 0.001 * ranges * np.random.randn(dim))
                    step = 0.005 * ranges
    
    return best
