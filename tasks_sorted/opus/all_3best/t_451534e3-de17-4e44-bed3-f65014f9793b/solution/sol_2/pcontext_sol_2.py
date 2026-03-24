#Looking at the previous results, the SHADE algorithm scored 25.28. I'll improve by implementing L-SHADE (Linear population size reduction SHADE) with a more aggressive local search phase, better population sizing, and a CMA-ES-inspired local search around the best solution during restarts.
#
#Key improvements:
#1. **L-SHADE with linear population reduction** for better convergence
#2. **Weighted recombination local search** (simplified CMA-ES-like)
#3. **More aggressive p-best range** that adapts over time
#4. **Periodic local search** using coordinate descent around the best solution
#5. **Faster numpy-vectorized operations** where possible
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_sol = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_sol, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_sol = x.copy()
        return f

    def local_search(x0, f0, sigma=0.01, max_evals=None):
        """Nelder-Mead-like simplex local search"""
        nonlocal best, best_sol
        if max_evals is None:
            max_evals = dim * 10
        
        x_cur = x0.copy()
        f_cur = f0
        n_evals = 0
        step = sigma * ranges
        
        for iteration in range(max(1, max_evals // (2 * dim))):
            improved = False
            for d in range(dim):
                if remaining() <= 0:
                    return x_cur, f_cur
                
                # Try positive direction
                x_try = x_cur.copy()
                x_try[d] += step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    step[d] *= 1.2
                    improved = True
                    continue
                
                # Try negative direction
                x_try = x_cur.copy()
                x_try[d] -= step[d]
                x_try = clip(x_try)
                f_try = eval_func(x_try)
                n_evals += 1
                
                if f_try < f_cur:
                    x_cur = x_try
                    f_cur = f_try
                    step[d] *= 1.2
                    improved = True
                else:
                    step[d] *= 0.5
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-12:
                    break
        
        return x_cur, f_cur

    # ============= Main L-SHADE loop with restarts =============
    restart_count = 0
    
    while remaining() > 0:
        restart_count += 1
        
        # L-SHADE parameters
        N_init = min(max(30, 10 * dim), 300)
        N_min = max(4, dim)
        pop_size = N_init
        max_nfe_estimate = int(remaining() * 500)  # rough estimate
        nfe_at_start = evals
        
        H = 100
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.5)
        if restart_count > 1:
            memory_F = np.full(H, 0.3 + 0.4 * np.random.rand())
            memory_CR = np.full(H, 0.3 + 0.4 * np.random.rand())
        k_idx = 0
        
        archive = []
        archive_max = N_init
        
        # Initialize population
        if restart_count == 1:
            population = np.random.uniform(lower, upper, (pop_size, dim))
        else:
            # Restart with some solutions near best
            population = np.random.uniform(lower, upper, (pop_size, dim))
            if best_sol is not None:
                n_local = max(1, pop_size // 4)
                for j in range(n_local):
                    scale = 0.1 * (0.5 ** (restart_count - 2))
                    population[j] = clip(best_sol + scale * ranges * np.random.randn(dim))
        
        fitness = np.array([eval_func(ind) for ind in population])
        
        if remaining() <= 0:
            return best
        
        generation = 0
        stagnation = 0
        prev_best = best
        
        while remaining() > 0:
            generation += 1
            
            # Linear population size reduction
            nfe_since_start = evals - nfe_at_start
            ratio = min(1.0, nfe_since_start / max(1, max_nfe_estimate))
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                # Reduce population keeping best
                si = np.argsort(fitness)
                population = population[si[:new_pop_size]]
                fitness = fitness[si[:new_pop_size]]
                pop_size = new_pop_size
            
            # Adaptive p value (decreases over time for more exploitation)
            p_min = 2.0 / pop_size
            p_max = 0.2
            p = p_max - (p_max - p_min) * ratio
            p_best_size = max(2, int(p * pop_size))
            
            # Generate F and CR from memory
            ri = np.random.randint(0, H, pop_size)
            
            # Cauchy distribution for F
            Fs = np.empty(pop_size)
            for idx in range(pop_size):
                while True:
                    f_val = memory_F[ri[idx]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        Fs[idx] = min(f_val, 1.0)
                        break
            
            CRs = np.clip(memory_CR[ri] + 0.1 * np.random.randn(pop_size), 0.0, 1.0)
            
            S_F, S_CR, S_delta = [], [], []
            
            sorted_idx = np.argsort(fitness)
            
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if remaining() <= 0:
                    return best
                
                # p-best index
                pi = sorted_idx[np.random.randint(0, p_best_size)]
                
                # Select r1 != i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                
                # r2 from population + archive, != i, != r1
                combined_size = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(combined_size)
                
                if r2 < pop_size:
                    x_r2 = population[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                # Mutation: current-to-pbest/1
                mutant = population[i] + Fs[i] * (population[pi] - population[i]) + Fs[i] * (population[r1] - x_r2)
                
                # Binomial crossover
                jrand = np.random.randint(dim)
                mask = (np.random.rand(dim) < CRs[i])
                mask[jrand] = True
                trial = np.where(mask, mutant, population[i])
                
                # Boundary handling
                below = trial < lower
                above = trial > upper
                trial[below] = (lower[below] + population[i][below]) / 2.0
                trial[above] = (upper[above] + population[i][above]) / 2.0
                
                trial_f = eval_func(trial)
                
                if trial_f <= fitness[i]:
                    delta = fitness[i] - trial_f
                    if trial_f < fitness[i]:
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(Fs[i])
                        S_CR.append(CRs[i])
                        S_delta.append(delta + 1e-30)
                    new_population[i] = trial
                    new_fitness[i] = trial_f
            
            population = new_population
            fitness = new_fitness
            
            # Update memory (weighted Lehmer mean)
            if S_F:
                weights = np.array(S_delta)
                weights /= weights.sum()
                sf = np.array(S_F)
                scr = np.array(S_CR)
                mean_F = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
                mean_CR = np.sum(weights * scr)
                memory_F[k_idx % H] = mean_F
                memory_CR[k_idx % H] = mean_CR
                k_idx += 1
            
            # Stagnation check
            if abs(prev_best - best) < 1e-15:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = best
            
            # Periodic local search
            if generation % 50 == 0 and best_sol is not None and remaining() > 1:
                sigma = 0.01 * max(0.1, 1.0 - ratio)
                best_sol, _ = local_search(best_sol, best, sigma=sigma, max_evals=dim*5)
            
            if stagnation > 60 or pop_size <= N_min:
                # Local search before restart
                if best_sol is not None and remaining() > 1:
                    best_sol, _ = local_search(best_sol, best, sigma=0.005, max_evals=dim*10)
                break
    
    return best
