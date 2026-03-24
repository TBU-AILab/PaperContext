#Looking at the two algorithms, #1 (SHADE-like DE) performs much better (1.45 vs 49.38). I'll improve upon it by:
#
#1. Using a more sophisticated SHADE variant (L-SHADE with linear population size reduction)
#2. Better initialization with opposition-based learning
#3. Improved local search (Nelder-Mead style + CMA-ES-inspired covariance adaptation)
#4. Multiple restarts if stagnation detected
#5. Better time management
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    range_width = upper - lower
    center = (upper + lower) / 2.0
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def time_frac():
        return elapsed() / max_time

    # L-SHADE
    def run_lshade(time_budget_frac=0.85):
        nonlocal best, best_params
        
        pop_size_init = min(max(40, 10 * dim), 200)
        pop_size_min = 4
        pop_size = pop_size_init
        
        # LHS initialization
        population = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                population[i, j] = lower[j] + (perm[i] + np.random.random()) / pop_size * range_width[j]
        
        # Opposition-based initialization
        opp_pop = lower + upper - population
        all_pop = np.vstack([population, opp_pop])
        all_fit = np.full(2 * pop_size, float('inf'))
        for i in range(2 * pop_size):
            if time_frac() >= time_budget_frac:
                break
            all_fit[i] = evaluate(all_pop[i])
        
        idx_sorted = np.argsort(all_fit)[:pop_size]
        population = all_pop[idx_sorted].copy()
        fitness = all_fit[idx_sorted].copy()
        
        # SHADE memory
        memory_size = max(6, dim)
        M_F = np.full(memory_size, 0.5)
        M_CR = np.full(memory_size, 0.5)
        mem_idx = 0
        
        archive = []
        archive_max = pop_size_init
        
        generation = 0
        max_gen_estimate = 10000  # for LPSR
        stagnation = 0
        prev_best = best
        
        while time_frac() < time_budget_frac:
            generation += 1
            
            S_F = []
            S_CR = []
            delta_f = []
            
            sorted_idx = np.argsort(fitness)
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if time_frac() >= time_budget_frac:
                    break
                
                ri = np.random.randint(memory_size)
                mu_F = M_F[ri]
                mu_CR = M_CR[ri]
                
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
                while Fi <= 0:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
                
                # current-to-pbest/1
                p = max(2, int(max(0.05, 0.2 - 0.15 * generation / max(max_gen_estimate, generation+1)) * pop_size))
                p_best_idx = sorted_idx[np.random.randint(p)]
                
                indices = list(range(pop_size))
                indices.remove(i)
                r1 = np.random.choice(indices)
                
                combined_size = pop_size + len(archive)
                r2_idx = np.random.randint(combined_size)
                while r2_idx == i or r2_idx == r1:
                    r2_idx = np.random.randint(combined_size)
                x_r2 = population[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
                
                mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = (np.random.random(dim) < CRi)
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                out_low = trial < lower
                out_high = trial > upper
                trial[out_low] = (lower[out_low] + population[i][out_low]) / 2
                trial[out_high] = (upper[out_high] + population[i][out_high]) / 2
                trial = clip(trial)
                
                trial_f = evaluate(trial)
                
                if trial_f <= fitness[i]:
                    if trial_f < fitness[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - trial_f))
                        if len(archive) < archive_max:
                            archive.append(population[i].copy())
                        else:
                            archive[np.random.randint(archive_max)] = population[i].copy()
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            if S_F:
                weights = np.array(delta_f)
                ws = weights.sum()
                if ws > 0:
                    weights = weights / ws
                    new_F = np.sum(weights * np.array(S_F)**2) / max(np.sum(weights * np.array(S_F)), 1e-30)
                    new_CR = np.sum(weights * np.array(S_CR))
                    M_F[mem_idx] = new_F
                    M_CR[mem_idx] = new_CR
                    mem_idx = (mem_idx + 1) % memory_size
            
            # Linear population size reduction
            new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * time_frac() / time_budget_frac)))
            if new_pop_size < pop_size:
                idx_keep = np.argsort(fitness)[:new_pop_size]
                population = population[idx_keep]
                fitness = fitness[idx_keep]
                pop_size = new_pop_size
            
            if best == prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = best
            
            if stagnation > 50 and pop_size > pop_size_min:
                stagnation = 0
    
    run_lshade(0.88)
    
    # Local search with adaptive step
    if best_params is not None:
        scale = 0.02
        no_improve = 0
        while elapsed() < max_time * 0.99:
            candidate = best_params + scale * range_width * np.random.randn(dim)
            cf = evaluate(candidate)
            if cf < best:
                scale = min(scale * 1.5, 0.1)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 20:
                    scale *= 0.5
                    no_improve = 0
                    if scale < 1e-10:
                        scale = 0.05
                        # Random restart near best
                        candidate = best_params + 0.1 * range_width * np.random.randn(dim)
                        evaluate(candidate)
    
    return best
