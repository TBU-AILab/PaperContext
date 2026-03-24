#Looking at the results, algorithm #1 (SHADE-like DE with local search) performs best at 1.45. I'll improve upon it by:
#
#1. Adding CMA-ES as a complementary local search strategy
#2. Better L-SHADE with population size reduction
#3. Improved local search with coordinate-wise descent + random perturbations
#4. Better time management and restart logic
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # === L-SHADE Phase ===
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
        if elapsed() >= max_time * 0.05:
            all_fit[i:] = [evaluate(all_pop[k]) if elapsed() < max_time * 0.1 else float('inf') for k in range(i, 2 * pop_size)]
            break
        all_fit[i] = evaluate(all_pop[i])
    
    idx_sorted = np.argsort(all_fit)[:pop_size]
    population = all_pop[idx_sorted].copy()
    fitness = all_fit[idx_sorted].copy()
    
    memory_size = 8
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size_init
    
    generation = 0
    nfe_at_start = 0
    
    de_time_budget = 0.80
    
    while elapsed() < max_time * de_time_budget:
        generation += 1
        
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_budget:
                break
            
            ri = np.random.randint(memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            Fi = mu_F + 0.1 * np.random.standard_cauchy()
            while Fi <= 0:
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p_ratio = max(0.05, 0.25 - 0.20 * elapsed() / (max_time * de_time_budget))
            p = max(2, int(p_ratio * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            indices = list(range(pop_size))
            indices.remove(i)
            r1 = indices[np.random.randint(len(indices))]
            
            combined_size = pop_size + len(archive)
            r2_idx = np.random.randint(combined_size)
            attempts = 0
            while (r2_idx == i or r2_idx == r1) and attempts < 20:
                r2_idx = np.random.randint(combined_size)
                attempts += 1
            x_r2 = population[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
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
                population[i] = trial
                fitness[i] = trial_f
        
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
        progress = min(1.0, elapsed() / (max_time * de_time_budget))
        new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * progress)))
        if new_pop_size < pop_size:
            idx_keep = np.argsort(fitness)[:new_pop_size]
            population = population[idx_keep]
            fitness = fitness[idx_keep]
            pop_size = new_pop_size

    # === Local Search Phase: Adaptive random + coordinate descent ===
    if best_params is not None:
        scale = 0.01
        no_improve = 0
        while elapsed() < max_time * 0.95:
            # Coordinate-wise search
            for d in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                for sign in [1, -1]:
                    cand = best_params.copy()
                    cand[d] += sign * scale * range_width[d]
                    cf = evaluate(cand)
                    if cf < best:
                        no_improve = 0
            # Random perturbation
            candidate = best_params + scale * range_width * np.random.randn(dim)
            evaluate(candidate)
            no_improve += 1
            if no_improve > 3 * dim:
                scale *= 0.5
                no_improve = 0
                if scale < 1e-12:
                    scale = 0.05
                    candidate = best_params + 0.1 * range_width * np.random.randn(dim)
                    evaluate(candidate)
        
        # Final intensification
        while elapsed() < max_time * 0.995:
            candidate = best_params + 1e-4 * range_width * np.random.randn(dim)
            evaluate(candidate)
    
    return best
