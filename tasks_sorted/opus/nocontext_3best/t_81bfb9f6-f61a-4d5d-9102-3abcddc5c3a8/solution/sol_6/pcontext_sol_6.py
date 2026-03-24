#Looking at the results, algorithm #1 (1.449) is the best. I'll analyze what made it work well and improve it with:
#
#1. Better SHADE with linear population size reduction (L-SHADE)
#2. More aggressive exploitation with a simplified but effective Nelder-Mead local search
#3. Better stagnation handling with restarts around the best
#4. Improved time management
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    range_width = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_frac():
        return elapsed() / max_time
    
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
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if time_frac() >= 0.10:
            fitness[i] = float('inf')
            continue
        fitness[i] = evaluate(population[i])
    
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size_init
    
    generation = 0
    stag_gen = 0
    prev_best_de = best
    
    de_budget = 0.85
    
    while time_frac() < de_budget and pop_size >= pop_size_min:
        generation += 1
        
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fitness)
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if time_frac() >= de_budget:
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
            p = max(2, int(0.11 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            candidates = [x for x in range(pop_size) if x != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            combined_size = pop_size + len(archive)
            r2_idx = np.random.randint(combined_size)
            attempts = 0
            while (r2_idx == i or r2_idx == r1) and attempts < 25:
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
                new_pop[i] = trial
                new_fit[i] = trial_f
        
        population = new_pop
        fitness = new_fit
        
        if S_F:
            w = np.array(delta_f)
            ws = w.sum()
            if ws > 0:
                w = w / ws
                sf = np.array(S_F)
                M_F[mem_idx] = np.sum(w * sf**2) / max(np.sum(w * sf), 1e-30)
                M_CR[mem_idx] = np.sum(w * np.array(S_CR))
                mem_idx = (mem_idx + 1) % memory_size
        
        # Linear population size reduction
        progress = min(1.0, time_frac() / de_budget)
        new_ps = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * progress)))
        if new_ps < pop_size:
            idx_keep = np.argsort(fitness)[:new_ps]
            population = population[idx_keep]
            fitness = fitness[idx_keep]
            pop_size = new_ps
        
        if best < prev_best_de:
            prev_best_de = best
            stag_gen = 0
        else:
            stag_gen += 1
        
        if stag_gen > 30:
            stag_gen = 0
            n_reset = max(1, pop_size // 2)
            worst = np.argsort(fitness)[-n_reset:]
            for idx in worst:
                population[idx] = best_params + 0.2 * range_width * np.random.randn(dim)
                population[idx] = clip(population[idx])
                fitness[idx] = evaluate(population[idx])

    # === Local search: adaptive coordinate descent + random perturbation ===
    if best_params is not None:
        scale = 0.05
        while time_frac() < 0.995:
            base = best_params.copy()
            base_f = best
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if time_frac() >= 0.995:
                    break
                for sign in [1, -1]:
                    c = best_params.copy()
                    c[d] += sign * scale * range_width[d]
                    cf = evaluate(c)
                    if cf < base_f:
                        base_f = cf
                        improved = True
                        break
            if improved:
                direction = best_params - base
                nrm = np.linalg.norm(direction)
                if nrm > 1e-15:
                    for alpha in [2.0, 3.0]:
                        if time_frac() >= 0.995:
                            break
                        evaluate(base + alpha * direction)
                scale = min(scale * 1.2, 0.1)
            else:
                evaluate(best_params + scale * range_width * np.random.randn(dim))
                scale *= 0.7
                if scale < 1e-10:
                    scale = 0.03
    
    return best
