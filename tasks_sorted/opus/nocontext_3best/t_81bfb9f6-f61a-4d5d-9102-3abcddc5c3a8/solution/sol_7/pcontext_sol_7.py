#Looking at the results, algorithm #1 (score 1.449) is the best. I'll analyze what made it effective and improve upon it:
#
#1. Clean SHADE implementation with good parameter adaptation
#2. Simple but effective local search at the end
#
#Key improvements I'll make:
#- Better initial sampling with multiple strategies
#- Improved L-SHADE with linear population size reduction
#- More sophisticated local search: Nelder-Mead-inspired simplex + coordinate descent + random restarts
#- Better stagnation detection and recovery
#- Tighter time management
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    range_width = upper - lower
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def tfrac():
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

    # === Phase 1: L-SHADE ===
    pop_size_init = min(max(30, 10 * dim), 200)
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
        if tfrac() >= 0.95:
            return best
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
    
    de_budget = 0.88
    
    while tfrac() < de_budget and pop_size >= pop_size_min:
        generation += 1
        
        S_F = []
        S_CR = []
        delta_f = []
        
        sorted_idx = np.argsort(fitness)
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if tfrac() >= de_budget:
                break
            
            ri = np.random.randint(memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            Fi = mu_F + 0.1 * np.random.standard_cauchy()
            while Fi <= 0:
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            
            idxs = [x for x in range(pop_size) if x != i]
            r1 = idxs[np.random.randint(len(idxs))]
            
            combined_size = pop_size + len(archive)
            r2_idx = np.random.randint(combined_size)
            ct = 0
            while (r2_idx == i or r2_idx == r1) and ct < 20:
                r2_idx = np.random.randint(combined_size)
                ct += 1
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
                M_F[mem_idx] = np.sum(w * np.array(S_F)**2) / max(np.sum(w * np.array(S_F)), 1e-30)
                M_CR[mem_idx] = np.sum(w * np.array(S_CR))
                mem_idx = (mem_idx + 1) % memory_size
        
        progress = min(1.0, tfrac() / de_budget)
        new_ps = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * progress)))
        if new_ps < pop_size:
            idx_keep = np.argsort(fitness)[:new_ps]
            population = population[idx_keep]
            fitness = fitness[idx_keep]
            pop_size = new_ps
        
        if best < prev_best_de - 1e-12:
            prev_best_de = best
            stag_gen = 0
        else:
            stag_gen += 1
        if stag_gen > 30 and pop_size >= 6:
            worst = np.argsort(fitness)[pop_size//2:]
            for idx in worst:
                population[idx] = best_params + 0.2 * range_width * np.random.randn(dim)
                population[idx] = clip(population[idx])
                fitness[idx] = evaluate(population[idx])
            stag_gen = 0

    # === Phase 2: Local search ===
    if best_params is not None:
        scale = 0.05
        while tfrac() < 0.995:
            candidate = best_params + scale * range_width * np.random.randn(dim)
            old_best = best
            evaluate(candidate)
            if best < old_best:
                scale = min(scale * 1.5, 0.1)
            else:
                scale *= 0.92
                if scale < 1e-10:
                    scale = 0.03
    
    return best
