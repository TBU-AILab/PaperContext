#Looking at the results, algorithm 1 (SHADE-based DE with multi-scale local search) performed best at 24.93. I'll improve upon it with:
#
#1. **Better initialization** - more diverse sampling
#2. **SHADE with linear population size reduction (L-SHADE)**
#3. **More aggressive local search** - coordinate descent with golden section + Nelder-Mead
#4. **Better time budget allocation**
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS + Opposition-based initialization ---
    pop_size_init = min(max(10 * dim, 40), 200)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    # Opposition-based
    opp_pop = lower + upper - population
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.full(len(all_pop), float('inf'))
    
    for i in range(len(all_pop)):
        if elapsed() >= max_time * 0.12:
            break
        all_fit[i] = eval_f(all_pop[i])
    
    valid = all_fit < float('inf')
    if np.sum(valid) >= pop_size:
        order = np.argsort(all_fit)
        population = all_pop[order[:pop_size]].copy()
        fitness = all_fit[order[:pop_size]].copy()
    else:
        population = all_pop[:pop_size].copy()
        fitness = all_fit[:pop_size].copy()

    # --- Phase 2: L-SHADE ---
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    max_archive = pop_size_init
    
    total_evals_estimate = 0
    gen_count = 0
    max_gen_estimate = 300
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.65:
        gen_count += 1
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.65:
                break
            
            r = np.random.randint(memory_size)
            
            # Generate F from Cauchy
            F_i = -1
            while F_i <= 0:
                F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
                if F_i >= 1.0:
                    F_i = 1.0
                    break
            F_i = min(F_i, 1.0)
            
            # Generate CR from Normal
            CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            
            # p-best
            p = max(2, int(max(0.05, 0.2 - 0.15 * gen_count / max(max_gen_estimate, gen_count)) * pop_size))
            pbest_idx = np.argsort(fitness)[:p]
            pbest = population[np.random.choice(pbest_idx)]
            
            # Select r1 from population
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            # Select r2 from population + archive
            combined_size = pop_size + len(archive)
            r2_candidates = [j for j in range(combined_size) if j != i and j != r1]
            if len(r2_candidates) == 0:
                r2_candidates = [j for j in range(pop_size) if j != i]
            r2 = np.random.choice(r2_candidates)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + F_i * (pbest - population[i]) + F_i * (population[r1] - x_r2)
            
            # Bounce-back
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross = np.random.random(dim) < CR_i
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            trial = clip(trial)
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                delta = fitness[i] - f_trial
                if delta > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(weights * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        # Linear pop reduction
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * elapsed() / (max_time * 0.65))))
        if new_pop_size < pop_size:
            order = np.argsort(fitness)
            population = population[order[:new_pop_size]]
            fitness = fitness[order[:new_pop_size]]
            pop_size = new_pop_size

    # --- Phase 3: Multi-scale coordinate descent ---
    if best_params is not None:
        for scale in [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]:
            step = scale * ranges
            improved = True
            while improved and elapsed() < max_time * 0.97:
                improved = False
                for d in range(dim):
                    if elapsed() >= max_time * 0.97:
                        break
                    current_best = best
                    for direction in [1, -1]:
                        trial = best_params.copy()
                        trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                        f_trial = eval_f(trial)
                        if f_trial < current_best:
                            improved = True
                            break
            if elapsed() >= max_time * 0.97:
                break
    
    # --- Phase 4: Random perturbation near best ---
    while elapsed() < max_time * 0.995:
        sigma = 0.001 * ranges * np.random.exponential(1.0)
        trial = best_params + sigma * np.random.randn(dim)
        eval_f(trial)
    
    return best
