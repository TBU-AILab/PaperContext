#Looking at the two algorithms, the second one (DE-based) performed significantly better (54.94 vs 160.89). Let me improve upon it with:
#
#1. Better initial sampling
#2. Multiple DE strategies (SHADE-like adaptive DE)
#3. More aggressive local search with multiple scales
#4. Better restart mechanisms
#5. Opposition-based learning
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_remaining():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size = min(max(30, 10 * dim), 300)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    # Also add opposition-based points
    opp_pop = lower + upper - population
    
    all_candidates = np.vstack([population, opp_pop])
    all_fitness = np.full(len(all_candidates), float('inf'))
    
    for i in range(len(all_candidates)):
        if elapsed() >= max_time * 0.15:
            all_fitness[i:] = float('inf')
            break
        all_fitness[i] = evaluate(all_candidates[i])
    
    # Select best pop_size
    valid = all_fitness < float('inf')
    valid_idx = np.where(valid)[0]
    sorted_valid = valid_idx[np.argsort(all_fitness[valid_idx])]
    sel = sorted_valid[:pop_size]
    population = all_candidates[sel].copy()
    fitness = all_fitness[sel].copy()
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: SHADE-like Adaptive DE ---
    # Memory for successful F and CR
    memory_size = 20
    memory_F = np.full(memory_size, 0.5)
    memory_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.80:
        generation += 1
        
        success_F = []
        success_CR = []
        success_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.80:
                break
            
            # Generate F and CR from memory
            ri = np.random.randint(memory_size)
            F_i = np.clip(np.random.standard_cauchy() * 0.1 + memory_F[ri], 0.01, 1.0)
            CR_i = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            
            # Select strategy randomly
            strategy = np.random.randint(3)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if strategy == 0:
                # current-to-pbest/1
                p = max(2, int(0.1 * pop_size))
                p_best_idx = np.random.randint(p)
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                # r2 can come from archive
                union = list(range(pop_size)) + list(range(len(archive)))
                mutant = population[i] + F_i * (population[p_best_idx] - population[i])
                if len(archive) > 0 and np.random.random() < 0.5:
                    arc_idx = np.random.randint(len(archive))
                    mutant += F_i * (population[r1] - archive[arc_idx])
                else:
                    mutant += F_i * (population[r1] - population[r2])
            elif strategy == 1:
                # rand/1
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = population[r1] + F_i * (population[r2] - population[r3])
            else:
                # current-to-best/2
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                idxs2 = [x for x in idxs if x != r1 and x != r2]
                if len(idxs2) >= 2:
                    r3, r4 = np.random.choice(idxs2, 2, replace=False)
                    mutant = population[i] + F_i * (population[0] - population[i]) + F_i * (population[r1] - population[r2]) + 0.5 * F_i * (population[r3] - population[r4])
                else:
                    mutant = population[i] + F_i * (population[0] - population[i]) + F_i * (population[r1] - population[r2])
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR_i
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back clipping
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            
            if f_trial < fitness[i]:
                delta = fitness[i] - f_trial
                success_F.append(F_i)
                success_CR.append(CR_i)
                success_delta.append(delta)
                
                if len(archive) < archive_max:
                    archive.append(population[i].copy())
                elif len(archive) > 0:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                
                new_population[i] = trial
                new_fitness[i] = f_trial
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if len(success_F) > 0:
            weights = np.array(success_delta)
            weights = weights / np.sum(weights)
            sf = np.array(success_F)
            scr = np.array(success_CR)
            # Lehmer mean for F
            memory_F[mem_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            memory_CR[mem_idx] = np.sum(weights * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        # Sort
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnated
        if stagnation > 25:
            half = pop_size // 2
            for i in range(half, pop_size):
                if np.random.random() < 0.3:
                    population[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
                else:
                    sigma = 0.05 * ranges * max(0.01, 1.0 - generation / 200.0)
                    population[i] = best_x + sigma * np.random.randn(dim)
                population[i] = clip(population[i])
                fitness[i] = evaluate(population[i])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Multi-scale local search ---
    if best_x is not None:
        for scale in [0.01, 0.001, 0.0001]:
            if elapsed() >= max_time * 0.98:
                break
            step = scale * ranges
            improved_outer = True
            while improved_outer:
                if elapsed() >= max_time * 0.98:
                    break
                improved_outer = False
                for j in range(dim):
                    if elapsed() >= max_time * 0.98:
                        break
                    for direction in [1, -1]:
                        trial = best_x.copy()
                        trial[j] += direction * step[j]
                        f_trial = evaluate(trial)
                        if f_trial < best:
                            improved_outer = True
                            step[j] *= 1.5
                            break
                    else:
                        step[j] *= 0.5
    
    # --- Phase 4: Use remaining time for random perturbations around best ---
    while elapsed() < max_time * 0.99 and best_x is not None:
        sigma = 0.001 * ranges * np.random.random()
        trial = best_x + sigma * np.random.randn(dim)
        evaluate(trial)
    
    return best
#
#Key improvements:
#1. **SHADE-like adaptive DE** with memory of successful F/CR parameters and Lehmer mean updates
#2. **Opposition-based learning** in initialization doubles coverage
#3. **Multiple mutation strategies** (current-to-pbest, rand/1, current-to-best/2) for diversity
#4. **External archive** for diversity maintenance
#5. **Bounce-back clipping** instead of simple clipping for better boundary handling
#6. **Multi-scale coordinate descent** refinement at 3 scales
#7. **Final perturbation phase** to exploit remaining time
