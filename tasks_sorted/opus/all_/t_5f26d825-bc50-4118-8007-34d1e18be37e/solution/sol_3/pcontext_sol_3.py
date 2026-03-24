#Looking at the results, Algorithm 1 (SHADE-like adaptive DE) performed best with 26.06. Let me improve upon it with:
#
#1. Better CMA-ES implementation (properly working, unlike Algorithm 2)
#2. Multi-population approach with information sharing
#3. More sophisticated local search (pattern search + golden section)
#4. Better time management
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
    
    # --- Phase 1: Stratified initial sampling ---
    n_init = min(max(40, 15 * dim), 500)
    population = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        population[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    # Opposition-based
    opp_pop = lower + upper - population
    all_cands = np.vstack([population, opp_pop])
    all_fit = np.array([evaluate(all_cands[i]) for i in range(len(all_cands)) if elapsed() < max_time * 0.10] + [float('inf')] * max(0, len(all_cands)))
    
    # Properly handle partial evaluation
    fit_list = []
    for i in range(len(all_cands)):
        if elapsed() >= max_time * 0.10:
            break
        fit_list.append(evaluate(all_cands[i]))
    
    if len(fit_list) == 0:
        return best
    
    eval_cands = all_cands[:len(fit_list)]
    eval_fits = np.array(fit_list)
    
    pop_size = min(max(30, 8 * dim), 200)
    sorted_idx = np.argsort(eval_fits)[:pop_size]
    population = eval_cands[sorted_idx].copy()
    fitness = eval_fits[sorted_idx].copy()
    
    # --- Phase 2: SHADE with linear population size reduction ---
    memory_size = 30
    memory_F = np.full(memory_size, 0.5)
    memory_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size
    
    min_pop_size = max(6, dim)
    initial_pop_size = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.75:
        generation += 1
        
        # Linear population size reduction
        progress = min(1.0, elapsed() / (max_time * 0.75))
        target_pop = max(min_pop_size, int(initial_pop_size - (initial_pop_size - min_pop_size) * progress))
        
        success_F = []
        success_CR = []
        success_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(len(population)):
            if elapsed() >= max_time * 0.75:
                break
            
            ri = np.random.randint(memory_size)
            # Cauchy for F
            F_i = -1
            while F_i <= 0:
                F_i = np.random.standard_cauchy() * 0.1 + memory_F[ri]
                if F_i >= 1.0:
                    F_i = 1.0
                    break
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            
            cur_pop_size = len(population)
            idxs = list(range(cur_pop_size))
            idxs.remove(i)
            
            # current-to-pbest/1
            p = max(2, int(0.15 * cur_pop_size))
            p_best_idx = np.random.randint(p)
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined_idxs = [j for j in idxs if j != r1]
            if len(archive) > 0 and np.random.random() < 0.5:
                arc_idx = np.random.randint(len(archive))
                diff2 = population[r1] - archive[arc_idx]
            else:
                if len(combined_idxs) > 0:
                    r2 = np.random.choice(combined_idxs)
                    diff2 = population[r1] - population[r2]
                else:
                    diff2 = np.random.randn(dim) * 0.01 * ranges
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * diff2
            
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
        
        # Update memory (weighted Lehmer mean)
        if len(success_F) > 0:
            weights = np.array(success_delta)
            weights = weights / (np.sum(weights) + 1e-30)
            sf = np.array(success_F)
            scr = np.array(success_CR)
            memory_F[mem_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            memory_CR[mem_idx] = np.sum(weights * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        # Sort and potentially reduce population
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if len(population) > target_pop:
            population = population[:target_pop]
            fitness = fitness[:target_pop]
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 30:
            half = len(population) // 2
            for i in range(half, len(population)):
                sigma = 0.05 * ranges * max(0.001, 1.0 - progress)
                population[i] = best_x + sigma * np.random.randn(dim)
                population[i] = clip(population[i])
                fitness[i] = evaluate(population[i])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Multi-scale local search ---
    if best_x is not None:
        for scale in [0.01, 0.002, 0.0005, 0.0001]:
            if elapsed() >= max_time * 0.97:
                break
            step = scale * ranges
            for _ in range(3):
                if elapsed() >= max_time * 0.97:
                    break
                improved = False
                for j in range(dim):
                    if elapsed() >= max_time * 0.97:
                        break
                    for direction in [1, -1]:
                        trial = best_x.copy()
                        trial[j] += direction * step[j]
                        if evaluate(trial) < best - 1e-15:
                            improved = True
                            step[j] *= 1.5
                            break
                    else:
                        step[j] *= 0.5
                if not improved:
                    break
    
    # --- Phase 4: Final random perturbations ---
    while elapsed() < max_time * 0.995 and best_x is not None:
        sigma = 0.0005 * ranges * np.random.random()
        evaluate(best_x + sigma * np.random.randn(dim))
    
    return best
