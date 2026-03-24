#Looking at the current algorithm, I can see several areas for improvement:
#
#1. The DE strategy could be more sophisticated with better adaptation
#2. Local search should be more aggressive and use better methods
#3. Need better exploration-exploitation balance
#4. Should use restarts when stagnating
#
#Let me design a hybrid algorithm combining SHADE (Success-History based Adaptive DE) with local search (Nelder-Mead) and restarts:
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
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
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
    
    # Nelder-Mead local search
    def nelder_mead(x0, max_evals=500, initial_step=0.05):
        n = len(x0)
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = evaluate(x0)
        
        for i in range(n):
            if remaining() <= 0:
                return
            p = x0.copy()
            step = initial_step * ranges[i]
            p[i] = p[i] + step
            p = clip(p)
            simplex[i + 1] = p
            f_simplex[i + 1] = evaluate(p)
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        evals = 0
        
        while evals < max_evals and remaining() > 0:
            # Sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            evals += 1
            
            if fr < f_simplex[0]:
                # Expand
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                evals += 1
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    evals += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if remaining() <= 0:
                                return
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                            evals += 1
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[-1]))
                    fc = evaluate(xc)
                    evals += 1
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if remaining() <= 0:
                                return
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                            evals += 1
    
    # SHADE parameters
    pop_size = max(30, 5 * dim)
    H = pop_size
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.5)
    mem_idx = 0
    archive = []
    max_archive = pop_size
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.random(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size) if remaining() > 0 or True])
    if remaining() <= 0:
        return best
    
    stagnation_counter = 0
    prev_best = best
    
    while remaining() > 0:
        S_F, S_CR = [], []
        delta_f = []
        
        # Generate F and CR from memory
        ri = np.random.randint(0, H, pop_size)
        F_vals = np.zeros(pop_size)
        CR_vals = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Cauchy for F
            while True:
                Fi = memory_F[ri[i]] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            F_vals[i] = min(Fi, 1.0)
            # Normal for CR
            CRi = np.clip(memory_CR[ri[i]] + 0.1 * np.random.randn(), 0, 1)
            CR_vals[i] = CRi
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        # Sort for p-best
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if remaining() <= 0:
                return best
            
            F = F_vals[i]
            CR = CR_vals[i]
            
            # p-best index
            p = max(2, int(0.1 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(0, p)]
            
            # r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            # r2 from pop + archive, != i, != r1
            union_size = pop_size + len(archive)
            r2 = np.random.randint(0, union_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, union_size)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - x_r2)
            
            # Bounce-back boundary handling
            for j in range(dim):
                if mutant[j] < lower[j]:
                    mutant[j] = (lower[j] + population[i][j]) / 2
                elif mutant[j] > upper[j]:
                    mutant[j] = (upper[j] + population[i][j]) / 2
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(abs(fitness[i] - f_trial))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)
            S_F = np.array(S_F)
            S_CR = np.array(S_CR)
            # Lehmer mean for F
            memory_F[mem_idx] = np.sum(weights * S_F ** 2) / (np.sum(weights * S_F) + 1e-30)
            # Weighted mean for CR
            memory_CR[mem_idx] = np.sum(weights * S_CR)
            mem_idx = (mem_idx + 1) % H
        
        # Stagnation detection
        if abs(best - prev_best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            prev_best = best
        
        # Local search periodically or on stagnation
        if stagnation_counter >= 15 and remaining() > 1.0:
            nelder_mead(best_params.copy(), max_evals=min(300, dim * 20), initial_step=0.02)
            stagnation_counter = 0
            prev_best = best
        
        # Hard restart if deeply stagnated
        if stagnation_counter >= 40:
            stagnation_counter = 0
            prev_best = best
            # Keep best, reinitialize rest
            best_idx = np.argmin(fitness)
            for i in range(pop_size):
                if remaining() <= 0:
                    return best
                if i == best_idx:
                    continue
                population[i] = lower + np.random.random(dim) * ranges
                fitness[i] = evaluate(population[i])
            memory_F[:] = 0.5
            memory_CR[:] = 0.5
            archive.clear()
    
    # Final local search
    if best_params is not None and remaining() > 0.1:
        nelder_mead(best_params.copy(), max_evals=1000, initial_step=0.01)
    
    return best
