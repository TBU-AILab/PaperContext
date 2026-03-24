#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Better DE strategy**: Use SHADE-like adaptive DE with memory of successful parameters
#2. **Multiple restarts**: Instead of one population, use restart-based approach
#3. **Better local search**: Use Nelder-Mead simplex with proper implementation
#4. **More efficient time management**
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
    
    def remaining():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Initial sampling with Sobol-like LHS ---
    n_init = min(max(10 * dim, 40), 300)
    
    pop = np.zeros((n_init, dim))
    for j in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, j] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    pop = lower + pop * ranges
    
    fit = np.array([eval_f(pop[i]) for i in range(n_init) if elapsed() < max_time * 0.9])
    if len(fit) < n_init:
        pop = pop[:len(fit)]
    
    # --- Phase 2: SHADE-like Adaptive DE with restarts ---
    pop_size = min(max(8 * dim, 30), 100)
    
    # Keep best pop_size individuals
    idx = np.argsort(fit)[:pop_size]
    population = pop[idx].copy()
    fitness = fit[idx].copy()
    
    # SHADE memory
    mem_size = 10
    M_F = np.full(mem_size, 0.5)
    M_CR = np.full(mem_size, 0.5)
    k = 0
    
    archive = []
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.85:
        S_F, S_CR, S_w = [], [], []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            ri = np.random.randint(mem_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best
            p = max(2, int(0.1 * pop_size))
            p_best = np.random.randint(p)
            sorted_idx = np.argsort(fitness)
            x_pbest = population[sorted_idx[p_best]]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            if archive:
                combined = np.vstack([population, np.array(archive)])
                r2 = np.random.randint(len(combined))
                x_r2 = combined[r2]
            else:
                r2 = np.random.choice([j for j in idxs if j != r1])
                x_r2 = population[r2]
            
            mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
            
            cross = np.random.random(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_f(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi); S_w.append(fitness[i] - f_trial)
                archive.append(population[i].copy())
                if len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population, fitness = new_pop, new_fit
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % mem_size
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0; prev_best = best
        
        if stagnation > 15:
            stagnation = 0
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = best_x + ranges * 0.05 * np.random.randn(dim)
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = eval_f(population[i])
    
    # --- Phase 3: Nelder-Mead local search ---
    if best_x is not None:
        n = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        simplex = [best_x.copy()]
        for i in range(n):
            p = best_x.copy()
            p[i] += ranges[i] * 0.02
            p = np.clip(p, lower, upper)
            simplex.append(p)
        f_simplex = [eval_f(s) for s in simplex]
        
        while elapsed() < max_time * 0.98:
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = eval_f(xr)
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lower, upper)
                fc = eval_f(xc)
                if fc < f_simplex[-1]:
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = eval_f(simplex[i])
                        if elapsed() >= max_time * 0.98:
                            break
    
    return best
