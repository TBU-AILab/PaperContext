#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better DE strategy**: Use SHADE (Success-History based Adaptive DE) for better parameter adaptation
#2. **Larger population with better diversity management**
#3. **More aggressive local search** using Nelder-Mead simplex method
#4. **CMA-ES-like covariance adaptation** for the local search phase
#5. **Better time management**
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
    
    # --- Phase 1: Initial sampling with LHS ---
    pop_size = min(max(30, 15 * dim), 300)
    
    # Latin Hypercube Sampling
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    # --- Phase 2: SHADE-like Adaptive DE ---
    H = 50  # history size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        generation += 1
        
        S_F = []
        S_CR = []
        S_df = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.68:
                break
            
            # Generate F and CR from history
            ri = np.random.randint(0, H)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0, 1)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # DE/current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(fitness)[:p])
            
            indices = list(range(pop_size))
            indices.remove(i)
            r1 = np.random.choice(indices)
            
            # r2 from population + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            if i in combined:
                combined.remove(i)
            if r1 in combined:
                combined.remove(r1)
            r2_idx = np.random.choice(combined) if combined else r1
            
            if r2_idx < pop_size:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back clipping
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(abs(fitness[i] - f_trial))
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive:
                        archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        # Update memory
        if S_F:
            weights = np.array(S_df)
            weights = weights / (weights.sum() + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % H
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Population size reduction
        if generation % 20 == 0 and pop_size > max(8, dim + 1):
            worst = np.argsort(fitness)[-max(1, pop_size // 10):]
            keep = np.setdiff1d(np.arange(pop_size), worst)
            population = population[keep]
            fitness = fitness[keep]
            pop_size = len(population)
        
        if stagnation > 25:
            stagnation = 0
            n_replace = pop_size // 3
            worst_indices = np.argsort(fitness)[-n_replace:]
            for idx in worst_indices:
                if np.random.random() < 0.7 and best_x is not None:
                    sigma = 0.05 * ranges * (0.5 + 0.5 * np.random.random())
                    population[idx] = best_x + np.random.randn(dim) * sigma
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = eval_f(population[idx])
    
    # --- Phase 3: Nelder-Mead local search ---
    if best_x is not None and elapsed() < max_time * 0.95:
        n_simplex = dim + 1
        simplex = np.zeros((n_simplex, dim))
        simplex[0] = best_x.copy()
        scale = 0.05 * ranges
        for i in range(1, n_simplex):
            simplex[i] = best_x + scale * np.random.randn(dim)
            simplex[i] = np.clip(simplex[i], lower, upper)
        
        f_simplex = np.array([eval_f(simplex[i]) for i in range(n_simplex)])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while elapsed() < max_time * 0.97:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = simplex[:-1].mean(axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = eval_f(xr)
            
            if fr < f_simplex[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, lower, upper)
                fc = eval_f(xc)
                if fc < f_simplex[-1]:
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for i in range(1, n_simplex):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = eval_f(simplex[i])
    
    # --- Phase 4: Fine-grained coordinate search ---
    if best_x is not None and elapsed() < max_time * 0.99:
        x_cur = best_x.copy()
        f_cur = best
        step = 0.001 * ranges
        
        for _ in range(100):
            if elapsed() >= max_time * 0.99:
                break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.99:
                    break
                for sign in [1, -1]:
                    xt = x_cur.copy()
                    xt[d] += sign * step[d]
                    xt = np.clip(xt, lower, upper)
                    ft = eval_f(xt)
                    if ft < f_cur:
                        x_cur, f_cur = xt, ft
                        step[d] *= 1.2
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-12:
                    break
    
    return best
