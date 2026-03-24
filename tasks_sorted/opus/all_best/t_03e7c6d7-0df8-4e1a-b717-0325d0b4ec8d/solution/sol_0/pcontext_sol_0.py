#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
            best_x = population[i].copy()
    
    # --- Phase 2: CMA-ES inspired + Differential Evolution hybrid ---
    # Sort population
    sort_idx = np.argsort(fitness)
    population = population[sort_idx]
    fitness = fitness[sort_idx]
    
    # DE parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    # Nelder-Mead simplex for local refinement
    def nelder_mead_step(simplex, simplex_f, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """One iteration of Nelder-Mead"""
        n = simplex.shape[0] - 1
        idx = np.argsort(simplex_f)
        simplex = simplex[idx]
        simplex_f = simplex_f[idx]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, lower, upper)
        fr = func(xr)
        
        if fr < simplex_f[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, lower, upper)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                simplex_f[-1] = fe
            else:
                simplex[-1] = xr
                simplex_f[-1] = fr
        elif fr < simplex_f[-2]:
            simplex[-1] = xr
            simplex_f[-1] = fr
        else:
            # Contraction
            if fr < simplex_f[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, lower, upper)
            fc = func(xc)
            if fc < simplex_f[-1]:
                simplex[-1] = xc
                simplex_f[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    simplex_f[i] = func(simplex[i])
        
        best_idx = np.argmin(simplex_f)
        return simplex, simplex_f, simplex[best_idx], simplex_f[best_idx]
    
    while True:
        passed_time = (datetime.now() - start)
        remaining = max_time - passed_time.total_seconds()
        if remaining < max_time * 0.05:
            break
        
        generation += 1
        
        # Adaptive DE with current-to-best mutation
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.7:
                break
            
            # Mutation: DE/current-to-best/1 + DE/rand/1 hybrid
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if np.random.rand() < 0.5:
                # current-to-best
                a, b = np.random.choice(idxs, 2, replace=False)
                # Adaptive F
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                mutant = population[i] + Fi * (population[0] - population[i]) + Fi * (population[a] - population[b])
            else:
                # rand/1
                a, b, c = np.random.choice(idxs, 3, replace=False)
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                mutant = population[a] + Fi * (population[b] - population[c])
            
            # Crossover
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounds handling - bounce back
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        population = new_pop
        fitness = new_fit
        
        # Re-sort
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # If stagnant, inject random individuals
        if stagnation > 5:
            n_replace = pop_size // 3
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * (upper - lower)
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_x = population[j].copy()
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead) around best solution ---
    if best_x is not None and dim <= 50:
        remaining = max_time - (datetime.now() - start).total_seconds()
        if remaining > max_time * 0.02:
            # Build simplex around best
            n_simplex = dim + 1
            simplex = np.zeros((n_simplex, dim))
            simplex[0] = best_x.copy()
            simplex_f = np.zeros(n_simplex)
            simplex_f[0] = best
            
            scale = 0.05 * (upper - lower)
            for i in range(1, n_simplex):
                simplex[i] = best_x.copy()
                simplex[i][i - 1] += scale[i - 1] if best_x[i - 1] + scale[i - 1] <= upper[i - 1] else -scale[i - 1]
                simplex[i] = np.clip(simplex[i], lower, upper)
                simplex_f[i] = func(simplex[i])
                if simplex_f[i] < best:
                    best = simplex_f[i]
                    best_x = simplex[i].copy()
            
            while (datetime.now() - start).total_seconds() < max_time * 0.95:
                simplex, simplex_f, sb, sf = nelder_mead_step(simplex, simplex_f)
                if sf < best:
                    best = sf
                    best_x = sb.copy()
    
    return best
