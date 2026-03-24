#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like approach
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness_vals = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness_vals[i] = f
        if f < best:
            best = f
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with restarts and Nelder-Mead refinement ---
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        improved_this_gen = False
        
        # Adaptive parameters
        F_adapt = 0.5 + 0.3 * np.random.random()
        CR_adapt = 0.7 + 0.3 * np.random.random()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.85):
                break
            
            # Select 3 distinct individuals different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            chosen = np.random.choice(idxs, 3, replace=False)
            a, b, c = population[chosen[0]], population[chosen[1]], population[chosen[2]]
            
            # Current-to-best mutation strategy (mix of DE/rand/1 and DE/current-to-best/1)
            if np.random.random() < 0.5:
                # DE/current-to-best/1
                mutant = population[i] + F_adapt * (best_params - population[i]) + F_adapt * (a - b)
            else:
                # DE/rand/1
                mutant = a + F_adapt * (b - c)
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CR_adapt
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            # Evaluate
            f_trial = func(trial)
            
            if f_trial < fitness_vals[i]:
                population[i] = trial
                fitness_vals[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                    improved_this_gen = True
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        # Restart worst half if stagnating
        if stagnation_count > 10:
            sorted_idx = np.argsort(fitness_vals)
            half = pop_size // 2
            for j in sorted_idx[half:]:
                population[j] = lower + np.random.random(dim) * (upper - lower)
                # Small perturbation around best for some
                if np.random.random() < 0.3:
                    scale = 0.1 * (upper - lower)
                    population[j] = best_params + np.random.randn(dim) * scale
                    population[j] = np.clip(population[j], lower, upper)
                fitness_vals[j] = func(population[j])
                if fitness_vals[j] < best:
                    best = fitness_vals[j]
                    best_params = population[j].copy()
            stagnation_count = 0
    
    # --- Phase 3: Nelder-Mead local search around best solution ---
    def nelder_mead_step():
        nonlocal best, best_params
        
        n = dim
        # Initialize simplex around best_params
        scale = 0.05 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = best
        
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += scale[i] if scale[i] != 0 else 0.01
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            f_simplex[i + 1] = func(simplex[i + 1])
            if f_simplex[i + 1] < best:
                best = f_simplex[i + 1]
                best_params = simplex[i + 1].copy()
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(1000):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                return
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best:
                best = f_simplex[0]
                best_params = simplex[0].copy()
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
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
                # Contraction
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, n + 1):
                            simplex[j] = np.clip(simplex[0] + sigma * (simplex[j] - simplex[0]), lower, upper)
                            f_simplex[j] = func(simplex[j])
                            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                                if f_simplex[j] < best:
                                    best = f_simplex[j]
                                    best_params = simplex[j].copy()
                                return
                else:
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = func(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, n + 1):
                            simplex[j] = np.clip(simplex[0] + sigma * (simplex[j] - simplex[0]), lower, upper)
                            f_simplex[j] = func(simplex[j])
                            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                                if f_simplex[j] < best:
                                    best = f_simplex[j]
                                    best_params = simplex[j].copy()
                                return
            
            if f_simplex[-1] < best:
                best = f_simplex[-1]
                best_params = simplex[-1].copy()
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
    
    nelder_mead_step()
    
    # Use remaining time for random perturbation search
    while (datetime.now() - start) < timedelta(seconds=max_time * 0.99):
        scale = 0.01 * (upper - lower) * np.random.random()
        candidate = best_params + np.random.randn(dim) * scale
        candidate = np.clip(candidate, lower, upper)
        f_cand = func(candidate)
        if f_cand < best:
            best = f_cand
            best_params = candidate.copy()
    
    return best
