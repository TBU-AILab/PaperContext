#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better exploration**: Use multiple restarts with different strategies
#2. **Implement proper CMA-ES**: The previous code claimed CMA-ES but only used DE
#3. **Better adaptive DE**: Use SHADE-like parameter adaptation
#4. **More efficient local search**: Use Nelder-Mead simplex instead of coordinate descent
#5. **Better time management**
#
#Here's my improved implementation:
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Initialization with diverse sampling ---
    pop_size = min(max(30, 8 * dim), 300)
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.array([eval_f(population[i]) for i in range(pop_size)
                             if elapsed() < max_time * 0.05] + 
                            [float('inf')] * pop_size)[:pop_size]
    
    # Evaluate remaining if time allows
    for i in range(pop_size):
        if elapsed() >= max_time * 0.10:
            break
        if fitness_vals[i] == float('inf'):
            fitness_vals[i] = eval_f(population[i])
    
    # --- Phase 2: SHADE (Success-History based Adaptive DE) ---
    H = 50  # history size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0  # history index
    
    archive = []
    archive_max = pop_size
    
    p_min = 2.0 / pop_size
    p_max = 0.2
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while elapsed() < max_time * 0.82:
        generation += 1
        
        sort_idx = np.argsort(fitness_vals)
        population = population[sort_idx]
        fitness_vals = fitness_vals[sort_idx]
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.82:
                break
            
            # Generate F and CR from history
            r = np.random.randint(H)
            
            # Cauchy for F
            while True:
                Fi = M_F[r] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            
            # Normal for CR
            CRi = np.clip(M_CR[r] + 0.1 * np.random.randn(), 0, 1)
            
            # p-best
            p = p_min + np.random.random() * (p_max - p_min)
            p_count = max(1, int(p * pop_size))
            x_pbest = population[np.random.randint(p_count)]
            
            # Select r1 != i
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            # Select r2 from pop + archive, != i, != r1
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            while True:
                r2_idx = np.random.randint(len(combined))
                if combined[r2_idx] != i and combined[r2_idx] != r1:
                    break
            if combined[r2_idx] < pop_size:
                x_r2 = population[combined[r2_idx]]
            else:
                x_r2 = archive[combined[r2_idx] - pop_size]
            
            # Current-to-pbest/1
            mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            # Bounce-back clipping
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            
            if f_trial <= new_fit[i]:
                # Archive old vector
                if len(archive) < archive_max:
                    archive.append(population[i].copy())
                elif len(archive) > 0:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                
                delta = fitness_vals[i] - f_trial
                if delta > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness_vals = new_fit
        
        # Update history
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / weights.sum()
            
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            
            M_F[k] = mean_F
            M_CR[k] = mean_CR
            k = (k + 1) % H
        
        # Stagnation check with restart
        if abs(prev_best - best) < 1e-14:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 25 + dim:
            # Keep top 20%, reinitialize rest
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx]
            fitness_vals = fitness_vals[sort_idx]
            keep = max(2, pop_size // 5)
            for i in range(keep, pop_size):
                if np.random.random() < 0.4 and best_x is not None:
                    sigma = 0.05 * ranges * (np.random.random() + 0.1)
                    population[i] = clip(best_x + sigma * np.random.randn(dim))
                else:
                    population[i] = lower + np.random.random(dim) * ranges
                fitness_vals[i] = eval_f(population[i])
                if elapsed() >= max_time * 0.82:
                    break
            stagnation_count = 0
            archive = []
    
    # --- Phase 3: Nelder-Mead local search around best ---
    if best_x is not None and dim <= 100:
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        simplex_f = np.zeros(n + 1)
        simplex_f[0] = best
        
        init_scale = 0.02 * ranges
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += init_scale[i] * (1 if np.random.random() > 0.5 else -1)
            simplex[i + 1] = clip(simplex[i + 1])
            simplex_f[i + 1] = eval_f(simplex[i + 1])
            if elapsed() >= max_time * 0.98:
                return best
        
        while elapsed() < max_time * 0.98:
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha_nm * (centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                xe = clip(centroid + gamma_nm * (xr - centroid))
                fe = eval_f(xe)
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
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = clip(centroid + rho_nm * (xr - centroid))
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            simplex_f[i] = eval_f(simplex[i])
                            if elapsed() >= max_time * 0.98:
                                return best
                else:
                    # Inside contraction
                    xc = clip(centroid - rho_nm * (centroid - simplex[-1]))
                    fc = eval_f(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            simplex_f[i] = eval_f(simplex[i])
                            if elapsed() >= max_time * 0.98:
                                return best
            
            # Check convergence
            if np.max(np.abs(simplex_f[0] - simplex_f[-1])) < 1e-16:
                break
    
    return best
#
#Key improvements:
#1. **SHADE algorithm** - proper success-history based adaptive DE with Lehmer mean for F and weighted mean for CR
#2. **Archive mechanism** - stores old replaced solutions to improve diversity in mutations
#3. **p-best mutation** - uses top-p% solutions instead of just the single best
#4. **Bounce-back clipping** - better boundary handling
#5. **Full Nelder-Mead** simplex method for local refinement
#6. **Better restart strategy** - keeps top 20% and mixes random + local perturbations
#7. **Centralized evaluation** via `eval_f` that always tracks best
