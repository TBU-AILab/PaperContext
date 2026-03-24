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
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * (upper[d] - lower[d])
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(init_pop[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_params = init_pop[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Use multiple restarts with different strategies
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    # Sort initial population and keep best ones
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2a: Nelder-Mead from best point ---
    def nelder_mead(x0, max_iter=None, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        nonlocal best, best_params
        n = dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            scale = (upper[i] - lower[i]) * 0.05
            if scale < 1e-10:
                scale = 1e-10
            p[i] = p[i] + scale
            p = clip(p)
            simplex[i + 1] = p
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if time_left() < max_time * 0.02:
                return
            f_simplex[i] = func(simplex[i])
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        iteration = 0
        max_it = max_iter if max_iter else 10000
        while iteration < max_it:
            iteration += 1
            if time_left() < max_time * 0.02:
                return
            
            # Sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                continue
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                continue
            
            # Contraction
            if fr < f_simplex[-1]:
                xc = clip(centroid + rho * (xr - centroid))
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc <= fr:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                    continue
            else:
                xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                    continue
            
            # Shrink
            for i in range(1, n + 1):
                if time_left() < max_time * 0.02:
                    return
                simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                f_simplex[i] = func(simplex[i])
                if f_simplex[i] < best:
                    best = f_simplex[i]
                    best_params = simplex[i].copy()
            
            # Check convergence
            if np.std(f_simplex) < 1e-15:
                return
    
    # --- Phase 2b: Differential Evolution ---
    def differential_evolution():
        nonlocal best, best_params
        
        pop_size = min(max(10 * dim, 40), 200)
        
        # Initialize population from best initial samples + random
        n_elite = min(pop_size // 4, len(sorted_idx))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        
        for i in range(n_elite, pop_size):
            pop[i] = np.array([np.random.uniform(lower[d], upper[d]) for d in range(dim)])
            if time_left() < max_time * 0.05:
                return
            fit[i] = func(pop[i])
            if fit[i] < best:
                best = fit[i]
                best_params = pop[i].copy()
        
        F = 0.8
        CR = 0.9
        
        generation = 0
        while True:
            generation += 1
            if time_left() < max_time * 0.05:
                return
            
            for i in range(pop_size):
                if time_left() < max_time * 0.02:
                    return
                
                # Adaptive F and CR
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                CRi = CR + 0.1 * np.random.randn()
                CRi = np.clip(CRi, 0.0, 1.0)
                
                # current-to-best/1
                best_idx = np.argmin(fit)
                
                # Select 2 random distinct indices != i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r = np.random.choice(candidates, 2, replace=False)
                r1, r2 = r[0], r[1]
                
                mutant = pop[i] + Fi * (pop[best_idx] - pop[i]) + Fi * (pop[r1] - pop[r2])
                mutant = clip(mutant)
                
                # Crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CRi or j == j_rand:
                        trial[j] = mutant[j]
                
                trial = clip(trial)
                f_trial = func(trial)
                
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    # Run Nelder-Mead on best found point (use ~30% of remaining time)
    time_for_nm = time_left() * 0.3
    if time_for_nm > 0.1 and best_params is not None:
        nelder_mead(best_params, max_iter=int(50 * dim))
    
    # Run DE for the bulk of remaining time
    if time_left() > 0.2:
        differential_evolution()
    
    # Final Nelder-Mead polish
    if time_left() > 0.2 and best_params is not None:
        nelder_mead(best_params, max_iter=int(100 * dim))
    
    return best
