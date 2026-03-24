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
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        f = func(init_pop[i])
        init_fitness[i] = f
        if f < best:
            best = f
            best_params = init_pop[i].copy()
    
    # --- Phase 2: CMA-ES inspired search from best candidates ---
    # Use multiple restarts with different strategies
    
    def time_left():
        return max_time * 0.95 - (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    # Nelder-Mead simplex search
    def nelder_mead(x0, initial_scale=0.1, max_iter=10000):
        nonlocal best, best_params
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            step = initial_scale * (upper[i] - lower[i])
            if step == 0:
                step = 0.1
            p[i] = p[i] + step
            p = clip(p)
            simplex[i + 1] = p
        
        f_values = np.zeros(n + 1)
        for i in range(n + 1):
            if time_left() <= 0:
                return
            f_values[i] = func(simplex[i])
            if f_values[i] < best:
                best = f_values[i]
                best_params = simplex[i].copy()
        
        for iteration in range(max_iter):
            if time_left() <= 0:
                return
            
            # Sort
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_values[0] <= fr < f_values[n - 1]:
                simplex[n] = xr
                f_values[n] = fr
            elif fr < f_values[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[n] = xe
                    f_values[n] = fe
                else:
                    simplex[n] = xr
                    f_values[n] = fr
            else:
                # Contraction
                if fr < f_values[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[n] - centroid))
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < f_values[n]:
                    simplex[n] = xc
                    f_values[n] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        if time_left() <= 0:
                            return
                        f_values[i] = func(simplex[i])
                        if f_values[i] < best:
                            best = f_values[i]
                            best_params = simplex[i].copy()
            
            # Check convergence
            if np.std(f_values) < 1e-15:
                return
    
    # CMA-ES
    def cma_es(x0, sigma0=0.3, pop_size=None):
        nonlocal best, best_params
        n = dim
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(n))
        mu = pop_size // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        mean = x0.copy()
        sigma = sigma0 * np.mean(upper - lower)
        
        gen = 0
        max_gen = 100000
        
        while gen < max_gen:
            if time_left() <= 0:
                return
            
            # Generate population
            arz = np.random.randn(pop_size, n)
            arx = np.zeros((pop_size, n))
            for k in range(pop_size):
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
            
            # Evaluate
            fitness = np.zeros(pop_size)
            for k in range(pop_size):
                if time_left() <= 0:
                    return
                fitness[k] = func(arx[k])
                if fitness[k] < best:
                    best = fitness[k]
                    best_params = arx[k].copy()
            
            # Sort
            order = np.argsort(fitness)
            arx = arx[order]
            arz = arz[order]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, arx[:mu])
            mean = clip(mean)
            
            # Update evolution paths
            zmean = np.dot(weights, arz[:mu])
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ (D * zmean))
            
            # Update covariance matrix
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.mean(upper - lower) * 2)  # cap sigma
            
            # Update B and D from C
            eigeneval += 1
            if eigeneval >= 1:
                eigeneval = 0
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    # Reset on numerical issues
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
            
            gen += 1
            
            # Check for convergence
            if sigma * np.max(D) < 1e-15:
                return
    
    # Differential evolution step interleaved
    def differential_evolution(pop, pop_fit, max_evals=None):
        nonlocal best, best_params
        n_pop = len(pop)
        F = 0.8
        CR = 0.9
        
        evals = 0
        while True:
            if time_left() <= 0:
                return pop, pop_fit
            if max_evals and evals >= max_evals:
                return pop, pop_fit
            
            for i in range(n_pop):
                if time_left() <= 0:
                    return pop, pop_fit
                if max_evals and evals >= max_evals:
                    return pop, pop_fit
                
                # Select 3 random distinct indices
                idxs = list(range(n_pop))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                # Mutation: best/1 strategy mixed with rand/1
                if np.random.random() < 0.5:
                    best_idx = np.argmin(pop_fit)
                    mutant = clip(pop[best_idx] + F * (pop[a] - pop[b]))
                else:
                    mutant = clip(pop[a] + F * (pop[b] - pop[c]))
                
                # Crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = clip(trial)
                
                f_trial = func(trial)
                evals += 1
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                
                if f_trial <= pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = f_trial
        
        return pop, pop_fit
    
    # Strategy: Use time budget wisely
    total_time = max_time * 0.95
    elapsed = (datetime.now() - start).total_seconds()
    remaining = total_time - elapsed
    
    if remaining <= 0:
        return best
    
    # Sort initial population, get top candidates
    sorted_idx = np.argsort(init_fitness)
    
    # Phase 2a: Run Nelder-Mead from best few points (use ~20% of remaining time)
    nm_time_budget = remaining * 0.15
    nm_start = (datetime.now() - start).total_seconds()
    n_nm_starts = min(3, n_init)
    for i in range(n_nm_starts):
        if time_left() <= 0 or (datetime.now() - start).total_seconds() - nm_start > nm_time_budget:
            break
        scale = 0.05 * (1 + i)
        nelder_mead(init_pop[sorted_idx[i]], initial_scale=scale)
    
    # Phase 2b: Run CMA-ES from best point (use ~40% of remaining time)
    if time_left() > 1:
        if best_params is not None:
            cma_es(best_params, sigma0=0.2)
    
    # Phase 2c: Differential Evolution with remaining time
    if time_left() > 1:
        # Create population from init + best found
        de_pop_size = min(max(10 * dim, 30), 100)
        de_pop = np.zeros((de_pop_size, dim))
        de_fit = np.full(de_pop_size, float('inf'))
        
        # Fill with best from init + random
        n_from_init = min(de_pop_size // 2, n_init)
        for i in range(n_from_init):
            de_pop[i] = init_pop[sorted_idx[i]]
            de_fit[i] = init_fitness[sorted_idx[i]]
        
        if best_params is not None:
            de_pop[0] = best_params.copy()
            de_fit[0] = best
        
        for i in range(n_from_init, de_pop_size):
            de_pop[i] = np.array([np.random.uniform(l, h) for l, h in bounds])
            if time_left() > 0:
                de_fit[i] = func(de_pop[i])
                if de_fit[i] < best:
                    best = de_fit[i]
                    best_params = de_pop[i].copy()
        
        differential_evolution(de_pop, de_fit)
    
    # Final local search around best with small perturbations
    if time_left() > 0.5 and best_params is not None:
        nelder_mead(best_params, initial_scale=0.01)
    
    return best
