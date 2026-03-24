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
    
    def nelder_mead(start_point, start_fitness, budget_seconds):
        """Nelder-Mead simplex method"""
        nonlocal best, best_params
        
        nm_start = datetime.now()
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = start_point.copy()
        f_values = np.full(n + 1, float('inf'))
        f_values[0] = start_fitness
        
        scale = (upper - lower) * 0.05
        for i in range(n):
            simplex[i + 1] = start_point.copy()
            simplex[i + 1][i] += scale[i] if scale[i] != 0 else 0.1
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            f_values[i + 1] = func(simplex[i + 1])
            if f_values[i + 1] < best:
                best = f_values[i + 1]
                best_params = simplex[i + 1].copy()
        
        max_iter = 10000
        for iteration in range(max_iter):
            if (datetime.now() - nm_start).total_seconds() >= budget_seconds:
                return
            
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            if f_values[0] < best:
                best = f_values[0]
                best_params = simplex[0].copy()
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_values[0] <= fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = func(xc)
                    if fc < best:
                        best = fc
                        best_params = xc.copy()
                    if fc <= fr:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = func(simplex[i])
                            if f_values[i] < best:
                                best = f_values[i]
                                best_params = simplex[i].copy()
                else:
                    xcc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fcc = func(xcc)
                    if fcc < best:
                        best = fcc
                        best_params = xcc.copy()
                    if fcc < f_values[-1]:
                        simplex[-1] = xcc
                        f_values[-1] = fcc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_values[i] = func(simplex[i])
                            if f_values[i] < best:
                                best = f_values[i]
                                best_params = simplex[i].copy()
            
            # Convergence check
            if np.std(f_values) < 1e-12:
                return
    
    def cmaes_search(start_point, budget_seconds):
        """Simplified CMA-ES"""
        nonlocal best, best_params
        
        cma_start = datetime.now()
        n = dim
        
        # Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        mean = start_point.copy()
        sigma = np.mean(upper - lower) * 0.3
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        if n <= 100:
            C = np.eye(n)
            use_full_cov = True
        else:
            diag_C = np.ones(n)
            use_full_cov = False
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        gen = 0
        while True:
            if (datetime.now() - cma_start).total_seconds() >= budget_seconds:
                return
            
            if use_full_cov:
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    D = np.sqrt(eigenvalues)
                    B = eigenvectors
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                if use_full_cov:
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
                else:
                    arx[k] = mean + sigma * np.sqrt(diag_C) * arz[k]
                arx[k] = np.clip(arx[k], lower, upper)
            
            # Evaluate
            fitness = np.full(lam, float('inf'))
            for k in range(lam):
                if (datetime.now() - cma_start).total_seconds() >= budget_seconds:
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
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            # Update evolution paths
            if use_full_cov:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ((mean - old_mean) / sigma / np.sqrt(diag_C))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance
            if use_full_cov:
                artmp = (arx[:mu] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (weights[:, None] * artmp).T @ artmp
                
                # Ensure symmetry
                C = (C + C.T) / 2
                np.fill_diagonal(C, np.maximum(np.diag(C), 1e-20))
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                diag_C = (1 - c1 - cmu_val) * diag_C + \
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C) + \
                         cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.mean(upper - lower))
            
            gen += 1
            
            if sigma < 1e-12:
                return
    
    def differential_evolution_search(budget_seconds):
        """Differential Evolution"""
        nonlocal best, best_params
        
        de_start = datetime.now()
        
        pop_size = min(max(10 * dim, 40), 200)
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = np.random.uniform(lower, upper, (pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        # Use some of the init samples
        sorted_idx = np.argsort(init_fitness)
        n_reuse = min(pop_size // 2, n_init)
        for i in range(n_reuse):
            pop[i] = init_pop[sorted_idx[i]]
            fit[i] = init_fitness[sorted_idx[i]]
        
        for i in range(n_reuse, pop_size):
            if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                return
            fit[i] = func(pop[i])
            if fit[i] < best:
                best = fit[i]
                best_params = pop[i].copy()
        
        while True:
            for i in range(pop_size):
                if (datetime.now() - de_start).total_seconds() >= budget_seconds:
                    return
                
                # Mutation: DE/best/1 with probability 0.5, else DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                if np.random.random() < 0.5:
                    # DE/best/1
                    best_idx = np.argmin(fit)
                    mutant = pop[best_idx] + F * (a - b)
                else:
                    mutant = a + F * (b - c)
                
                # Crossover
                cross_points = np.random.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial = np.clip(trial, lower, upper)
                
                f_trial = func(trial)
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    # Strategy: allocate time among methods
    remaining = time_left()
    if remaining <= 0:
        return best
    
    # Sort initial samples
    sorted_idx = np.argsort(init_fitness)
    
    # Allocate: 40% DE, 30% CMA-ES, 30% Nelder-Mead restarts
    remaining = time_left()
    if remaining > 0:
        de_budget = remaining * 0.35
        differential_evolution_search(de_budget)
    
    remaining = time_left()
    if remaining > 0:
        # CMA-ES from best known point
        cma_budget = remaining * 0.5
        if best_params is not None:
            cmaes_search(best_params.copy(), cma_budget)
        else:
            cmaes_search(init_pop[sorted_idx[0]].copy(), cma_budget)
    
    # Nelder-Mead restarts with remaining time
    remaining = time_left()
    restart = 0
    while remaining > 0.5:
        if best_params is not None:
            # Perturb the best point slightly for diversity on restarts
            if restart == 0:
                sp = best_params.copy()
                sf = best
            else:
                perturbation = np.random.randn(dim) * (upper - lower) * 0.1 * (0.5 ** (restart - 1))
                sp = np.clip(best_params + perturbation, lower, upper)
                sf = func(sp)
                if sf < best:
                    best = sf
                    best_params = sp.copy()
        else:
            sp = np.random.uniform(lower, upper)
            sf = func(sp)
            if sf < best:
                best = sf
                best_params = sp.copy()
        
        remaining = time_left()
        nm_budget = min(remaining * 0.8, remaining - 0.2)
        if nm_budget > 0.2:
            nelder_mead(sp, sf, nm_budget)
        
        restart += 1
        remaining = time_left()
    
    return best
