#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    # --- Phase 1: CMA-ES inspired search ---
    # Initialize population
    pop_size = min(4 + int(3 * np.log(dim)), 50)
    if pop_size % 2 != 0:
        pop_size += 1
    
    # Multiple restarts with different strategies
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x_clipped = clip(x)
        f = func(x_clipped)
        if f < best:
            best = f
            best_x = x_clipped.copy()
        return f
    
    # Latin Hypercube initial sampling
    n_init = min(max(10 * dim, 50), 200)
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = lower + np.random.rand(dim) * ranges
        eval_func(x)
    
    # CMA-ES implementation
    def run_cmaes(x0, sigma0, budget_fraction):
        nonlocal best, best_x
        
        time_limit = start + timedelta(seconds=max_time * budget_fraction)
        
        n = dim
        lam = pop_size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # State variables
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_diag = dim > 50
        
        if use_diag:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            eigeneval = 0
            B = np.eye(n)
            D = np.ones(n)
        
        gen = 0
        stagnation = 0
        prev_best = float('inf')
        
        while datetime.now() < time_limit:
            gen += 1
            
            # Generate offspring
            arx = np.zeros((lam, n))
            arz = np.zeros((lam, n))
            
            for k in range(lam):
                if use_diag:
                    arz[k] = np.random.randn(n)
                    arx[k] = mean + sigma * np.sqrt(C_diag) * arz[k]
                else:
                    arz[k] = np.random.randn(n)
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
            
            # Evaluate
            arfitness = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= max_time * 0.95:
                    return
                arfitness[k] = eval_func(arx[k])
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Check stagnation
            if arfitness[arindex[0]] < prev_best - 1e-12:
                stagnation = 0
                prev_best = arfitness[arindex[0]]
            else:
                stagnation += 1
            
            if stagnation > 10 + 30 * n / lam:
                return  # restart
            
            # Recombination
            xold = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            # Update evolution paths
            if use_diag:
                invsqrtC = 1.0 / np.sqrt(C_diag)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC * (mean - xold) / sigma
            else:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - xold)) / sigma
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - xold) / sigma
            
            # Update covariance
            artmp = (arx[arindex[:mu]] - xold) / sigma
            
            if use_diag:
                C_diag = ((1 - c1 - cmu) * C_diag 
                         + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * C_diag)
                         + cmu * np.sum(weights[:, None] * artmp ** 2, axis=0))
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                C = ((1 - c1 - cmu) * C
                     + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                     + cmu * (artmp.T @ np.diag(weights) @ artmp))
                
                # Eigen decomposition
                eigeneval += 1
                if eigeneval >= lam / (c1 + cmu) / n / 10 or gen == 1:
                    eigeneval = 0
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                    except:
                        return  # restart on numerical error
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            if sigma < 1e-16:
                return  # converged
    
    # Differential Evolution phase
    def run_de(budget_time_end):
        nonlocal best, best_x
        
        de_pop_size = min(max(10 * dim, 30), 100)
        
        # Initialize population
        pop = np.zeros((de_pop_size, dim))
        fit = np.zeros(de_pop_size)
        
        for i in range(de_pop_size):
            if elapsed() >= max_time * 0.95:
                return
            pop[i] = lower + np.random.rand(dim) * ranges
            if best_x is not None and i == 0:
                pop[i] = best_x.copy()
            fit[i] = eval_func(pop[i])
        
        F = 0.8
        CR = 0.9
        
        while elapsed() < budget_time_end:
            for i in range(de_pop_size):
                if elapsed() >= max_time * 0.95:
                    return
                
                # Mutation: DE/best/1 with some DE/rand/1
                idxs = list(range(de_pop_size))
                idxs.remove(i)
                
                if np.random.rand() < 0.5 and best_x is not None:
                    # DE/best/1
                    a, b = np.random.choice(idxs, 2, replace=False)
                    Fi = F + 0.1 * np.random.randn()
                    mutant = best_x + Fi * (pop[a] - pop[b])
                else:
                    # DE/rand/1
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    Fi = F + 0.1 * np.random.randn()
                    mutant = pop[a] + Fi * (pop[b] - pop[c])
                
                # Crossover
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == jrand:
                        trial[j] = mutant[j]
                
                trial = clip(trial)
                f_trial = eval_func(trial)
                
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    # Nelder-Mead local search
    def run_nelder_mead(x0, budget_time_end):
        nonlocal best, best_x
        
        n = dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        
        step = ranges * 0.05
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += step[i] if step[i] > 1e-10 else 0.01
        
        simplex = clip(simplex)
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if elapsed() >= max_time * 0.95:
                return
            f_simplex[i] = eval_func(simplex[i])
        
        while elapsed() < budget_time_end:
            # Sort
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            if elapsed() >= max_time * 0.95:
                return
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                if elapsed() >= max_time * 0.95:
                    return
                fe = eval_func(xe)
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
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                if elapsed() >= max_time * 0.95:
                    return
                fc = eval_func(xc)
                
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        if elapsed() >= max_time * 0.95:
                            return
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_func(simplex[i])
            
            # Check convergence
            if np.std(f_simplex) < 1e-15:
                return
    
    # Strategy: allocate time between methods
    total_time = max_time
    
    # Run CMA-ES with restarts (60% of time)
    cmaes_end = 0.60
    restart_count = 0
    while elapsed() < total_time * cmaes_end:
        if elapsed() >= max_time * 0.95:
            return best
        
        if best_x is not None and restart_count > 0 and np.random.rand() < 0.3:
            x0 = best_x + 0.1 * ranges * np.random.randn(dim)
            x0 = clip(x0)
            sigma0 = 0.1 * np.max(ranges)
        else:
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.3 * np.max(ranges)
        
        run_cmaes(x0, sigma0, cmaes_end)
        restart_count += 1
    
    # Run DE (25% of time)
    de_end_time = total_time * 0.85
    if elapsed() < de_end_time:
        run_de(de_end_time)
    
    # Run Nelder-Mead local search from best (remaining time)
    if best_x is not None and elapsed() < total_time * 0.95:
        run_nelder_mead(best_x.copy(), total_time * 0.95)
    
    return best
