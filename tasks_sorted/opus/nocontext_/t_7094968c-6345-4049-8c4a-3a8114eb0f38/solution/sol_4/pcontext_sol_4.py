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
    
    # Initial sampling - Latin Hypercube-like
    mu = pop_size // 2  # number of parents
    
    # Weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)
    
    # CMA-ES parameters
    sigma = 0.3
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    # Initialize mean in center or random
    mean = lower + 0.5 * ranges
    
    # Evaluate center
    fitness_center = func(np.clip(mean, lower, upper))
    if fitness_center < best:
        best = fitness_center
        best_x = mean.copy()
    
    # Also try a few random points initially
    n_init = min(max(10, pop_size * 2), 100)
    for _ in range(n_init):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        x = np.random.uniform(lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
    
    mean = best_x.copy() if best_x is not None else lower + 0.5 * ranges
    
    # CMA-ES state
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    
    if dim <= 200:
        C = np.eye(dim)
        use_full_cov = True
    else:
        # For high dimensions, use diagonal approximation
        C_diag = np.ones(dim)
        use_full_cov = False
    
    eigeneval = 0
    count_eval = 0
    
    generation = 0
    stagnation_count = 0
    last_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.92):
            break
        
        # Update eigen decomposition periodically
        if use_full_cov:
            if generation % max(1, int(1 / (c1 + cmu) / dim / 10)) == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)
                    invsqrtC = np.eye(dim)
            
            # Generate offspring
            arx = np.zeros((pop_size, dim))
            arz = np.zeros((pop_size, dim))
            for k in range(pop_size):
                arz[k] = np.random.randn(dim)
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], lower, upper)
        else:
            D = np.sqrt(np.maximum(C_diag, 1e-20))
            arx = np.zeros((pop_size, dim))
            arz = np.zeros((pop_size, dim))
            for k in range(pop_size):
                arz[k] = np.random.randn(dim)
                arx[k] = mean + sigma * D * arz[k]
                arx[k] = np.clip(arx[k], lower, upper)
        
        # Evaluate
        fitnesses = np.zeros(pop_size)
        for k in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.92):
                return best
            fitnesses[k] = func(arx[k])
            count_eval += 1
            if fitnesses[k] < best:
                best = fitnesses[k]
                best_x = arx[k].copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        
        # Recombination: new mean
        old_mean = mean.copy()
        mean = np.zeros(dim)
        for i in range(mu):
            mean += weights[i] * arx[idx[i]]
        
        # Normalize step in original space
        mean_shift = (mean - old_mean) / sigma
        
        if use_full_cov:
            # CSA: cumulation for sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ mean_shift)
            
            # Heaviside function
            hs_left = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (count_eval / pop_size + 1)))
            hs = 1 if hs_left < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # CCA: cumulation for rank-one update
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * mean_shift
            
            # Covariance matrix adaptation
            artmp = np.zeros((mu, dim))
            for i in range(mu):
                artmp[i] = (arx[idx[i]] - old_mean) / sigma
            
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + \
                cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Sigma adaptation
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        else:
            # Diagonal CMA-ES
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * mean_shift / D
            
            hs_left = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (count_eval / pop_size + 1)))
            hs = 1 if hs_left < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * mean_shift
            
            artmp = np.zeros((mu, dim))
            for i in range(mu):
                artmp[i] = (arx[idx[i]] - old_mean) / sigma
            
            C_diag = (1 - c1 - cmu) * C_diag + \
                     c1 * (pc**2 + (1 - hs) * cc * (2 - cc) * C_diag) + \
                     cmu * np.sum(weights[:, None] * artmp**2, axis=0)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        
        # Clamp sigma
        sigma = max(sigma, 1e-20)
        sigma = min(sigma, np.max(ranges))
        
        generation += 1
        
        # Check stagnation
        if best >= last_best * (1 - 1e-12):
            stagnation_count += 1
        else:
            stagnation_count = 0
            last_best = best
        
        # Restart if stagnated
        if stagnation_count > 20 + dim:
            # Restart with smaller sigma around best
            if np.random.rand() < 0.5 and best_x is not None:
                mean = best_x.copy() + 0.1 * ranges * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
                sigma = 0.1
            else:
                mean = np.random.uniform(lower, upper)
                sigma = 0.3
            
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            if use_full_cov:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                invsqrtC = np.eye(dim)
            else:
                C_diag = np.ones(dim)
            stagnation_count = 0
    
    # --- Phase 2: Local refinement with Nelder-Mead-like search around best ---
    if best_x is not None:
        # Quick local search
        step = 0.001 * ranges
        improved = True
        while improved:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                break
            improved = False
            for i in range(dim):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[i] = np.clip(trial[i] + direction * step[i], lower[i], upper[i])
                    f = func(trial)
                    if f < best:
                        best = f
                        best_x = trial.copy()
                        improved = True
    
    return best
