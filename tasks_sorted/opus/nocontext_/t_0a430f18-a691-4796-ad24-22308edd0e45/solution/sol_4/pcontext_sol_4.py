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
    
    # --- Phase 1: CMA-ES inspired search ---
    # Initialize population
    pop_size = min(4 + int(3 * np.log(dim)), 50)
    if pop_size % 2 != 0:
        pop_size += 1
    
    # Multiple restarts with different strategies
    def time_remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    # Initial Latin Hypercube-like sampling
    n_init = min(pop_size * 5, max(100, dim * 10))
    for i in range(n_init):
        if time_remaining() < 0.1:
            return best
        params = lower + np.random.random(dim) * (upper - lower)
        fitness = func(params)
        if fitness < best:
            best = fitness
            best_params = params.copy()
    
    # CMA-ES implementation
    def run_cmaes(mean, sigma, budget_fraction=1.0):
        nonlocal best, best_params
        
        lam = pop_size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(dim+1)) - 1) + cs
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = dim <= 50
        
        if use_full_cov:
            C = np.eye(dim)
            eigeneval = 0
            B = np.eye(dim)
            D = np.ones(dim)
        else:
            diagC = np.ones(dim)
        
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        current_mean = mean.copy()
        current_sigma = sigma
        
        gen = 0
        no_improvement_count = 0
        prev_best = best
        
        while True:
            if time_remaining() < 0.2:
                return
            
            gen += 1
            
            # Generate offspring
            arx = np.zeros((lam, dim))
            arz = np.zeros((lam, dim))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_remaining() < 0.1:
                    return
                arz[k] = np.random.randn(dim)
                if use_full_cov:
                    arx[k] = current_mean + current_sigma * (B @ (D * arz[k]))
                else:
                    arx[k] = current_mean + current_sigma * (np.sqrt(diagC) * arz[k])
                arx[k] = clip_to_bounds(arx[k])
                arfitness[k] = func(arx[k])
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_params = arx[k].copy()
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = current_mean.copy()
            current_mean = np.zeros(dim)
            for k in range(mu):
                current_mean += weights[k] * arx[arindex[k]]
            
            # CSA
            if use_full_cov:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (current_mean - old_mean) / current_sigma)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ((current_mean - old_mean) / current_sigma / np.sqrt(diagC))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen)) / chiN < 1.4 + 2/(dim+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (current_mean - old_mean) / current_sigma
            
            # Covariance matrix adaptation
            if use_full_cov:
                artmp = (1.0 / current_sigma) * (arx[arindex[:mu]] - old_mean)
                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * (artmp.T @ np.diag(weights) @ artmp)
                
                eigeneval += 1
                if eigeneval >= 1.0 / (c1 + cmu) / dim / 10:
                    eigeneval = 0
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                    except:
                        C = np.eye(dim)
                        D = np.ones(dim)
                        B = np.eye(dim)
            else:
                artmp = (1.0 / current_sigma) * (arx[arindex[:mu]] - old_mean)
                diagC = (1 - c1 - cmu) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu * np.sum(weights[:, None] * artmp**2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            
            # Step size update
            current_sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            current_sigma = min(current_sigma, np.max(upper - lower))
            
            # Check for stagnation
            if best >= prev_best:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                prev_best = best
            
            if no_improvement_count > 10 + 30 * dim / lam:
                return
            
            if current_sigma < 1e-12:
                return
    
    # Run CMA-ES with restarts
    restart_count = 0
    while time_remaining() > 0.5:
        if restart_count == 0 and best_params is not None:
            init_mean = best_params.copy()
            init_sigma = 0.3 * np.max(upper - lower)
        else:
            # IPOP: increase population size on restarts
            if restart_count > 0:
                pop_size_new = int(pop_size * (2 ** (restart_count * 0.5)))
                pop_size_new = min(pop_size_new, 256)
            
            if np.random.random() < 0.5 and best_params is not None:
                # Local restart near best
                init_mean = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
                init_mean = clip_to_bounds(init_mean)
                init_sigma = 0.1 * np.max(upper - lower) * (0.5 ** restart_count)
                init_sigma = max(init_sigma, 1e-6)
            else:
                # Random restart
                init_mean = lower + np.random.random(dim) * (upper - lower)
                init_sigma = 0.3 * np.max(upper - lower)
        
        run_cmaes(init_mean, init_sigma)
        restart_count += 1
    
    # Final local Nelder-Mead style refinement if time permits
    if time_remaining() > 0.3 and best_params is not None:
        # Simple pattern search around best
        step = 0.001 * (upper - lower)
        current = best_params.copy()
        while time_remaining() > 0.1:
            improved = False
            for i in range(dim):
                if time_remaining() < 0.05:
                    return best
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[i] = trial[i] + direction * step[i]
                    trial = clip_to_bounds(trial)
                    f = func(trial)
                    if f < best:
                        best = f
                        best_params = trial.copy()
                        current = trial.copy()
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step) < 1e-15:
                    break
    
    return best
