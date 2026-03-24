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
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params
        x_clipped = clip(x)
        f = func(x_clipped)
        if f < best:
            best = f
            best_params = x_clipped.copy()
        return f
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(dim * 10, max(pop_size * 2, 20))
    init_params = []
    init_fitness = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        params = lower + np.random.rand(dim) * (upper - lower)
        f = eval_func(params)
        init_params.append(params.copy())
        init_fitness.append(f)
    
    # Sort and keep best
    sorted_idx = np.argsort(init_fitness)
    
    # Phase 2: CMA-ES
    def run_cmaes(mean_init, sigma_init, time_budget):
        nonlocal best, best_params
        
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        mean = mean_init.copy()
        sigma = sigma_init
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        if dim <= 200:
            C = np.eye(dim)
            use_full_cov = True
        else:
            # Use diagonal CMA for high dimensions
            C_diag = np.ones(dim)
            use_full_cov = False
        
        eigeneval = 0
        count_eval = 0
        
        cma_start = datetime.now()
        
        while True:
            if elapsed() >= max_time * 0.95:
                return
            
            if (datetime.now() - cma_start).total_seconds() >= time_budget:
                return
            
            # Eigendecomposition
            if use_full_cov:
                if count_eval - eigeneval > pop_size / (c1 + cmu) / dim / 10:
                    eigeneval = count_eval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        C = np.eye(dim)
                        D = np.ones(dim)
                        B = np.eye(dim)
                        invsqrtC = np.eye(dim)
                else:
                    if 'B' not in dir():
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
            
            # Generate offspring
            offspring = []
            offspring_fitness = []
            
            for k in range(pop_size):
                if elapsed() >= max_time * 0.95:
                    return
                
                if use_full_cov:
                    z = np.random.randn(dim)
                    x = mean + sigma * (B @ (D * z))
                else:
                    z = np.random.randn(dim)
                    x = mean + sigma * (np.sqrt(C_diag) * z)
                
                f = eval_func(x)
                offspring.append(x.copy())
                offspring_fitness.append(f)
                count_eval += 1
            
            # Sort by fitness
            idx = np.argsort(offspring_fitness)
            
            # Update mean
            old_mean = mean.copy()
            selected = np.array([offspring[idx[i]] for i in range(mu)])
            mean = np.sum(weights[:, None] * selected, axis=0)
            
            # Update evolution paths
            mean_diff = mean - old_mean
            
            if use_full_cov:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ mean_diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * mean_diff / (sigma * np.sqrt(C_diag))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * count_eval / pop_size)) / chiN < 1.4 + 2 / (dim + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * mean_diff / sigma
            
            # Update covariance
            if use_full_cov:
                artmp = (selected - old_mean) / sigma
                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * (artmp.T @ np.diag(weights) @ artmp)
            else:
                artmp = (selected - old_mean) / sigma
                C_diag = (1 - c1 - cmu) * C_diag + \
                    c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * C_diag) + \
                    cmu * np.sum(weights[:, None] * artmp ** 2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Check for stagnation
            if sigma < 1e-16:
                return
    
    # Run CMA-ES with multiple restarts
    remaining_time = max_time - elapsed()
    
    if remaining_time > 0.5:
        # First run from best initial point
        best_init_idx = sorted_idx[0]
        sigma0 = np.mean(upper - lower) / 4.0
        
        run_cmaes(init_params[best_init_idx], sigma0, remaining_time * 0.4)
    
    # Restart from different points
    restart_count = 0
    while elapsed() < max_time * 0.90:
        restart_count += 1
        remaining = max_time - elapsed()
        if remaining < 0.5:
            break
        
        # Choose restart point: mix of best known and random
        if restart_count % 3 == 0 and best_params is not None:
            # Perturb best known
            perturbation = 0.3 * (upper - lower) * np.random.randn(dim)
            start_point = clip(best_params + perturbation)
            sigma0 = np.mean(upper - lower) / 6.0
        elif restart_count % 3 == 1:
            # Random restart
            start_point = lower + np.random.rand(dim) * (upper - lower)
            sigma0 = np.mean(upper - lower) / 3.0
        else:
            # From one of the top initial samples
            idx_choice = sorted_idx[np.random.randint(0, min(5, len(sorted_idx)))]
            start_point = init_params[idx_choice].copy()
            sigma0 = np.mean(upper - lower) / 4.0
        
        time_for_this_run = min(remaining * 0.35, remaining - 0.3)
        run_cmaes(start_point, sigma0, time_for_this_run)
    
    # Phase 3: Local Nelder-Mead style refinement around best
    if best_params is not None and elapsed() < max_time * 0.95:
        # Simple pattern search refinement
        step = np.mean(upper - lower) * 0.01
        current = best_params.copy()
        current_f = best
        
        while elapsed() < max_time * 0.95 and step > 1e-15:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                
                # Try positive direction
                trial = current.copy()
                trial[i] = min(trial[i] + step, upper[i])
                f = eval_func(trial)
                if f < current_f:
                    current = trial
                    current_f = f
                    improved = True
                    continue
                
                # Try negative direction
                trial = current.copy()
                trial[i] = max(trial[i] - step, lower[i])
                f = eval_func(trial)
                if f < current_f:
                    current = trial
                    current_f = f
                    improved = True
            
            if not improved:
                step *= 0.5
    
    return best
