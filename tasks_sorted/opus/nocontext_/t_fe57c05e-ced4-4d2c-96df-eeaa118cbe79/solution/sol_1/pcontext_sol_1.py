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
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Latin Hypercube initial sampling
    n_initial = min(pop_size * 3, max(20, dim * 2))
    for i in range(n_initial):
        if elapsed() >= max_time * 0.95:
            return best
        x = lower + np.random.rand(dim) * (upper - lower)
        eval_func(x)
    
    # CMA-ES implementation
    def run_cmaes(mean_init, sigma_init, budget_fraction):
        nonlocal best, best_params
        
        target_end = elapsed() + (max_time - elapsed()) * budget_fraction
        
        mean = mean_init.copy()
        sigma = sigma_init
        
        lam = pop_size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = dim <= 50
        
        if use_full_cov:
            C = np.eye(dim)
            invsqrtC = np.eye(dim)
            eigeneval = 0
        else:
            diag_C = np.ones(dim)
        
        generation = 0
        
        while elapsed() < target_end and elapsed() < max_time * 0.95:
            # Generate offspring
            arx = np.zeros((lam, dim))
            arz = np.zeros((lam, dim))
            arf = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= max_time * 0.95:
                    return
                
                arz[k] = np.random.randn(dim)
                if use_full_cov:
                    arx[k] = mean + sigma * (invsqrtC @ arz[k] if False else np.dot(C_sqrt, arz[k]) if False else arz[k])
                    # Actually use eigendecomposition properly
                    arx[k] = mean + sigma * np.dot(BD, arz[k]) if 'BD' in dir() else mean + sigma * arz[k]
                else:
                    arx[k] = mean + sigma * np.sqrt(diag_C) * arz[k]
                
                arx[k] = clip(arx[k])
                arf[k] = eval_func(arx[k])
            
            # Sort by fitness
            idx = np.argsort(arf)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * arx[idx[i]]
            
            # Update evolution paths
            mean_diff = mean - old_mean
            
            if use_full_cov:
                # Simplified: skip full covariance, use sep-CMA
                pass
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * mean_diff / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chi_n < 1.4 + 2 / (dim + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * mean_diff / sigma
            
            if not use_full_cov:
                # Sep-CMA-ES update
                diag_C = (1 - c1 - cmu) * diag_C + \
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C)
                for i in range(mu):
                    diff = (arx[idx[i]] - old_mean) / sigma
                    diag_C += cmu * weights[i] * diff ** 2
                diag_C = np.maximum(diag_C, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, np.max(upper - lower))
            sigma = max(sigma, 1e-20)
            
            generation += 1
            
            # Check for convergence
            if sigma < 1e-12:
                return
    
    # Run CMA-ES with restarts
    restart = 0
    while elapsed() < max_time * 0.90:
        remaining = max_time - elapsed()
        if remaining < 0.1:
            break
        
        if restart == 0 and best_params is not None:
            mean_init = best_params.copy()
            sigma_init = 0.3 * np.mean(upper - lower)
        else:
            mean_init = lower + np.random.rand(dim) * (upper - lower)
            sigma_init = 0.3 * np.mean(upper - lower)
        
        budget_frac = min(0.5, max(0.15, 1.0 / (restart + 2)))
        run_cmaes(mean_init, sigma_init, budget_frac)
        restart += 1
    
    # --- Phase 2: Differential Evolution style refinement ---
    # Use Nelder-Mead simplex for final refinement if time permits
    if best_params is not None and elapsed() < max_time * 0.95:
        # Local search around best
        step = 0.01 * (upper - lower)
        current = best_params.copy()
        current_f = best
        
        while elapsed() < max_time * 0.95:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                
                # Try positive step
                trial = current.copy()
                trial[i] = min(current[i] + step[i], upper[i])
                f = eval_func(trial)
                if f < current_f:
                    current = trial
                    current_f = f
                    improved = True
                    continue
                
                # Try negative step
                trial = current.copy()
                trial[i] = max(current[i] - step[i], lower[i])
                f = eval_func(trial)
                if f < current_f:
                    current = trial
                    current_f = f
                    improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step) < 1e-15:
                    break
    
    return best
#
#Wait, let me reconsider. The CMA-ES implementation above has some issues (referencing undefined variables like `BD`, `C_sqrt`). Let me provide a clean, working implementation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Separable CMA-ES with restarts ---
    def run_sep_cmaes(mean_init, sigma_init, time_limit):
        nonlocal best, best_params
        
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        mean = mean_init.copy()
        sigma = sigma_init
        diag_C = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        gen = 0
        
        end_time = elapsed() + time_limit
        
        while elapsed() < end_time and elapsed() < max_time * 0.92:
            arx = np.zeros((lam, dim))
            arf = np.zeros(lam)
            
            sqrt_diag = np.sqrt(diag_C)
            
            for k in range(lam):
                if elapsed() >= max_time * 0.95:
                    return
                z = np.random.randn(dim)
                arx[k] = mean + sigma * sqrt_diag * z
                arx[k] = clip(arx[k])
                arf[k] = eval_func(arx[k])
            
            idx = np.argsort(arf)
            old_mean = mean.copy()
            
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * arx[idx[i]]
            
            diff = (mean - old_mean) / sigma
            invsqrt_diag = 1.0 / np.maximum(sqrt_diag, 1e-30)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrt_diag * diff
            
            norm_ps = np.linalg.norm(ps)
            hsig = float(norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (dim + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff
            
            new_diag = (1 - c1 - cmu_val) * diag_C
            new_diag += c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag_C)
            for i in range(mu):
                d = (arx[idx[i]] - old_mean) / sigma
                new_diag += cmu_val * weights[i] * d ** 2
            diag_C = np.maximum(new_diag, 1e-20)
            
            sigma *= np.exp((cs / ds) * (norm_ps / chi_n - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 2)
            
            gen += 1
            
            if sigma * np.max(np.sqrt(diag_C)) < 1e-12:
                return
    
    # Initial sampling
    n_init = min(max(2 * dim, 20), 200)
    for _ in range(n_init):
        if elapsed() >= max_time * 0.1:
            break
        x = lower + np.random.rand(dim) * ranges
        eval_func(x)
    
    # CMA-ES restarts
    restart = 0
    while elapsed() < max_time * 0.85:
        remaining = max_time - elapsed()
        if remaining < 0.2:
            break
        
        if restart == 0 and best_params is not None:
            mean_init = best_params.copy()
            sigma_init = 0.2 * np.mean(ranges)
        elif restart % 3 == 0 and best_params is not None:
            # Restart near best with small sigma
            mean_init = best_params + 0.1 * ranges * np.random.randn(dim)
            mean_init = clip(mean_init)
            sigma_init = 0.1 * np.mean(ranges)
        else:
            mean_init = lower + np.random.rand(dim) * ranges
            sigma_init = 0.3 * np.mean(ranges)
        
        time_budget = max(0.5, (max_time - elapsed()) * 0.3)
        run_sep_cmaes(mean_init, sigma_init, time_budget)
        restart += 1
    
    # Local coordinate descent refinement
    if best_params is not None:
        current = best_params.copy()
        current_f = best
        step = 0.001 * ranges
        
        while elapsed() < max_time * 0.98:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                
                trial = current.copy()
                trial[i] = min(current[i] + step[i], upper[i])
                f = eval_func(trial)
                if f < current_f:
                    current = trial
                    current_f = f
                    step[i] *= 1.5
                    improved = True
                    continue
                
                trial = current.copy()
                trial[i] = max(current[i] - step[i], lower[i])
                f = eval_func(trial)
                if f < current_f:
                    current = trial
                    current_f = f
                    step[i] *= 1.5
                    improved = True
                    continue
                
                step[i] *= 0.5
            
            if not improved:
                if np.max(step / ranges) < 1e-14:
                    break
    
    return best
