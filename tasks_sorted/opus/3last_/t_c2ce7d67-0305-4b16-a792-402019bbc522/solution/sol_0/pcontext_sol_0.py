#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    # --- CMA-ES inspired + multi-restart approach ---
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    best_x = None
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(200, max(50, dim * 10))
    init_samples = np.random.uniform(0, 1, (n_init, dim))
    for i in range(n_init):
        if elapsed() >= max_time:
            return best
        x = lower + init_samples[i] * ranges
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
    
    # Phase 2: CMA-ES with restarts
    def run_cmaes(x0, sigma0, pop_size, budget_time):
        nonlocal best, best_x
        
        cma_start = datetime.now()
        n = len(x0)
        
        # CMA-ES parameters
        lam = pop_size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step-size control
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        E_norm = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        
        # Covariance matrix adaptation
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        
        # State variables
        mean = x0.copy()
        sigma = sigma0
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = (n <= 100)
        
        if use_full_cov:
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        else:
            # Separable CMA-ES: only diagonal
            diagC = np.ones(n)
        
        gen = 0
        no_improve_count = 0
        local_best = best
        
        while True:
            if (datetime.now() - cma_start).total_seconds() >= budget_time:
                break
            if elapsed() >= max_time:
                break
            
            gen += 1
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            
            if use_full_cov:
                # Update eigen decomposition periodically
                if gen == 1 or (gen - eigeneval) > (lam / (c1 + cmu) / n / 10):
                    eigeneval = gen
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_vals, B = np.linalg.eigh(C)
                        D_vals = np.maximum(D_vals, 1e-20)
                        D = np.sqrt(D_vals)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        C = np.eye(n)
                        D = np.ones(n)
                        B = np.eye(n)
                        invsqrtC = np.eye(n)
                
                arx = mean + sigma * (arz @ (B * D).T)
            else:
                sqrtDiagC = np.sqrt(diagC)
                arx = mean + sigma * (arz * sqrtDiagC)
            
            # Clip to bounds
            arx = np.clip(arx, lower, upper)
            
            # Evaluate
            fitvals = np.full(lam, float('inf'))
            for i in range(lam):
                if elapsed() >= max_time:
                    return
                fitvals[i] = func(arx[i])
                if fitvals[i] < best:
                    best = fitvals[i]
                    best_x = arx[i].copy()
            
            # Sort
            idx = np.argsort(fitvals)
            
            # Check improvement
            if fitvals[idx[0]] < local_best - 1e-12:
                local_best = fitvals[idx[0]]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[idx[:mu]]
            mean = weights @ selected
            
            # Clip mean
            mean = np.clip(mean, lower, upper)
            
            diff = mean - old_mean
            
            if use_full_cov:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff / (sigma * sqrtDiagC)
            
            # Heaviside function
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / E_norm) < (1.4 + 2.0 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff / sigma
            
            # Covariance matrix update
            if use_full_cov:
                artmp = (selected - old_mean) / sigma
                C = ((1 - c1 - cmu) * C +
                     c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                     cmu * (artmp.T @ np.diag(weights) @ artmp))
            else:
                artmp = (selected - old_mean) / sigma
                diagC = ((1 - c1 - cmu) * diagC +
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diagC) +
                         cmu * np.sum(weights[:, None] * artmp ** 2, axis=0))
                diagC = np.maximum(diagC, 1e-20)
            
            # Step size update
            sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / E_norm - 1))
            sigma = min(sigma, np.max(ranges))
            
            # Termination conditions
            if sigma < 1e-12:
                break
            if no_improve_count > 10 + 30 * n // lam:
                break
            if use_full_cov and np.max(D_vals) > 1e7 * np.min(D_vals):
                break
    
    # Run CMA-ES with restarts
    restart = 0
    while remaining() > 0.1:
        pop_multiplier = min(2 ** restart, 20)
        pop_size = max(int(4 + 3 * np.log(dim)), 10) * pop_multiplier
        pop_size = min(pop_size, 500)
        
        budget_time = remaining() * 0.8 if restart == 0 else remaining() * 0.5
        
        if restart == 0 and best_x is not None:
            x0 = best_x.copy()
            sigma0 = 0.3 * np.mean(ranges)
        else:
            # Random restart with some bias towards best known
            if best_x is not None and np.random.rand() < 0.3:
                x0 = best_x + 0.5 * ranges * np.random.randn(dim)
                x0 = np.clip(x0, lower, upper)
                sigma0 = 0.2 * np.mean(ranges)
            else:
                x0 = lower + np.random.rand(dim) * ranges
                sigma0 = 0.3 * np.mean(ranges)
        
        run_cmaes(x0, sigma0, pop_size, budget_time)
        restart += 1
    
    # Phase 3: Local refinement with Nelder-Mead-like search around best
    if best_x is not None and remaining() > 0.05:
        step = 0.001 * ranges
        while remaining() > 0.02:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time:
                    return best
                x_try = best_x.copy()
                x_try[i] = min(best_x[i] + step[i], upper[i])
                f = func(x_try)
                if f < best:
                    best = f
                    best_x = x_try.copy()
                    improved = True
                    continue
                
                x_try[i] = max(best_x[i] - step[i], lower[i])
                f = func(x_try)
                if f < best:
                    best = f
                    best_x = x_try.copy()
                    improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break
    
    return best
