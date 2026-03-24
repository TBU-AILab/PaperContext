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
    
    # --- CMA-ES Implementation ---
    def run_cmaes(mean_init, sigma_init, budget_fraction=0.5):
        nonlocal best, best_params
        
        time_budget = max_time * budget_fraction
        cma_start = elapsed()
        
        n = dim
        lam = pop_size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = mean_init.copy()
        sigma = sigma_init
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_sep = (n > 50)
        
        if use_sep:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            eigeneval = 0
            B = np.eye(n)
            D = np.ones(n)
        
        gen = 0
        while elapsed() - cma_start < time_budget and elapsed() < max_time - 0.1:
            # Generate offspring
            solutions = []
            for i in range(lam):
                if use_sep:
                    z = np.random.randn(n)
                    x = mean + sigma * (np.sqrt(C_diag) * z)
                else:
                    z = np.random.randn(n)
                    x = mean + sigma * (B @ (D * z))
                x = clip(x)
                solutions.append(x)
            
            # Evaluate
            fitnesses = []
            for x in solutions:
                if elapsed() >= max_time - 0.05:
                    return
                f = eval_func(x)
                fitnesses.append(f)
            
            # Sort by fitness
            idx = np.argsort(fitnesses)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.zeros(n)
            for i in range(mu):
                mean += weights[i] * solutions[idx[i]]
            
            # Update evolution paths
            if use_sep:
                invsqrtC = 1.0 / np.sqrt(C_diag)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC * (mean - old_mean) / sigma)
            else:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance
            if use_sep:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (solutions[idx[i]] - old_mean) / sigma
                C_diag = (1 - c1 - cmu) * C_diag + \
                         c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag) + \
                         cmu * np.sum(weights[:, None] * artmp**2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (solutions[idx[i]] - old_mean) / sigma
                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * sum(weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Decompose C
            if not use_sep:
                eigeneval += lam
                if eigeneval >= lam * 10:
                    eigeneval = 0
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                    except:
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
            
            gen += 1
            
            # Check for convergence
            if sigma < 1e-12:
                break
    
    # --- Phase 2: Latin Hypercube Sampling for initial points ---
    n_init = min(max(10 * dim, 50), 200)
    init_points = []
    init_fits = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.1:
            break
        x = np.array([np.random.uniform(l, u) for l, u in zip(lower, upper)])
        f = eval_func(x)
        init_points.append(x)
        init_fits.append(f)
    
    if not init_points:
        return best
    
    # Sort initial points
    sorted_idx = np.argsort(init_fits)
    
    # --- Phase 3: Run CMA-ES from best initial points ---
    n_restarts = max(1, min(5, int(max_time / 2)))
    time_per_restart = (max_time - elapsed()) / max(n_restarts, 1)
    
    for restart in range(n_restarts):
        if elapsed() >= max_time - 0.2:
            break
        
        # Pick starting point
        if restart < len(sorted_idx):
            start_idx = sorted_idx[restart]
            mean_init = init_points[start_idx].copy()
        else:
            mean_init = np.array([np.random.uniform(l, u) for l, u in zip(lower, upper)])
        
        # Initial sigma: fraction of range
        sigma_init = np.mean(upper - lower) * 0.3 / (restart + 1)
        
        remaining = max_time - elapsed()
        if remaining < 0.2:
            break
        
        budget = min(time_per_restart, remaining - 0.1) / max_time
        run_cmaes(mean_init, sigma_init, budget_fraction=budget)
    
    # --- Phase 4: Local refinement around best found ---
    if best_params is not None and elapsed() < max_time - 0.5:
        # Nelder-Mead style local search
        step = np.mean(upper - lower) * 0.01
        current = best_params.copy()
        current_fit = best
        
        while elapsed() < max_time - 0.1:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time - 0.05:
                    break
                
                # Try positive direction
                trial = current.copy()
                trial[i] = min(trial[i] + step, upper[i])
                f = eval_func(trial)
                if f < current_fit:
                    current = trial
                    current_fit = f
                    improved = True
                    continue
                
                # Try negative direction
                trial = current.copy()
                trial[i] = max(trial[i] - step, lower[i])
                f = eval_func(trial)
                if f < current_fit:
                    current = trial
                    current_fit = f
                    improved = True
            
            if not improved:
                step *= 0.5
                if step < 1e-15:
                    # Random restart local search
                    if best_params is not None:
                        current = best_params + np.random.randn(dim) * np.mean(upper - lower) * 0.001
                        current = clip(current)
                        current_fit = eval_func(current)
                        step = np.mean(upper - lower) * 0.001
    
    return best
#
#Wait, I realize there's an issue with the algorithm structure. The CMA-ES phases are defined but the initial sampling (Phase 2) runs first, then CMA-ES (Phase 3). Let me restructure this properly:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
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
    
    # Phase 1: Initial sampling
    n_init = min(max(10 * dim, 50), 300)
    init_points = []
    init_fits = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.15:
            break
        x = np.array([np.random.uniform(l, u) for l, u in zip(lower, upper)])
        f = eval_func(x)
        init_points.append(x.copy())
        init_fits.append(f)
    
    if not init_points:
        return best
    
    sorted_idx = np.argsort(init_fits)
    
    # Phase 2: CMA-ES from best points
    def run_cmaes(mean_init, sigma_init, time_limit):
        nonlocal best, best_params
        
        cma_start = elapsed()
        n = dim
        lam = 4 + int(3 * np.log(n))
        if lam % 2 != 0:
            lam += 1
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = mean_init.copy()
        sigma = sigma_init
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_sep = (n > 40)
        
        if use_sep:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            update_count = 0
        
        gen = 0
        while elapsed() - cma_start < time_limit and elapsed() < max_time - 0.05:
            solutions = []
            for _ in range(lam):
                if use_sep:
                    z = np.random.randn(n)
                    x = mean + sigma * (np.sqrt(np.maximum(C_diag, 1e-20)) * z)
                else:
                    z = np.random.randn(n)
                    x = mean + sigma * (B @ (D * z))
                solutions.append(clip(x))
            
            fitnesses = []
            for x in solutions:
                if elapsed() >= max_time - 0.05:
                    return
                fitnesses.append(eval_func(x))
            
            idx = np.argsort(fitnesses)
            old_mean = mean.copy()
            mean = sum(weights[i] * solutions[idx[i]] for i in range(mu))
            
            diff = (mean - old_mean) / sigma
            
            if use_sep:
                invsqrtC_diff = diff / np.sqrt(np.maximum(C_diag, 1e-20))
            else:
                invsqrtC_diff = B @ (1.0/D * (B.T @ diff))
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC_diff
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig *
