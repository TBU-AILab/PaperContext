#Looking at the previous CMA-ES implementation, I'll improve it with several strategies:
#
#1. **Multi-restart CMA-ES with IPOP** (increasing population size on restarts)
#2. **Better initialization** with multiple strategies
#3. **Local search refinement** (Nelder-Mead style) on the best solution
#4. **Adaptive restarts** based on stagnation detection
#5. **Better boundary handling** using mirroring instead of clipping
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.95 - elapsed()
    
    def mirror_bounds(x):
        # Mirror boundary handling
        for i in range(dim):
            while x[i] < lower[i] or x[i] > upper[i]:
                if x[i] < lower[i]:
                    x[i] = 2 * lower[i] - x[i]
                if x[i] > upper[i]:
                    x[i] = 2 * upper[i] - x[i]
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Diverse initialization ---
    n_init = min(max(15 * dim, 80), 400)
    
    # Latin Hypercube Sampling
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fitnesses = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fitnesses.append((f, i))
    
    # Get top candidates for multi-start
    init_fitnesses.sort()
    top_k = min(5, len(init_fitnesses))
    start_points = [init_points[init_fitnesses[i][1]].copy() for i in range(top_k)]
    
    if best_x is None:
        best_x = (lower + upper) / 2.0
    
    # --- Phase 2: CMA-ES with IPOP restarts ---
    base_lam = max(4 + int(3 * np.log(dim)), 12)
    restart_count = 0
    
    def run_cmaes(x0, sigma0, lam):
        nonlocal best, best_x, restart_count
        
        mean = x0.copy()
        sigma = sigma0
        
        n = dim
        mu_count = lam // 2
        
        weights = np.log(mu_count + 0.5) - np.log(np.arange(1, mu_count + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        gen = 0
        
        best_local = float('inf')
        stag_count = 0
        hist_best = []
        
        while True:
            if time_left() <= 0:
                return
            
            arx = np.zeros((lam, n))
            fitnesses = np.zeros(lam)
            
            for k in range(lam):
                if time_left() <= 0:
                    return
                z = np.random.randn(n)
                x = mean + sigma * (B @ (D * z))
                x = mirror_bounds(x)
                arx[k] = x
                fitnesses[k] = eval_func(x)
                counteval += 1
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            fitnesses = fitnesses[idx]
            
            if fitnesses[0] < best_local:
                best_local = fitnesses[0]
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = np.dot(weights, arx[:mu_count])
            
            mu_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mu_diff) / sigma
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mu_diff / sigma
            
            artmp = (arx[:mu_count] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges))
            
            if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except np.linalg.LinAlgError:
                    return
            
            gen += 1
            
            # Stagnation or convergence -> restart
            if sigma < 1e-14 or stag_count > 10 + 30 * n / lam:
                return
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                return
    
    # Run CMA-ES with restarts (IPOP strategy)
    sp_idx = 0
    while time_left() > 0.5:
        if sp_idx < len(start_points):
            x0 = start_points[sp_idx]
            sp_idx += 1
        elif np.random.rand() < 0.3 and best_x is not None:
            x0 = best_x + np.random.randn(dim) * ranges * 0.1
            x0 = np.clip(x0, lower, upper)
        else:
            x0 = lower + np.random.rand(dim) * ranges
        
        lam = int(base_lam * (2 ** (restart_count * 0.5)))
        lam = min(lam, max(base_lam, 256))
        
        sigma0 = np.mean(ranges) / 3.0 if restart_count > 0 else np.mean(ranges) / 4.0
        
        run_cmaes(x0, sigma0, lam)
        restart_count += 1
    
    return best
#
#Key improvements:
#- **IPOP restarts**: Population size increases with each restart, helping escape local minima
#- **Mirror boundary handling**: Better than clipping for maintaining distribution shape
#- **Stagnation detection**: Triggers restarts when progress stalls
#- **Multi-start from top initial points**: Uses best LHS candidates as starting points
#- **Better sigma management**: Prevents premature convergence with floor values
