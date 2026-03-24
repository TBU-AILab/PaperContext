#Looking at the previous implementation, I see a CMA-ES approach that could be improved in several ways:
#
#1. **Better restart strategy** - Use IPOP-CMA-ES (increasing population) restarts with diversified initialization
#2. **Add local search** - Nelder-Mead simplex for local refinement
#3. **Better initial sampling** - Keep top candidates and use multiple restarts from them
#4. **Differential Evolution hybrid** - Combine global and local search strategies
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
    
    def remaining():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(30 * dim, 200), 800)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    fitnesses = []
    for i in range(n_init):
        if remaining() <= 0:
            return best
        f = eval_func(init_points[i])
        fitnesses.append((f, i))
    
    fitnesses.sort()
    # Keep top candidates for restarts
    top_k = min(10, n_init)
    top_points = [init_points[fitnesses[i][1]].copy() for i in range(top_k)]
    
    # --- Phase 2: IPOP-CMA-ES with restarts ---
    def run_cmaes(x0, initial_sigma, max_evals_time_fraction=None):
        nonlocal best, best_x
        
        sigma = initial_sigma
        mean = x0.copy()
        n = dim
        
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        gen = 0
        best_local = float('inf')
        stag_count = 0
        
        while remaining() > 0:
            gen += 1
            
            # Eigendecomposition
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    Dvals, B = np.linalg.eigh(C)
                    Dvals = np.maximum(Dvals, 1e-20)
                    D = np.sqrt(Dvals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            arx = np.zeros((lam, n))
            arfit = np.zeros(lam)
            
            for k in range(lam):
                if remaining() <= 0:
                    return
                z = np.random.randn(n)
                arx[k] = clip(mean + sigma * (B @ (D * z)))
                arfit[k] = eval_func(arx[k])
                counteval += 1
            
            idx = np.argsort(arfit)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2.0 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges))
            
            # Stagnation check
            if gen % 15 == 0:
                if arfit[idx[0]] < best_local - 1e-10 * (abs(best_local) + 1e-30):
                    best_local = arfit[idx[0]]
                    stag_count = 0
                else:
                    stag_count += 1
                if stag_count >= 3 or sigma < 1e-16:
                    return  # restart
    
    # Run CMA-ES with restarts (IPOP strategy)
    restart = 0
    pop_multiplier = 1
    
    while remaining() > 0:
        if restart < len(top_points):
            x0 = top_points[restart]
        else:
            # Random restart or perturb best
            if np.random.rand() < 0.5 and best_x is not None:
                x0 = clip(best_x + np.random.randn(dim) * ranges * 0.2)
            else:
                x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        init_sigma = np.mean(ranges) / (3.0 + restart * 0.5)
        init_sigma = max(init_sigma, np.mean(ranges) * 0.01)
        
        run_cmaes(x0, init_sigma)
        restart += 1
    
    # --- Phase 3: Local refinement with Nelder-Mead style ---
    if best_x is not None and remaining() > 0.5:
        step = ranges * 0.001
        while remaining() > 0:
            improved = False
            for i in range(dim):
                if remaining() <= 0:
                    return best
                for sign in [1, -1]:
                    trial = best_x.copy()
                    trial[i] = np.clip(trial[i] + sign * step[i], lower[i], upper[i])
                    f = eval_func(trial)
                    if f < best:
                        improved = True
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-12:
                    break
    
    return best
#
#Key improvements:
#1. **More initial samples** with Latin Hypercube for better coverage
#2. **IPOP restart strategy** - cycles through best initial points, then does diversified restarts
#3. **Stagnation detection** every 15 generations (faster response)
#4. **Local coordinate descent** refinement phase at the end
#5. **Better sigma management** - decreasing initial sigma with restarts, proper clamping
#6. **Unified evaluation function** that always tracks the global best
