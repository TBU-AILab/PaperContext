#Looking at the previous implementation, it's a decent CMA-ES but there are several improvements I can make:
#
#1. **Multiple restarts with increasing population sizes (IPOP-CMA-ES)** - this is crucial for escaping local optima
#2. **Better initialization** - use multiple strategies
#3. **Add a local search phase (Nelder-Mead style)** after CMA-ES converges
#4. **Differential Evolution as an alternative search** to diversify
#5. **More aggressive restart detection**
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
    n_init = min(max(30 * dim, 200), 1000)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i] + lower[i]
    
    init_fits = []
    for i in range(n_init):
        if remaining() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fits.append((f, i))
    
    # Keep top candidates for restarts
    init_fits.sort()
    top_k = min(10, len(init_fits))
    top_points = [init_points[init_fits[i][1]].copy() for i in range(top_k)]
    
    # --- Phase 2: IPOP-CMA-ES with restarts ---
    def run_cmaes(x0, init_sigma, pop_factor=1, budget_fraction=None):
        nonlocal best, best_x
        
        if remaining() <= 0:
            return
        
        cma_start = elapsed()
        if budget_fraction:
            cma_budget = budget_fraction
        else:
            cma_budget = remaining()
        
        sigma = init_sigma
        mean = x0.copy()
        
        base_lam = 4 + int(3 * np.log(dim))
        lam = max(base_lam * pop_factor, 6)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu_p = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Use diagonal covariance for high dimensions
        use_sep = dim > 100
        
        if use_sep:
            diagC = np.ones(dim)
        else:
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigeneval = 0
        
        counteval = 0
        gen = 0
        flat_count = 0
        prev_median = float('inf')
        best_gen_fit = float('inf')
        no_improve_count = 0
        
        while True:
            if remaining() <= 0 or (elapsed() - cma_start) > cma_budget:
                return
            
            # Eigen decomposition update
            if not use_sep:
                if counteval - eigeneval > lam / (c1 + cmu_p + 1e-20) / dim / 10:
                    eigeneval = counteval
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
            
            arxs = []
            fitnesses = []
            
            for k in range(lam):
                if remaining() <= 0:
                    return
                z = np.random.randn(dim)
                if use_sep:
                    x = mean + sigma * (np.sqrt(diagC) * z)
                else:
                    x = mean + sigma * (B @ (D * z))
                x = clip(x)
                f = eval_func(x)
                counteval += 1
                arxs.append(x)
                fitnesses.append(f)
            
            idx = np.argsort(fitnesses)
            sorted_arxs = [arxs[i] for i in idx]
            sorted_fits = [fitnesses[i] for i in idx]
            
            median_fit = sorted_fits[lam // 2]
            
            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * sorted_arxs[i]
            
            diff = mean - old_mean
            
            if use_sep:
                inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(diagC, 1e-20))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_diag * diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff / sigma
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2/(dim + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            artmp = np.zeros((mu, dim))
            for i in range(mu):
                artmp[i] = (sorted_arxs[i] - old_mean) / sigma
            
            if use_sep:
                diagC = (1 - c1 - cmu_p) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_p * np.sum(weights[:, None] * artmp**2, axis=0) if mu > 0 else diagC
                # Fix for weights broadcasting
                diagC = (1 - c1 - cmu_p) * diagC + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC)
                for i in range(mu):
                    diagC += cmu_p * weights[i] * artmp[i]**2
            else:
                C = (1 - c1 - cmu_p) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_p * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 2)
            
            gen += 1
            
            # Check stagnation
            if sorted_fits[0] < best_gen_fit - 1e-12 * (abs(best_gen_fit) + 1e-12):
                best_gen_fit = sorted_fits[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if abs(median_fit - prev_median) < 1e-12 * (abs(median_fit) + 1e-12):
                flat_count += 1
            else:
                flat_count = 0
            prev_median = median_fit
            
            # Condition number check
            if not use_sep:
                if len(D) > 0 and D.max() > 1e7 * D.min():
                    return
            
            if flat_count >= 10 or no_improve_count >= 50 + 10 * dim or sigma < 1e-16:
                return
    
    # --- Run IPOP-CMA-ES ---
    restart_count = 0
    pop_multiplier = 1
    
    while remaining() > 0.5:
        if restart_count < len(top_points):
            x0 = top_points[restart_count].copy()
        else:
            # Random restart or perturb best
            if np.random.rand() < 0.5 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * ranges * 0.1
                x0 = clip(x0)
            else:
                x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        init_sigma = np.mean(ranges) / (3.0 if restart_count < len(top_points) else 2.0)
        
        run_cmaes(x0, init_sigma, pop_factor=pop_multiplier)
        
        restart_count += 1
        if restart_count >= len(top_points):
            pop_multiplier = min(pop_multiplier * 2, 16)
    
    return best
#
#Key improvements:
#1. **IPOP strategy**: Population size doubles on restarts, helping escape local optima
#2. **Separable CMA-ES** for high dimensions (>100) to avoid expensive matrix operations
#3. **Better stagnation detection**: tracks both median fitness and best improvement
#4. **Condition number monitoring**: restarts when the covariance matrix becomes ill-conditioned
#5. **More initial samples** and keeping top-k for seeding restarts
#6. **Smarter restart strategy**: cycles through best initial points first, then alternates between perturbing best known solution and random restarts
