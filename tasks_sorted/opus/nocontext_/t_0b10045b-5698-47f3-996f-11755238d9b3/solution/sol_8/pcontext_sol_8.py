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
    pop_size = 4 + int(3 * np.log(dim))
    pop_size = max(pop_size, 10)
    
    # Start with multiple restarts of Nelder-Mead + CMA-ES hybrid
    # First, do a Latin Hypercube-like initial sampling
    n_initial = min(200, max(50, dim * 10))
    
    for i in range(n_initial):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.05):
            break
        params = np.array([np.random.uniform(low, high) for low, high in bounds])
        fitness = func(params)
        if fitness < best:
            best = fitness
            best_params = params.copy()
    
    if best_params is None:
        best_params = (lower + upper) / 2.0
    
    # --- Phase 2: CMA-ES ---
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    def run_cmaes(x0, sigma0, budget_fraction):
        nonlocal best, best_params
        
        time_limit = timedelta(seconds=max_time * budget_fraction)
        phase_start = datetime.now()
        
        n = len(x0)
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        sigma = sigma0
        mean = x0.copy()
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        generation = 0
        counteval = 0
        
        while True:
            if (datetime.now() - phase_start) >= time_limit:
                break
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
                break
            
            # Generate offspring
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                z = np.random.randn(n)
                arx[k] = clip_to_bounds(mean + sigma * (B @ (D * z)))
                arfitness[k] = func(arx[k])
                counteval += 1
                
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_params = arx[k].copy()
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = np.sum(weights[:, None] * selected, axis=0)
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance matrix adaptation
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            # Sigma adaptation
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            # Update B and D from C
            if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            generation += 1
            
            # Check for stagnation
            if sigma < 1e-12:
                break
    
    # Run CMA-ES with restarts
    sigma0 = np.mean(upper - lower) / 4.0
    
    # First run with best known point
    run_cmaes(best_params.copy(), sigma0, 0.4)
    
    # Restart with larger sigma from best known
    if (datetime.now() - start) < timedelta(seconds=max_time * 0.7):
        run_cmaes(best_params.copy(), sigma0 * 0.5, 0.25)
    
    # Another restart from random point
    if (datetime.now() - start) < timedelta(seconds=max_time * 0.85):
        random_start = np.array([np.random.uniform(low, high) for low, high in bounds])
        run_cmaes(random_start, sigma0, 0.15)
    
    # --- Phase 3: Local refinement with Nelder-Mead ---
    def nelder_mead(x0, step=None):
        nonlocal best, best_params
        
        n = len(x0)
        if step is None:
            step = np.maximum(np.abs(x0) * 0.05, (upper - lower) * 0.01)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            point = x0.copy()
            point[i] += step[i]
            simplex[i + 1] = clip_to_bounds(point)
        
        f_values = np.array([func(clip_to_bounds(s)) for s in simplex])
        
        for i in range(n + 1):
            if f_values[i] < best:
                best = f_values[i]
                best_params = simplex[i].copy()
        
        max_iter = 1000 * n
        for iteration in range(max_iter):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                break
            
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip_to_bounds(centroid + alpha * (centroid - simplex[-1]))
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if f_values[0] <= fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                # Expansion
                xe = clip_to_bounds(centroid + gamma * (xr - centroid))
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                # Contraction
                xc = clip_to_bounds(centroid + rho * (simplex[-1] - centroid))
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < f_values[-1]:
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_values[i] = func(simplex[i])
                        if f_values[i] < best:
                            best = f_values[i]
                            best_params = simplex[i].copy()
            
            # Convergence check
            if np.std(f_values) < 1e-15:
                break
    
    if (datetime.now() - start) < timedelta(seconds=max_time * 0.95):
        nelder_mead(best_params.copy())
    
    return best
