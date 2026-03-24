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
    
    # Start with multiple restarts strategy
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    # Evaluate a point
    def evaluate(x):
        nonlocal best, best_params
        x_clipped = clip_to_bounds(x)
        f = func(x_clipped)
        if f < best:
            best = f
            best_params = x_clipped.copy()
        return f
    
    # --- Initial Latin Hypercube Sampling ---
    n_init = min(max(10 * dim, 50), 200)
    for i in range(n_init):
        if time_left() < 0.1:
            return best
        x = np.array([np.random.uniform(l, u) for l, u in bounds])
        evaluate(x)
    
    # --- CMA-ES implementation ---
    def run_cmaes(x0, sigma0, budget_fraction=1.0):
        nonlocal best, best_params
        
        n = dim
        lam = pop_size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        # State variables
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = (n <= 100)
        
        if use_full_cov:
            C = np.eye(n)
            eigeneval = 0
            B = np.eye(n)
            D = np.ones(n)
        else:
            diagC = np.ones(n)
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        generation = 0
        
        while True:
            if time_left() < 0.2:
                return
            
            # Sample population
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.1:
                    return
                
                if use_full_cov:
                    z = np.random.randn(n)
                    arx[k] = mean + sigma * (B @ (D * z))
                else:
                    arx[k] = mean + sigma * np.sqrt(diagC) * np.random.randn(n)
                
                arfitness[k] = evaluate(arx[k])
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = np.dot(weights, selected)
            
            # CSA
            if use_full_cov:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ((mean - old_mean) / sigma) / np.sqrt(diagC)
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(n + 1))
            
            # CCA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance matrix adaptation
            artmp = (selected - old_mean) / sigma
            
            if use_full_cov:
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (artmp.T @ np.diag(weights) @ artmp)
                
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                
                eigeneval += lam
                if eigeneval >= lam / (c1 + cmu_val) / n / 10:
                    eigeneval = 0
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                    except:
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
            else:
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            
            # Sigma adaptation
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            generation += 1
            
            # Check termination conditions
            if sigma < 1e-12:
                return
            if generation > 1000 + 500 * n:
                return
    
    # --- Multi-restart CMA-ES ---
    restart_count = 0
    sigma_multiplier = 1.0
    
    while time_left() > 0.5:
        if restart_count == 0 and best_params is not None:
            x0 = best_params.copy()
            sigma0 = 0.3 * np.mean(upper - lower)
        elif restart_count < 3 and best_params is not None:
            # Local restart near best
            perturbation = np.random.randn(dim) * 0.1 * (upper - lower)
            x0 = clip_to_bounds(best_params + perturbation)
            sigma0 = 0.2 * np.mean(upper - lower) * sigma_multiplier
        else:
            # Random restart
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            sigma0 = 0.3 * np.mean(upper - lower)
        
        run_cmaes(x0, sigma0)
        restart_count += 1
        sigma_multiplier *= 0.8
        if sigma_multiplier < 0.05:
            sigma_multiplier = 1.0
    
    # --- Final local refinement with Nelder-Mead ---
    if best_params is not None and time_left() > 0.3:
        # Simple Nelder-Mead
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step = 0.01 * (upper - lower)
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i]
            simplex[i + 1] = clip_to_bounds(simplex[i + 1])
        
        f_simplex = np.array([evaluate(simplex[i]) for i in range(n + 1)])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while time_left() > 0.1:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip_to_bounds(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                # Expand
                xe = clip_to_bounds(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contract
                if fr < f_simplex[-1]:
                    xc = clip_to_bounds(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                else:
                    xc = clip_to_bounds(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip_to_bounds(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
    
    return best
