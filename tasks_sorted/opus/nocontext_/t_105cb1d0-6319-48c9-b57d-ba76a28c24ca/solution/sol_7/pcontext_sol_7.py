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
    if pop_size % 2 == 1:
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
    n_init = min(pop_size * 3, max(10 * dim, 50))
    
    # Initial random sampling
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = lower + np.random.rand(dim) * (upper - lower)
        eval_func(x)
    
    # CMA-ES implementation
    def run_cmaes(mean, sigma, budget_fraction):
        nonlocal best, best_params
        
        target_time = start + timedelta(seconds=max_time * budget_fraction)
        
        N = dim
        lam = pop_size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cc = (4 + mueff/N) / (N + 4 + 2*mueff/N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
        
        # State variables
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        if N <= 200:
            C = np.eye(N)
            use_full_cov = True
        else:
            # Use diagonal only for high dimensions
            diagC = np.ones(N)
            use_full_cov = False
        
        chiN = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))
        
        sig = sigma
        m = mean.copy()
        
        gen = 0
        eigeneval = 0
        
        while elapsed() < (target_time - start).total_seconds() + start.timestamp() - start.timestamp():
            if datetime.now() >= target_time:
                break
            if elapsed() >= max_time * 0.95:
                break
            
            gen += 1
            
            # Generate candidates
            if use_full_cov:
                if gen % max(1, int(lam / (c1 + cmu) / N / 10)) == 0 or gen == 1:
                    try:
                        C = np.triu(C) + np.triu(C, 1).T
                        D_vals, B = np.linalg.eigh(C)
                        D_vals = np.maximum(D_vals, 1e-20)
                        D = np.sqrt(D_vals)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        C = np.eye(N)
                        D = np.ones(N)
                        B = np.eye(N)
                        invsqrtC = np.eye(N)
                
                arz = np.random.randn(lam, N)
                arx = np.array([m + sig * (B @ (D * z)) for z in arz])
            else:
                sqrtDiagC = np.sqrt(diagC)
                arz = np.random.randn(lam, N)
                arx = np.array([m + sig * sqrtDiagC * z for z in arz])
            
            # Evaluate
            fitnesses = np.array([eval_func(x) for x in arx])
            
            if elapsed() >= max_time * 0.95:
                break
            
            # Sort
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            # Update mean
            old_m = m.copy()
            m = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            # Update evolution paths
            if use_full_cov:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (m - old_m) / sig
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (m - old_m) / (sig * sqrtDiagC)
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(N+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m - old_m) / sig
            
            # Update covariance
            if use_full_cov:
                artmp = (arx[:mu] - old_m) / sig
                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * np.sum([weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu)], axis=0)
            else:
                artmp = (arx[:mu] - old_m) / sig
                diagC = (1 - c1 - cmu) * diagC + \
                         c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                         cmu * np.sum(weights[:, None] * artmp**2, axis=0)
            
            # Update sigma
            sig = sig * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # Prevent sigma from getting too small or too large
            sig = min(sig, np.max(upper - lower))
            sig = max(sig, 1e-20)
            
            # Check for convergence
            if sig < 1e-16:
                break
    
    # Run CMA-ES with restarts
    n_restarts = 0
    total_budget = 0.90  # reserve 10% for final local search
    restart_budgets = []
    
    # First run from best found point
    if best_params is not None:
        init_mean = best_params.copy()
    else:
        init_mean = (lower + upper) / 2
    
    init_sigma = np.mean(upper - lower) / 4
    
    # Allocate time for restarts with increasing population (IPOP-CMA-ES)
    current_pop = pop_size
    restart_count = 0
    
    while elapsed() < max_time * 0.85:
        remaining = max_time * 0.85 - elapsed()
        if remaining < 0.5:
            break
        
        if restart_count == 0:
            mean = init_mean.copy()
            sigma = init_sigma
        else:
            # Random restart or restart near best
            if np.random.rand() < 0.5 and best_params is not None:
                mean = best_params + np.random.randn(dim) * (upper - lower) * 0.1
                mean = clip(mean)
                sigma = init_sigma * 0.5
            else:
                mean = lower + np.random.rand(dim) * (upper - lower)
                sigma = init_sigma
        
        budget_end = min(0.85, elapsed() / max_time + max(0.1, remaining * 0.4 / max_time))
        run_cmaes(mean, sigma, budget_end)
        restart_count += 1
    
    # --- Phase 2: Nelder-Mead local search around best ---
    if best_params is not None and elapsed() < max_time * 0.95:
        # Simple Nelder-Mead
        n_nm = dim + 1
        scale = np.mean(upper - lower) * 0.01
        
        # Initialize simplex around best
        simplex = np.zeros((n_nm, dim))
        simplex[0] = best_params.copy()
        for i in range(1, n_nm):
            simplex[i] = best_params.copy()
            simplex[i][i-1 if i-1 < dim else 0] += scale * (1 if np.random.rand() > 0.5 else -1)
            simplex[i] = clip(simplex[i])
        
        f_simplex = np.array([eval_func(x) for x in simplex])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while elapsed() < max_time * 0.98:
            idx = np.argsort(f_simplex)
            simplex = simplex[idx]
            f_simplex = f_simplex[idx]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_func(xr)
            
            if elapsed() >= max_time * 0.98:
                break
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
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
                # Contraction
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n_nm):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                            if elapsed() >= max_time * 0.98:
                                break
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = eval_func(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n_nm):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                            if elapsed() >= max_time * 0.98:
                                break
    
    return best
