#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    
    init_points = []
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points.append((perm + np.random.uniform(0, 1, n_init)) / n_init * (upper[i] - lower[i]) + lower[i])
    init_points = np.array(init_points).T  # (n_init, dim)
    
    population = []
    for i in range(n_init):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        x = init_points[i]
        x = np.clip(x, lower, upper)
        f = func(x)
        population.append((f, x.copy()))
        if f < best:
            best = f
            best_x = x.copy()
    
    # Sort population
    population.sort(key=lambda t: t[0])
    
    # --- Phase 2: CMA-ES inspired search from best points ---
    # We'll implement a simplified CMA-ES
    def cmaes_search(x0, sigma0, budget_fraction):
        nonlocal best, best_x
        
        n = dim
        lam = 4 + int(3 * np.log(n))  # population size
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
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        if n > 50:
            diagC = np.ones(n)
            use_full = False
        else:
            C = np.eye(n)
            use_full = True
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen = 0
        time_limit = max_time * budget_fraction
        
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.95:
                return
            
            remaining_for_this = time_limit - (elapsed - (max_time * (1 - budget_fraction) if budget_fraction < 1 else 0))
            
            # Generate offspring
            if use_full:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    sqrtC = eigvecs @ np.diag(np.sqrt(eigvals))
                    invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                except:
                    sqrtC = np.eye(n)
                    invsqrtC = np.eye(n)
                    C = np.eye(n)
            
            arxs = []
            arfitness = []
            for k in range(lam):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    return
                
                z = np.random.randn(n)
                if use_full:
                    x = xmean + sigma * (sqrtC @ z)
                else:
                    x = xmean + sigma * (np.sqrt(diagC) * z)
                
                x = np.clip(x, lower, upper)
                f = func(x)
                arxs.append(x)
                arfitness.append(f)
                
                if f < best:
                    best = f
                    best_x = x.copy()
            
            # Sort by fitness
            idx = np.argsort(arfitness)
            
            # Recombination
            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arxs[idx[i]]
            
            # Update evolution paths
            diff = xmean - xold
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (diff / (np.sqrt(diagC) * sigma + 1e-30))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            # Update covariance
            if use_full:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (arxs[idx[i]] - xold) / sigma
                
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * sum(weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
                
                # Ensure symmetry
                C = np.triu(C) + np.triu(C, 1).T
            else:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (arxs[idx[i]] - xold) / sigma
                
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * sum(weights[i] * artmp[i]**2 for i in range(mu))
                diagC = np.maximum(diagC, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(upper - lower))
            
            gen += 1
            
            # Restart condition
            if sigma < 1e-12:
                return
            if gen > 1000:
                return
    
    # --- Phase 3: Run CMA-ES from multiple restarts ---
    # Use top candidates as starting points
    n_restarts = max(3, min(10, dim))
    
    for r in range(n_restarts):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.85:
            break
        
        remaining = max_time - elapsed
        budget_frac = remaining / max_time
        
        if r == 0:
            x0 = best_x.copy()
        elif r < len(population):
            x0 = population[r][1].copy()
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        
        sigma0 = 0.3 * np.mean(upper - lower)
        if r > 0:
            sigma0 *= (1 + r * 0.2)  # Increase sigma for diversity
        
        cmaes_search(x0, sigma0, budget_frac / n_restarts)
    
    # --- Phase 4: Local refinement with Nelder-Mead style ---
    elapsed = (datetime.now() - start).total_seconds()
    if elapsed < max_time * 0.95 and best_x is not None:
        # Simple coordinate descent refinement
        step = 0.001 * (upper - lower)
        improved = True
        while improved:
            improved = False
            for i in range(dim):
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= max_time * 0.95:
                    return best
                
                for direction in [1, -1]:
                    x_trial = best_x.copy()
                    x_trial[i] = np.clip(x_trial[i] + direction * step[i], lower[i], upper[i])
                    f = func(x_trial)
                    if f < best:
                        best = f
                        best_x = x_trial.copy()
                        improved = True
        
        # Try smaller steps
        for scale in [0.0001, 0.00001]:
            step = scale * (upper - lower)
            improved = True
            while improved:
                improved = False
                for i in range(dim):
                    elapsed = (datetime.now() - start).total_seconds()
                    if elapsed >= max_time * 0.95:
                        return best
                    for direction in [1, -1]:
                        x_trial = best_x.copy()
                        x_trial[i] = np.clip(x_trial[i] + direction * step[i], lower[i], upper[i])
                        f = func(x_trial)
                        if f < best:
                            best = f
                            best_x = x_trial.copy()
                            improved = True
    
    return best
