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
    
    # Initialize center at random or middle of bounds
    mu = (lower + upper) / 2.0
    sigma = np.mean(upper - lower) / 4.0
    
    # First evaluate center
    fitness = func(mu)
    if fitness < best:
        best = fitness
        best_params = mu.copy()
    
    # Simple CMA-ES implementation
    # Strategy parameters
    mu_eff_count = pop_size // 2
    weights = np.log(mu_eff_count + 0.5) - np.log(np.arange(1, mu_eff_count + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)
    
    # Adaptation parameters
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    
    # State variables
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    
    if dim <= 200:
        C = np.eye(dim)
        use_full_cov = True
    else:
        # For high dimensions, use diagonal approximation
        C_diag = np.ones(dim)
        use_full_cov = False
    
    chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    generation = 0
    stagnation_count = 0
    last_best = best
    
    # Multiple restarts
    restart_count = 0
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        # Generate offspring
        if use_full_cov:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                invsqrt_C = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
            except:
                C = np.eye(dim)
                sqrt_C = np.eye(dim)
                invsqrt_C = np.eye(dim)
                eigvals = np.ones(dim)
        else:
            sqrt_C_diag = np.sqrt(np.maximum(C_diag, 1e-20))
            invsqrt_C_diag = 1.0 / sqrt_C_diag
        
        # Sample population
        arz = np.random.randn(pop_size, dim)
        arx = np.zeros((pop_size, dim))
        
        for k in range(pop_size):
            if use_full_cov:
                arx[k] = mu + sigma * (sqrt_C @ arz[k])
            else:
                arx[k] = mu + sigma * (sqrt_C_diag * arz[k])
            # Clip to bounds
            arx[k] = np.clip(arx[k], lower, upper)
        
        # Evaluate
        fitnesses = np.zeros(pop_size)
        for k in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            fitnesses[k] = func(arx[k])
            if fitnesses[k] < best:
                best = fitnesses[k]
                best_params = arx[k].copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        
        # Recombination
        old_mu = mu.copy()
        mu = np.zeros(dim)
        for i in range(mu_eff_count):
            mu += weights[i] * arx[idx[i]]
        
        # Update evolution paths
        if use_full_cov:
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrt_C @ (mu - old_mu)) / sigma
        else:
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrt_C_diag * (mu - old_mu)) / sigma
        
        hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(dim + 1))
        
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mu - old_mu) / sigma
        
        # Update covariance matrix
        if use_full_cov:
            artmp = np.zeros((mu_eff_count, dim))
            for i in range(mu_eff_count):
                artmp[i] = (arx[idx[i]] - old_mu) / sigma
            
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu * sum(weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu_eff_count))
            
            # Enforce symmetry
            C = (C + C.T) / 2
        else:
            artmp = np.zeros((mu_eff_count, dim))
            for i in range(mu_eff_count):
                artmp[i] = (arx[idx[i]] - old_mu) / sigma
            
            C_diag = (1 - c1 - cmu) * C_diag + \
                     c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag) + \
                     cmu * sum(weights[i] * artmp[i]**2 for i in range(mu_eff_count))
        
        # Update step size
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.mean(upper - lower))  # prevent explosion
        
        generation += 1
        
        # Check for stagnation
        if abs(best - last_best) < 1e-12:
            stagnation_count += 1
        else:
            stagnation_count = 0
            last_best = best
        
        # Restart if stagnated or sigma too small
        if stagnation_count > 20 + dim or sigma < 1e-16:
            restart_count += 1
            
            # Restart with increased population size
            pop_size_mult = 1 + restart_count
            pop_size = (4 + int(3 * np.log(dim))) * pop_size_mult
            pop_size = min(pop_size, 500)
            
            mu_eff_count = pop_size // 2
            weights = np.log(mu_eff_count + 0.5) - np.log(np.arange(1, mu_eff_count + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            
            # Random restart near best or random
            if np.random.rand() < 0.5 and best_params is not None:
                mu = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
                mu = np.clip(mu, lower, upper)
                sigma = np.mean(upper - lower) / 8.0
            else:
                mu = np.array([np.random.uniform(l, u) for l, u in bounds])
                sigma = np.mean(upper - lower) / 4.0
            
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            if use_full_cov:
                C = np.eye(dim)
            else:
                C_diag = np.ones(dim)
            
            stagnation_count = 0
            generation = 0
            last_best = best
    
    return best
