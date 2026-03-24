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
    
    # Initial center: random or midpoint
    mu = (lower + upper) / 2.0
    sigma = np.mean(upper - lower) / 4.0
    
    # Evaluate center
    fitness = func(mu)
    if fitness < best:
        best = fitness
        best_params = mu.copy()
    
    # CMA-ES parameters
    mu_eff_size = pop_size // 2
    weights = np.log(mu_eff_size + 0.5) - np.log(np.arange(1, mu_eff_size + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)
    
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C = np.eye(dim)
    eigeneval = 0
    chiN = dim ** 0.5 * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
    
    B = np.eye(dim)
    D = np.ones(dim)
    invsqrtC = np.eye(dim)
    
    generation = 0
    counteval = 1
    
    # Track stagnation for restarts
    stagnation_counter = 0
    last_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        # Update eigen decomposition
        if counteval - eigeneval > pop_size / (c1 + cmu) / dim / 10:
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
        
        # Generate offspring
        arz = np.random.randn(pop_size, dim)
        arx = np.zeros((pop_size, dim))
        fitnesses = np.zeros(pop_size)
        
        for k in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            arx[k] = mu + sigma * (B @ (D * arz[k]))
            # Clip to bounds
            arx[k] = np.clip(arx[k], lower, upper)
            fitnesses[k] = func(arx[k])
            counteval += 1
            
            if fitnesses[k] < best:
                best = fitnesses[k]
                best_params = arx[k].copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        
        # Recombination
        old_mu = mu.copy()
        mu = np.zeros(dim)
        for i in range(mu_eff_size):
            mu += weights[i] * arx[idx[i]]
        
        # Cumulation: update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mu - old_mu) / sigma
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / pop_size)) / chiN < 1.4 + 2.0 / (dim + 1))
        
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mu - old_mu) / sigma
        
        # Adapt covariance matrix
        artmp = np.zeros((mu_eff_size, dim))
        for i in range(mu_eff_size):
            artmp[i] = (arx[idx[i]] - old_mu) / sigma
        
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
        
        # Adapt step size
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.mean(upper - lower))
        
        generation += 1
        
        # Check stagnation
        if generation % 20 == 0:
            if best >= last_best * (1 - 1e-12):
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_best = best
        
        # Restart if stagnated or sigma too small
        if stagnation_counter > 3 or sigma < 1e-16:
            # Restart with random center near best or fully random
            if np.random.rand() < 0.5 and best_params is not None:
                mu = best_params + np.random.randn(dim) * (upper - lower) * 0.1
                mu = np.clip(mu, lower, upper)
            else:
                mu = np.array([np.random.uniform(l, h) for l, h in bounds])
            sigma = np.mean(upper - lower) / 4.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            stagnation_counter = 0
            eigeneval = 0
    
    return best
