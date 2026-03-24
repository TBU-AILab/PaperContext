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
    pop_size = min(4 + int(3 * np.log(dim)), 100)
    if pop_size % 2 == 1:
        pop_size += 1
    
    # Initial center: random or center of bounds
    mean = (lower + upper) / 2.0
    sigma = np.mean((upper - lower)) / 4.0
    
    # CMA-ES parameters
    mu = pop_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C = np.eye(dim)
    eigeneval = 0
    chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    count_eval = 0
    
    # Also do some random initial evaluations
    for _ in range(min(pop_size, 20)):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        params = np.random.uniform(lower, upper)
        fitness = func(params)
        count_eval += 1
        if fitness < best:
            best = fitness
            best_params = params.copy()
            mean = params.copy()
    
    generation = 0
    stagnation_count = 0
    last_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.92):
            break
        
        # Eigendecomposition
        try:
            C = (C + C.T) / 2
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
        except:
            C = np.eye(dim)
            D = np.ones(dim)
            B = np.eye(dim)
        
        # Generate population
        pop = np.zeros((pop_size, dim))
        fitnesses = np.zeros(pop_size)
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.92):
                return best
            
            z = np.random.randn(dim)
            pop[i] = mean + sigma * (B @ (D * z))
            # Clip to bounds
            pop[i] = np.clip(pop[i], lower, upper)
            fitnesses[i] = func(pop[i])
            count_eval += 1
            
            if fitnesses[i] < best:
                best = fitnesses[i]
                best_params = pop[i].copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        pop = pop[idx]
        fitnesses = fitnesses[idx]
        
        # Update mean
        old_mean = mean.copy()
        mean = np.sum(weights[:, None] * pop[:mu], axis=0)
        mean = np.clip(mean, lower, upper)
        
        # Update evolution paths
        diff = mean - old_mean
        invsqrtC = B @ np.diag(1.0 / D) @ B.T
        
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ diff) / sigma
        
        hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2 / (dim + 1))
        
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * diff / sigma
        
        # Update covariance matrix
        artmp = (pop[:mu] - old_mean) / sigma
        C = (1 - c1 - cmu) * C + \
            c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + \
            cmu * (artmp.T @ np.diag(weights) @ artmp)
        
        # Update sigma
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.mean(upper - lower))
        sigma = max(sigma, 1e-20)
        
        generation += 1
        
        # Check stagnation and restart if needed
        if best >= last_best:
            stagnation_count += 1
        else:
            stagnation_count = 0
            last_best = best
        
        if stagnation_count > 10 + int(30 * dim / pop_size) or sigma < 1e-16:
            # Restart with smaller region around best
            if best_params is not None and np.random.rand() < 0.5:
                mean = best_params.copy() + 0.1 * (upper - lower) * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
            else:
                mean = np.random.uniform(lower, upper)
            sigma = np.mean((upper - lower)) / 4.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            stagnation_count = 0
    
    # --- Phase 2: Local refinement with Nelder-Mead style ---
    if best_params is not None:
        # Quick local search around best
        step = (upper - lower) * 0.001
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.99):
                break
            
            candidate = best_params + step * np.random.randn(dim)
            candidate = np.clip(candidate, lower, upper)
            fitness = func(candidate)
            if fitness < best:
                best = fitness
                best_params = candidate.copy()
    
    return best
