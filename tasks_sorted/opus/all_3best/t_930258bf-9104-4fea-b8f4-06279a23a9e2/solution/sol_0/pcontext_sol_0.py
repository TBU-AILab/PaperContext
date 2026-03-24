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
        init_points.append(lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * (upper[i] - lower[i]))
    init_points = np.array(init_points).T  # shape (n_init, dim)
    
    for i in range(n_init):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        x = init_points[i]
        fitness = func(x)
        if fitness < best:
            best = fitness
            best_x = x.copy()
    
    if best_x is None:
        best_x = (lower + upper) / 2.0
    
    # --- Phase 2: CMA-ES inspired search ---
    # Simple (mu/mu_w, lambda)-CMA-ES
    
    sigma = np.mean(upper - lower) / 4.0
    mean = best_x.copy()
    
    # Strategy parameters
    lam = max(4 + int(3 * np.log(dim)), 12)  # population size
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights**2)
    
    # Adaptation parameters
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    
    chiN = dim**0.5 * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
    
    # State variables
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C = np.eye(dim)
    eigeneval = 0
    counteval = 0
    
    # Also maintain a restart mechanism
    stagnation_count = 0
    prev_best = best
    
    generation = 0
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        
        generation += 1
        
        # Eigendecomposition of C
        try:
            if counteval - eigeneval > lam / (c1 + cmu) / dim / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
        except:
            # Reset on numerical issues
            C = np.eye(dim)
            D = np.ones(dim)
            B = np.eye(dim)
            invsqrtC = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
        
        if generation == 1:
            try:
                D_vals, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_vals, 1e-20))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
            except:
                D = np.ones(dim)
                B = np.eye(dim)
                invsqrtC = np.eye(dim)
        
        # Generate offspring
        arx = np.zeros((lam, dim))
        arfitness = np.zeros(lam)
        
        for k in range(lam):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.92:
                return best
            
            z = np.random.randn(dim)
            x = mean + sigma * (B @ (D * z))
            # Clip to bounds
            x = np.clip(x, lower, upper)
            arx[k] = x
            arfitness[k] = func(x)
            counteval += 1
            
            if arfitness[k] < best:
                best = arfitness[k]
                best_x = x.copy()
        
        # Sort by fitness
        arindex = np.argsort(arfitness)
        
        # Recombination
        old_mean = mean.copy()
        mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
        
        # CSA
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
        
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2.0 / (dim + 1))
        
        # CCA
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
        
        # Covariance matrix adaptation
        artmp = (arx[arindex[:mu]] - old_mean) / sigma
        C = (1 - c1 - cmu) * C + \
            c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu * (artmp.T @ np.diag(weights) @ artmp)
        
        # Step-size update
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        
        # Clamp sigma
        sigma = max(sigma, 1e-20 * np.mean(upper - lower))
        sigma = min(sigma, np.mean(upper - lower))
        
        # Check for stagnation and restart if needed
        if generation % 20 == 0:
            if best >= prev_best - 1e-12 * abs(prev_best + 1e-30):
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best = best
            
            if stagnation_count >= 3 or sigma < 1e-16:
                # Restart with smaller region around best
                sigma = np.mean(upper - lower) / 4.0
                mean = best_x.copy() + np.random.randn(dim) * sigma * 0.1
                mean = np.clip(mean, lower, upper)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                invsqrtC = np.eye(dim)
                eigeneval = 0
                stagnation_count = 0
    
    return best
