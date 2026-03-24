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
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
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
    
    mu = best_x.copy()
    sigma = np.mean(upper - lower) / 4.0
    
    # Covariance matrix as identity initially
    C = np.eye(dim)
    pc = np.zeros(dim)  # evolution path for C
    ps = np.zeros(dim)  # evolution path for sigma
    
    lam = max(4 + int(3 * np.log(dim)), 12)  # population size
    mu_eff_count = lam // 2
    
    # Weights
    weights = np.log(mu_eff_count + 0.5) - np.log(np.arange(1, mu_eff_count + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)
    
    # Learning rates
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    generation = 0
    eigeneval = 0
    counteval = n_init
    
    B = np.eye(dim)
    D = np.ones(dim)
    invsqrtC = np.eye(dim)
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        
        # Generate lambda offspring
        arz = np.random.randn(lam, dim)
        arx = np.zeros((lam, dim))
        fitnesses = np.zeros(lam)
        
        for k in range(lam):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.92:
                return best
            
            z = arz[k]
            x = mu + sigma * (B @ (D * z))
            # Clip to bounds
            x = np.clip(x, lower, upper)
            arx[k] = x
            fitnesses[k] = func(x)
            counteval += 1
            
            if fitnesses[k] < best:
                best = fitnesses[k]
                best_x = x.copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        arx = arx[idx]
        arz = arz[idx]
        
        # Recombination: new mean
        old_mu = mu.copy()
        mu = np.sum(weights[:, None] * arx[:mu_eff_count], axis=0)
        
        # Update evolution paths
        mu_diff = mu - old_mu
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mu_diff) / sigma
        
        hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chiN < 1.4 + 2 / (dim + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mu_diff / sigma
        
        # Update covariance matrix
        artmp = (arx[:mu_eff_count] - old_mu) / sigma
        C = (1 - c1 - cmu) * C + \
            c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu * (artmp.T @ np.diag(weights) @ artmp)
        
        # Update sigma
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.mean(upper - lower))  # cap sigma
        
        # Decompose C
        if counteval - eigeneval > lam / (c1 + cmu) / dim / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.maximum(D_sq, 1e-20)
                D = np.sqrt(D_sq)
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                invsqrtC = np.eye(dim)
        
        generation += 1
        
        # Restart if sigma is too small
        if sigma < 1e-12:
            sigma = np.mean(upper - lower) / 4.0
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            # Restart from best + noise or random
            if np.random.rand() < 0.5:
                mu = best_x.copy() + np.random.randn(dim) * (upper - lower) * 0.01
                mu = np.clip(mu, lower, upper)
            else:
                mu = lower + np.random.rand(dim) * (upper - lower)
    
    return best
