#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    # CMA-ES implementation
    # Initialize
    x_mean = np.array([(l + u) / 2.0 for l, u in bounds])
    sigma = np.mean(upper - lower) / 4.0
    
    lam = 4 + int(3 * np.log(dim))
    mu = lam // 2
    
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)
    
    # Adaptation parameters
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    
    # State variables
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    C = np.eye(dim)
    invsqrtC = np.eye(dim)
    eigeneval = 0
    counteval = 0
    chiN = dim ** 0.5 * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
    
    generation = 0
    
    while elapsed() < max_time * 0.95:
        # Generate offspring
        arx = np.zeros((lam, dim))
        arfitness = np.zeros(lam)
        
        for k in range(lam):
            if elapsed() >= max_time * 0.95:
                return best
            z = np.random.randn(dim)
            arx[k] = clip(x_mean + sigma * (B @ (D * z)))
            arfitness[k] = func(arx[k])
            counteval += 1
            if arfitness[k] < best:
                best = arfitness[k]
                best_x = arx[k].copy()
        
        # Sort by fitness
        arindex = np.argsort(arfitness)
        
        # Recombination
        xold = x_mean.copy()
        selected = arx[arindex[:mu]]
        x_mean = np.dot(weights, selected)
        
        # Cumulation: update evolution paths
        ps_update = invsqrtC @ (x_mean - xold) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ps_update
        
        hsig_val = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN)
        hsig = 1.0 if hsig_val < (1.4 + 2.0 / (dim + 1)) else 0.0
        
        pc_update = (x_mean - xold) / sigma
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * pc_update
        
        # Adapt covariance matrix
        artmp = (selected - xold) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T @ np.diag(weights) @ artmp))
        
        # Adapt step size
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        
        # Update B and D from C
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
        
        # Restart if sigma is too small or too large
        if sigma < 1e-20 or sigma > 1e10:
            x_mean = np.array([np.random.uniform(l, u) for l, u in bounds])
            sigma = np.mean(upper - lower) / 4.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            invsqrtC = np.eye(dim)
        
        generation += 1
    
    return best
