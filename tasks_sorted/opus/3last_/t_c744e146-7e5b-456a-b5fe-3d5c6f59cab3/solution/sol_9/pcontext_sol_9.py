#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    
    # CMA-ES inspired optimization
    
    # Initialize
    pop_size = 4 + int(3 * np.log(dim))
    if pop_size < 8:
        pop_size = 8
    mu = pop_size // 2
    
    # Weights
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)
    
    # Adaptation parameters
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    # Multiple restarts
    best_x = None
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        # Initialize mean randomly
        mean = np.array([np.random.uniform(l, u) for l, u in bounds])
        sigma = np.mean(upper - lower) / 4.0
        
        # CMA-ES state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        eigeneval = 0
        counteval = 0
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        
        generation = 0
        
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Generate population
            arx = np.zeros((pop_size, dim))
            arfitness = np.zeros(pop_size)
            
            for k in range(pop_size):
                z = np.random.randn(dim)
                x = mean + sigma * (B @ (D * z))
                # Clip to bounds
                x = np.clip(x, lower, upper)
                arx[k] = x
                arfitness[k] = func(x)
                if arfitness[k] < best:
                    best = arfitness[k]
                    best_x = x.copy()
                counteval += 1
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = np.dot(weights, selected)
            
            # Cumulation: update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / pop_size)) / chiN < 1.4 + 2.0 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Adapt covariance matrix
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            # Adapt step size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.mean(upper - lower))
            
            # Update B and D from C
            if counteval - eigeneval > pop_size / (c1 + cmu_val + 1e-20) / dim / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)
                    invsqrtC = np.eye(dim)
            
            generation += 1
            
            # Restart condition
            if sigma * np.max(D) < 1e-12:
                break
            if generation > 100 + 50 * dim:
                break
    
    return best
