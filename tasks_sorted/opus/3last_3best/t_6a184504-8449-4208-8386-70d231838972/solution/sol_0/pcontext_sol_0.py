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
    init_points = np.array(init_points).T  # shape: (n_init, dim)
    
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
    cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mueff)
    cmu_param = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
    
    chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    # State variables
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C = np.eye(dim)
    eigeneval = 0
    counteval = 0
    
    B = np.eye(dim)
    D = np.ones(dim)
    invsqrtC = np.eye(dim)
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.95):
            return best
        
        # Generate and evaluate lambda offspring
        arxs = []
        fitnesses = []
        
        # Update eigen decomposition periodically
        if counteval - eigeneval > lam / (c1 + cmu_param) / dim / 10:
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
        
        for k in range(lam):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
                return best
            
            z = np.random.randn(dim)
            x = mean + sigma * (B @ (D * z))
            # Clip to bounds
            x = np.clip(x, lower, upper)
            
            f = func(x)
            counteval += 1
            arxs.append(x)
            fitnesses.append(f)
            
            if f < best:
                best = f
                best_x = x.copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        arxs = [arxs[i] for i in idx]
        
        # Recombination: new mean
        old_mean = mean.copy()
        mean = np.zeros(dim)
        for i in range(mu):
            mean += weights[i] * arxs[i]
        
        # CSA: cumulation for sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
        
        hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2/(dim + 1))
        
        # CMA: cumulation for covariance
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
        
        # Covariance matrix adaptation
        artmp = np.zeros((mu, dim))
        for i in range(mu):
            artmp[i] = (arxs[i] - old_mean) / sigma
        
        C = (1 - c1 - cmu_param) * C + \
            c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
            cmu_param * (artmp.T @ np.diag(weights) @ artmp)
        
        # Sigma adaptation
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.mean(upper - lower))  # cap sigma
        
        generation += 1
        
        # Restart logic: if stagnating, restart from best with perturbation
        if generation % 20 == 0:
            if abs(prev_best - best) < 1e-12 * (abs(best) + 1e-12):
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best = best
            
            if stagnation_count >= 3:
                # Restart
                sigma = np.mean(upper - lower) / 4.0
                mean = best_x.copy() + np.random.randn(dim) * sigma * 0.1
                mean = np.clip(mean, lower, upper)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                invsqrtC = np.eye(dim)
                eigeneval = 0
                stagnation_count = 0
    
    return best
