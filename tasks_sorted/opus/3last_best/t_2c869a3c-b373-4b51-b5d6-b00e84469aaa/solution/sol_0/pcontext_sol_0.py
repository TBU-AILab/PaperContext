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
    init_points = np.array(init_points).T
    
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
    
    pop_size = 4 + int(3 * np.log(dim))
    pop_size = max(pop_size, 10)
    mu = pop_size // 2
    
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    # Adaptation parameters
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    C = np.eye(dim)
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        
        # Restart if stagnation or sigma too small
        if sigma < 1e-12 or stagnation_count > 50 + 10 * dim:
            # Restart around best known or random
            if np.random.random() < 0.5:
                mean = best_x.copy()
                sigma = np.mean(upper - lower) / 8.0
            else:
                mean = np.array([np.random.uniform(l, u) for l, u in bounds])
                sigma = np.mean(upper - lower) / 4.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            stagnation_count = 0
        
        try:
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            D = np.sqrt(eigenvalues)
        except:
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
        
        invsqrtC = B @ np.diag(1.0 / D) @ B.T
        
        # Generate offspring
        solutions = []
        fitnesses = []
        for k in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.95:
                return best
            z = np.random.randn(dim)
            x = mean + sigma * (B @ (D * z))
            # Clip to bounds
            x = np.clip(x, lower, upper)
            f = func(x)
            solutions.append(x)
            fitnesses.append(f)
            if f < best:
                best = f
                best_x = x.copy()
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        
        # Update mean
        old_mean = mean.copy()
        mean = np.zeros(dim)
        for i in range(mu):
            mean += weights[i] * solutions[idx[i]]
        
        # Update evolution paths
        mean_diff = mean - old_mean
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * invsqrtC @ mean_diff / sigma
        
        h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
        
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff / sigma
        
        # Update covariance matrix
        artmp = np.zeros((dim, mu))
        for i in range(mu):
            artmp[:, i] = (solutions[idx[i]] - old_mean) / sigma
        
        C = (1 - c1 - c_mu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * (artmp * weights) @ artmp.T
        
        # Enforce symmetry
        C = (C + C.T) / 2
        
        # Update sigma
        sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma = min(sigma, np.mean(upper - lower))  # cap sigma
        
        generation += 1
        
        if best < prev_best - 1e-10:
            stagnation_count = 0
            prev_best = best
        else:
            stagnation_count += 1
    
    return best
