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
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    n_init = min(max(20 * dim, 100), 500)
    
    init_samples = []
    for i in range(n_init):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        params = lower + (upper - lower) * np.random.rand(dim)
        fitness = func(params)
        init_samples.append((fitness, params.copy()))
        if fitness < best:
            best = fitness
            best_params = params.copy()
    
    # Sort by fitness
    init_samples.sort(key=lambda x: x[0])
    
    # --- Phase 2: CMA-ES inspired search ---
    # Use top candidates to initialize
    pop_size = max(4 + int(3 * np.log(dim)), 20)
    pop_size = min(pop_size, 50)
    mu = pop_size // 2
    
    # Initialize mean from best solution
    mean = best_params.copy()
    sigma = 0.3 * np.mean(upper - lower)
    
    # Weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    # Adaptation parameters
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu_cov = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    
    # State variables
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    
    # Use diagonal covariance for high dimensions
    use_full_cov = dim <= 100
    
    if use_full_cov:
        C = np.eye(dim)
    else:
        C_diag = np.ones(dim)
    
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.95):
            return best
        
        # Generate population
        population = []
        fitnesses = []
        
        if use_full_cov:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                invsqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C = np.eye(dim)
                sqrt_C = np.eye(dim)
                invsqrt_C = np.eye(dim)
        else:
            sqrt_C_diag = np.sqrt(np.maximum(C_diag, 1e-20))
            invsqrt_C_diag = 1.0 / sqrt_C_diag
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
                return best
            
            z = np.random.randn(dim)
            if use_full_cov:
                y = sqrt_C @ z
            else:
                y = sqrt_C_diag * z
            
            x = mean + sigma * y
            # Clip to bounds
            x = np.clip(x, lower, upper)
            
            fitness = func(x)
            population.append((x.copy(), y.copy(), z.copy()))
            fitnesses.append(fitness)
            
            if fitness < best:
                best = fitness
                best_params = x.copy()
        
        # Sort by fitness
        indices = np.argsort(fitnesses)
        
        # Recombination
        old_mean = mean.copy()
        mean = np.zeros(dim)
        y_w = np.zeros(dim)
        for i in range(mu):
            idx = indices[i]
            mean += weights[i] * population[idx][0]
            y_w += weights[i] * population[idx][1]
        
        # Clip mean to bounds
        mean = np.clip(mean, lower, upper)
        
        # CSA: cumulation for sigma
        if use_full_cov:
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrt_C @ y_w)
        else:
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrt_C_diag * y_w)
        
        h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
        
        # CCA: cumulation for rank-one update
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        
        # Covariance matrix adaptation
        if use_full_cov:
            artmp = np.zeros((dim, mu))
            for i in range(mu):
                idx = indices[i]
                artmp[:, i] = population[idx][1]
            
            C = (1 - c1 - c_mu_cov + (1 - h_sigma) * c1 * c_c * (2 - c_c)) * C \
                + c1 * np.outer(p_c, p_c) \
                + c_mu_cov * (artmp * weights[np.newaxis, :]) @ artmp.T
            
            # Enforce symmetry
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-12 * np.eye(dim)
        else:
            C_diag = (1 - c1 - c_mu_cov + (1 - h_sigma) * c1 * c_c * (2 - c_c)) * C_diag \
                     + c1 * p_c ** 2
            for i in range(mu):
                idx = indices[i]
                C_diag += c_mu_cov * weights[i] * population[idx][1] ** 2
            C_diag = np.maximum(C_diag, 1e-20)
        
        # Sigma adaptation
        sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma = min(sigma, 2 * np.mean(upper - lower))
        sigma = max(sigma, 1e-15)
        
        generation += 1
        
        # Restart mechanism
        if abs(best - prev_best) < 1e-12:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 50 + 10 * dim or sigma < 1e-15:
            # Restart with larger sigma centered around best or random
            if np.random.rand() < 0.5:
                mean = best_params.copy() + 0.1 * (upper - lower) * np.random.randn(dim)
                mean = np.clip(mean, lower, upper)
            else:
                mean = lower + (upper - lower) * np.random.rand(dim)
            sigma = 0.3 * np.mean(upper - lower)
            if use_full_cov:
                C = np.eye(dim)
            else:
                C_diag = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            stagnation_count = 0
            generation = 0
    
    return best
