#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clamp(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clamp(x)
        val = func(x)
        if val < best:
            best = val
            best_params = x.copy()
        return val
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(50 * dim, max(200, 20 * dim))
    
    # Generate initial population
    population = []
    fitness_vals = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = lower + np.random.rand(dim) * ranges
        val = evaluate(x)
        population.append(x.copy())
        fitness_vals.append(val)
    
    population = np.array(population)
    fitness_vals = np.array(fitness_vals)
    
    # Phase 2: CMA-ES inspired search
    # Sort and pick elite
    pop_size = min(max(20, 4 * dim), 100)
    mu = pop_size // 2
    
    # Initialize from best found so far
    sorted_idx = np.argsort(fitness_vals)
    elite = population[sorted_idx[:mu]]
    
    mean = np.average(elite, axis=0, weights=np.arange(mu, 0, -1).astype(float))
    sigma = np.mean(ranges) * 0.3
    
    # Weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    # Adaptation parameters
    c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))
    
    chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
    
    # State variables
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    
    if dim <= 100:
        C = np.eye(dim)
        use_full_cov = True
    else:
        diag_C = np.ones(dim)
        use_full_cov = False
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    while elapsed() < max_time * 0.95:
        generation += 1
        
        # Generate offspring
        if use_full_cov:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except Exception:
                C = np.eye(dim)
                sqrt_C = np.eye(dim)
                eigvals = np.ones(dim)
                eigvecs = np.eye(dim)
        
        offspring = []
        offspring_fitness = []
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.95:
                return best
            
            z = np.random.randn(dim)
            if use_full_cov:
                y = sqrt_C @ z
            else:
                y = np.sqrt(diag_C) * z
            
            x = mean + sigma * y
            x = clamp(x)
            val = evaluate(x)
            offspring.append(x)
            offspring_fitness.append(val)
        
        offspring = np.array(offspring)
        offspring_fitness = np.array(offspring_fitness)
        
        # Sort by fitness
        sorted_idx = np.argsort(offspring_fitness)
        
        # Recombination
        old_mean = mean.copy()
        selected = offspring[sorted_idx[:mu]]
        mean = np.sum(weights[:, None] * selected, axis=0)
        
        # Update evolution paths
        mean_diff = (mean - old_mean) / sigma
        
        if use_full_cov:
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ mean_diff)
        else:
            inv_sqrt_diag = 1.0 / np.sqrt(diag_C)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_diag * mean_diff)
        
        h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) < (1.4 + 2.0 / (dim + 1)) * chi_n else 0.0
        
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff
        
        # Update covariance
        if use_full_cov:
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - c_mu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(artmp[i], artmp[i])
            # Enforce symmetry
            C = (C + C.T) / 2.0
            # Add small diagonal for numerical stability
            C += 1e-20 * np.eye(dim)
        else:
            artmp = (selected - old_mean) / sigma
            diag_C = (1 - c1 - c_mu) * diag_C + c1 * (p_c ** 2 + (1 - h_sigma) * c_c * (2 - c_c) * diag_C)
            for i in range(mu):
                diag_C += c_mu * weights[i] * artmp[i] ** 2
            diag_C = np.maximum(diag_C, 1e-20)
        
        # Update step size
        sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma = max(sigma, 1e-20)
        sigma = min(sigma, np.mean(ranges) * 2)
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12 * (1 + abs(best)):
            stagnation_count += 1
        else:
            stagnation_count = 0
            prev_best = best
        
        # Restart if stagnated
        if stagnation_count > 20 + dim:
            # Restart with reduced range around best
            if best_params is not None:
                restart_range = ranges * 0.5
                mean = best_params + np.random.randn(dim) * restart_range * 0.1
                mean = clamp(mean)
            else:
                mean = lower + np.random.rand(dim) * ranges
            
            sigma = np.mean(ranges) * 0.3
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            if use_full_cov:
                C = np.eye(dim)
            else:
                diag_C = np.ones(dim)
            stagnation_count = 0
    
    return best
