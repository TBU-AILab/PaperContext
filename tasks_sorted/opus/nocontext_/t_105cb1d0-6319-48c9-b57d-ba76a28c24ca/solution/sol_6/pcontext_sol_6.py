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
    
    # Initialize center at random or midpoint
    mu = (lower + upper) / 2.0
    sigma = np.mean(upper - lower) / 4.0
    
    # Evaluate center
    fitness_center = func(mu)
    if fitness_center < best:
        best = fitness_center
        best_params = mu.copy()
    
    # CMA-ES parameters
    weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size // 2 + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    n_parents = len(weights)
    
    # Adaptation parameters
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    # State variables
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    C = np.eye(dim)
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    
    # Multiple restarts
    restart_count = 0
    max_restarts = 50
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        try:
            # Generate eigendecomposition
            if generation % 1 == 0:
                try:
                    C = (C + C.T) / 2  # Ensure symmetry
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)
                    sqrt_eigenvalues = np.sqrt(eigenvalues)
                    invsqrt_C = eigenvectors @ np.diag(1.0 / sqrt_eigenvalues) @ eigenvectors.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    eigenvalues = np.ones(dim)
                    sqrt_eigenvalues = np.ones(dim)
                    eigenvectors = np.eye(dim)
                    invsqrt_C = np.eye(dim)
            
            # Sample population
            population = []
            fitnesses = []
            
            for i in range(pop_size):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                
                z = np.random.randn(dim)
                y = eigenvectors @ (sqrt_eigenvalues * z)
                x = mu + sigma * y
                
                # Clip to bounds
                x = np.clip(x, lower, upper)
                
                f = func(x)
                population.append(x)
                fitnesses.append(f)
                
                if f < best:
                    best = f
                    best_params = x.copy()
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            
            # Recombination
            old_mu = mu.copy()
            mu = np.zeros(dim)
            for i in range(n_parents):
                mu += weights[i] * population[indices[i]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrt_C @ (mu - old_mu) / sigma
            
            # CCA
            h_sig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)
            pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * (mu - old_mu) / sigma
            
            # Covariance matrix adaptation
            artmp = np.zeros((dim, n_parents))
            for i in range(n_parents):
                artmp[:, i] = (population[indices[i]] - old_mu) / sigma
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - h_sig) * cc * (2 - cc) * C) + cmu * artmp @ np.diag(weights) @ artmp.T
            
            # Step size update
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, np.mean(upper - lower))
            
            generation += 1
            
            # Check stagnation
            if abs(best - prev_best) < 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best = best
            
            # Restart if stagnated or sigma too small
            if stagnation_count > 20 + dim or sigma < 1e-16 or np.max(eigenvalues) / np.min(eigenvalues) > 1e14:
                restart_count += 1
                
                # Restart with increased population or random center
                if restart_count < max_restarts:
                    pop_size_new = int(pop_size * (1 + restart_count * 0.5))
                    pop_size = min(pop_size_new, 200)
                    
                    # Random restart point
                    mu = np.array([np.random.uniform(l, u) for l, u in bounds])
                    sigma = np.mean(upper - lower) / (2 + restart_count * 0.5)
                    
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    eigenvalues = np.ones(dim)
                    sqrt_eigenvalues = np.ones(dim)
                    eigenvectors = np.eye(dim)
                    invsqrt_C = np.eye(dim)
                    
                    weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size // 2 + 1))
                    weights = weights / np.sum(weights)
                    mu_eff = 1.0 / np.sum(weights ** 2)
                    n_parents = len(weights)
                    
                    cs = (mu_eff + 2) / (dim + mu_eff + 5)
                    ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
                    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
                    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
                    
                    generation = 0
                    stagnation_count = 0
                    prev_best = best
                else:
                    # Fall back to local search around best
                    break
                    
        except Exception:
            # If anything goes wrong, restart
            mu = np.array([np.random.uniform(l, u) for l, u in bounds])
            sigma = np.mean(upper - lower) / 4.0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            generation = 0
            stagnation_count = 0
    
    # Phase 2: Local Nelder-Mead style refinement around best
    if best_params is not None:
        step = np.mean(upper - lower) * 0.01
        while True:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            improved = False
            for i in range(dim):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                
                trial = best_params.copy()
                trial[i] = min(trial[i] + step, upper[i])
                f = func(trial)
                if f < best:
                    best = f
                    best_params = trial.copy()
                    improved = True
                    continue
                
                trial = best_params.copy()
                trial[i] = max(trial[i] - step, lower[i])
                f = func(trial)
                if f < best:
                    best = f
                    best_params = trial.copy()
                    improved = True
            
            if not improved:
                step *= 0.5
                if step < 1e-15:
                    break
    
    return best
