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
    pop_size = max(20, 4 + int(3 * np.log(dim)))
    mu = pop_size // 2  # number of parents
    
    # Weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)
    
    # Adaptation parameters
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    
    # State variables
    mean = lower + (upper - lower) * np.random.rand(dim)
    sigma = np.mean(upper - lower) / 4.0
    C = np.eye(dim)
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    generation = 0
    no_improve_count = 0
    
    # Evaluate initial mean
    mean_clipped = np.clip(mean, lower, upper)
    fitness = func(mean_clipped)
    if fitness < best:
        best = fitness
        best_params = mean_clipped.copy()
    
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    # Main CMA-ES loop
    while time_left() > 0.5:
        generation += 1
        
        # Eigendecomposition of C
        try:
            if dim <= 200:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                sqrt_eigenvalues = np.sqrt(eigenvalues)
                invsqrt_C = eigenvectors @ np.diag(1.0 / sqrt_eigenvalues) @ eigenvectors.T
            else:
                # For high dimensions, use diagonal approximation
                diag_C = np.diag(C).copy()
                diag_C = np.maximum(diag_C, 1e-20)
                sqrt_eigenvalues = np.sqrt(diag_C)
                eigenvalues = diag_C
                eigenvectors = np.eye(dim)
                invsqrt_C = np.diag(1.0 / sqrt_eigenvalues)
        except:
            C = np.eye(dim)
            eigenvalues = np.ones(dim)
            eigenvectors = np.eye(dim)
            sqrt_eigenvalues = np.ones(dim)
            invsqrt_C = np.eye(dim)
        
        # Sample new population
        solutions = []
        fitnesses = []
        
        for i in range(pop_size):
            if time_left() <= 0.2:
                return best
            
            z = np.random.randn(dim)
            if dim <= 200:
                y = eigenvectors @ (sqrt_eigenvalues * z)
            else:
                y = sqrt_eigenvalues * z
            
            x = mean + sigma * y
            x_clipped = np.clip(x, lower, upper)
            
            fit = func(x_clipped)
            solutions.append((x_clipped, y, fit))
            fitnesses.append(fit)
            
            if fit < best:
                best = fit
                best_params = x_clipped.copy()
        
        # Sort by fitness
        indices = np.argsort(fitnesses)
        
        # Recombination
        old_mean = mean.copy()
        mean = np.zeros(dim)
        y_w = np.zeros(dim)
        for i in range(mu):
            idx = indices[i]
            mean += weights[i] * solutions[idx][0]
            y_w += weights[i] * solutions[idx][1]
        
        # Update evolution paths
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrt_C @ y_w)
        
        h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(generation+1)))) < (1.4 + 2/(dim+1)) * chi_n else 0
        
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        
        # Update covariance matrix
        if dim <= 200:
            C = (1 - c1 - c_mu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(mu):
                idx = indices[i]
                yi = solutions[idx][1]
                C += c_mu * weights[i] * np.outer(yi, yi)
        else:
            # Diagonal update for high dimensions
            diag_update = (1 - c1 - c_mu) * np.diag(C) + c1 * (p_c**2 + (1 - h_sigma) * c_c * (2 - c_c) * np.diag(C))
            for i in range(mu):
                idx = indices[i]
                yi = solutions[idx][1]
                diag_update += c_mu * weights[i] * yi**2
            C = np.diag(diag_update)
        
        # Update step size
        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma = min(sigma, np.mean(upper - lower))  # cap sigma
        
        # Check improvement
        gen_best = fitnesses[indices[0]]
        if gen_best >= best:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # Restart if stagnated
        if no_improve_count > 50 + 10 * dim or sigma < 1e-16:
            # Restart with different initialization
            mean = lower + (upper - lower) * np.random.rand(dim)
            sigma = np.mean(upper - lower) / 4.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            no_improve_count = 0
            generation = 0
    
    # --- Phase 2: Local refinement with Nelder-Mead style if time remains ---
    while time_left() > 0.1:
        if best_params is None:
            break
        
        # Try small perturbations around best
        perturbation_scale = 0.001 * (upper - lower)
        candidate = best_params + perturbation_scale * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        
        fit = func(candidate)
        if fit < best:
            best = fit
            best_params = candidate.copy()
    
    return best
