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
    
    # Parameters for CMA-ES
    pop_size = 4 + int(3 * np.log(dim))
    if pop_size < 8:
        pop_size = 8
    mu = pop_size // 2
    
    # Weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    # Adaptation parameters
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    # Number of restarts with increasing population
    restart_count = 0
    
    def time_remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    while time_remaining() > 0.1:
        # Initialize for this restart
        current_pop_size = pop_size * (2 ** restart_count)
        current_mu = current_pop_size // 2
        
        # Recompute weights
        w = np.log(current_mu + 0.5) - np.log(np.arange(1, current_mu + 1))
        w = w / np.sum(w)
        mu_eff_r = 1.0 / np.sum(w ** 2)
        
        cs_r = (mu_eff_r + 2) / (dim + mu_eff_r + 5)
        ds_r = 1 + 2 * max(0, np.sqrt((mu_eff_r - 1) / (dim + 1)) - 1) + cs_r
        cc_r = (4 + mu_eff_r / dim) / (dim + 4 + 2 * mu_eff_r / dim)
        c1_r = 2 / ((dim + 1.3) ** 2 + mu_eff_r)
        cmu_r = min(1 - c1_r, 2 * (mu_eff_r - 2 + 1 / mu_eff_r) / ((dim + 2) ** 2 + mu_eff_r))
        
        # Initialize mean, sigma, covariance
        if best_params is not None and np.random.random() < 0.3:
            mean = best_params.copy() + np.random.randn(dim) * 0.1 * (upper - lower)
            mean = np.clip(mean, lower, upper)
        else:
            mean = np.random.uniform(lower, upper)
        
        sigma = 0.3 * np.mean(upper - lower)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = dim <= 100
        
        if use_full_cov:
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
        else:
            diagC = np.ones(dim)
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        eigeneval = 0
        count_eval = 0
        stagnation = 0
        prev_best_fitness = float('inf')
        
        max_iter = 100 + 200 * dim // current_pop_size
        
        for generation in range(max_iter):
            if time_remaining() < 0.05:
                return best
            
            # Generate offspring
            solutions = []
            fitnesses = []
            
            for i in range(current_pop_size):
                if time_remaining() < 0.05:
                    return best
                
                if use_full_cov:
                    z = np.random.randn(dim)
                    x = mean + sigma * (B @ (D * z))
                else:
                    z = np.random.randn(dim)
                    x = mean + sigma * (np.sqrt(diagC) * z)
                
                # Clip to bounds
                x = np.clip(x, lower, upper)
                
                f = func(x)
                solutions.append(x)
                fitnesses.append(f)
                
                if f < best:
                    best = f
                    best_params = x.copy()
            
            # Sort by fitness
            idx = np.argsort(fitnesses)
            sorted_solutions = [solutions[i] for i in idx]
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(dim)
            for i in range(current_mu):
                mean += w[i] * sorted_solutions[i]
            
            # Clip mean
            mean = np.clip(mean, lower, upper)
            
            diff = (mean - old_mean) / sigma
            
            # Update evolution paths
            if use_full_cov:
                ps = (1 - cs_r) * ps + np.sqrt(cs_r * (2 - cs_r) * mu_eff_r) * (invsqrtC @ diff)
            else:
                ps = (1 - cs_r) * ps + np.sqrt(cs_r * (2 - cs_r) * mu_eff_r) * (diff / np.sqrt(diagC))
            
            hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs_r) ** (2 * (count_eval / current_pop_size + 1))) / chi_n < 1.4 + 2 / (dim + 1) else 0
            
            pc = (1 - cc_r) * pc + hs * np.sqrt(cc_r * (2 - cc_r) * mu_eff_r) * diff
            
            # Update covariance
            if use_full_cov:
                artmp = np.zeros((dim, current_mu))
                for i in range(current_mu):
                    artmp[:, i] = (sorted_solutions[i] - old_mean) / sigma
                
                C = (1 - c1_r - cmu_r + (1 - hs) * c1_r * cc_r * (2 - cc_r)) * C \
                    + c1_r * np.outer(pc, pc) \
                    + cmu_r * (artmp * w[np.newaxis, :]) @ artmp.T
                
                # Enforce symmetry
                C = (C + C.T) / 2
                
                # Update eigendecomposition
                count_eval += current_pop_size
                eigeneval += current_pop_size
                if eigeneval >= current_pop_size / (c1_r + cmu_r) / dim / 10:
                    eigeneval = 0
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        C = np.eye(dim)
                        B = np.eye(dim)
                        D = np.ones(dim)
                        invsqrtC = np.eye(dim)
            else:
                artmp = np.zeros((current_mu, dim))
                for i in range(current_mu):
                    artmp[i] = (sorted_solutions[i] - old_mean) / sigma
                
                diagC = (1 - c1_r - cmu_r + (1 - hs) * c1_r * cc_r * (2 - cc_r)) * diagC \
                    + c1_r * pc ** 2 \
                    + cmu_r * np.sum(w[:, np.newaxis] * artmp ** 2, axis=0)
                
                diagC = np.maximum(diagC, 1e-20)
                count_eval += current_pop_size
            
            # Update sigma
            sigma *= np.exp((cs_r / ds_r) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6 * np.mean(upper - lower))
            sigma = max(sigma, 1e-20)
            
            # Check stagnation
            gen_best = fitnesses[idx[0]]
            if gen_best >= prev_best_fitness - 1e-12 * abs(prev_best_fitness + 1e-30):
                stagnation += 1
            else:
                stagnation = 0
            prev_best_fitness = gen_best
            
            # Termination conditions
            if sigma < 1e-16:
                break
            if stagnation > 10 + int(30 * dim / current_pop_size):
                break
            if use_full_cov and np.max(D) / np.min(D) > 1e7:
                break
        
        restart_count += 1
        
        # After some restarts, also try local search around best
        if best_params is not None and time_remaining() > 0.5:
            # Nelder-Mead like local refinement
            local_sigma = 0.01 * np.mean(upper - lower)
            for _ in range(min(50 * dim, int(time_remaining() * 100))):
                if time_remaining() < 0.05:
                    return best
                perturbation = np.random.randn(dim) * local_sigma
                candidate = np.clip(best_params + perturbation, lower, upper)
                f = func(candidate)
                if f < best:
                    best = f
                    best_params = candidate.copy()
                    local_sigma *= 1.1
                else:
                    local_sigma *= 0.95
                if local_sigma < 1e-15:
                    break
    
    return best
