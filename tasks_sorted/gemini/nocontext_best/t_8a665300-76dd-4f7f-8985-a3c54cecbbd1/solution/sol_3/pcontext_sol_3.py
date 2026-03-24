#The following algorithm implements **IPO-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy)**.
#
#### Why this is an improvement:
#1.  **Learned Geometry (CMA):** Unlike Differential Evolution (which relies on random vector differences), CMA-ES explicitly calculates and updates a Covariance Matrix. This allows it to "learn" the shape of the function (e.g., narrow, rotated valleys) and scale the search steps accordingly. It is generally considered the state-of-the-art for continuous parameter optimization in dimensions $N < 100$.
#2.  **Restart Strategy (IPO):** The algorithm detects when it has converged to a local minimum. It then restarts, but **doubles the population size**. A larger population forces a more global search capability, helping to escape local basins that trapped the previous iteration.
#3.  **Boundary Handling:** It employs a penalization method where the internal mathematical model is allowed to wander slightly outside bounds (to calculate gradients/moments correctly), but the evaluated fitness includes a heavy penalty for constraints, guiding the "cloud" of points back into the valid region.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using IPO-CMA-ES (Covariance Matrix Adaptation Evolution Strategy 
    with Increasing Population Size Restarts).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds = np.array(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    bounds_range = upper_bounds - lower_bounds
    
    # Global best tracking
    best_fitness = float('inf')
    best_solution = None

    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    def get_fitness_with_penalty(x):
        """
        Evaluates function. If x is out of bounds, repairs it for evaluation
        but adds a penalty to the fitness to guide the distribution back in.
        """
        # Repair strategy: Clamp to bounds
        x_repaired = np.clip(x, lower_bounds, upper_bounds)
        
        # Calculate squared distance from bounds (penalty metric)
        dist = x - x_repaired
        penalty_sq = np.sum(dist**2)
        
        # Evaluate valid point
        f = func(x_repaired)
        
        # If valid, return f. If invalid, return f + penalty.
        # The penalty factor scales with the typical value range to be significant.
        # We use a simple adaptive penalty or a large constant.
        if penalty_sq > 0:
            return f + 1e5 * penalty_sq + 1e5 # Large penalty
        return f

    # --- IPO-CMA-ES Loop ---
    # We start with a default lambda and increase it on every restart.
    restart_count = 0
    pop_size_factor = 1 # Multiplier for population size
    
    while not check_timeout():
        
        # 1. Initialize Strategy Parameters
        # Population size (lambda) - default heuristic
        lam = int(4 + 3 * np.log(dim)) * pop_size_factor
        
        # Parent size (mu)
        mu = int(lam / 2)
        
        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights) # Normalize
        mueff = 1 / np.sum(weights**2)      # Variance effective selection mass
        
        # Adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Dynamic State Variables
        # Initialize mean uniformly within bounds
        mean = lower_bounds + np.random.rand(dim) * bounds_range
        
        # Initial step size (sigma): 0.3 * domain width is a standard heuristic
        sigma = 0.3 * np.max(bounds_range)
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Covariance Matrix (B and D for decomposition)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim) # B * D^2 * B.T
        
        # Expectation of N(0,I)
        chiN = dim**0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Termination for this run
        stop_run = False
        gen = 0
        eigensystem_uptodate = True
        
        while not stop_run and not check_timeout():
            gen += 1
            
            # 2. Sampling
            # x_i = m + sigma * B * D * z_i
            try:
                # Generate standard normal vectors
                Z = np.random.normal(0, 1, (lam, dim))
                
                # Apply transformation
                Y = Z.dot(np.diag(D)).dot(B.T) # Y ~ N(0, C)
                X = mean + sigma * Y
                
            except Exception:
                # Fallback for numerical instability
                stop_run = True
                break
                
            # 3. Evaluation
            fitness_values = np.zeros(lam)
            for i in range(lam):
                # Check actual best (using repaired solution)
                x_candidate = X[i]
                x_repaired = np.clip(x_candidate, lower_bounds, upper_bounds)
                
                # Evaluate
                val = func(x_repaired)
                
                # Update Global Best immediately
                if val < best_fitness:
                    best_fitness = val
                    best_solution = x_repaired
                    
                # Store penalized fitness for CMA evolution logic
                # Penalize based on distance from bounds
                dist = x_candidate - x_repaired
                penalty = np.sum(dist**2)
                fitness_values[i] = val + (1e8 * penalty if penalty > 0 else 0)
                
                if check_timeout():
                    return best_fitness

            # 4. Selection and Recombination
            # Sort by fitness
            sorted_indices = np.argsort(fitness_values)
            best_idx = sorted_indices[:mu]
            
            # Compute new mean (weighted average of top mu)
            # z_mean = sum(w_i * z_i)
            z_selected = Z[best_idx]
            y_selected = Y[best_idx]
            x_selected = X[best_idx]
            
            z_w = np.dot(weights, z_selected)
            y_w = np.dot(weights, y_selected)
            
            # Update mean
            old_mean = mean
            mean = mean + sigma * y_w
            
            # 5. Step-size control (Evolution Path)
            # C^-1/2 = B * D^-1 * B.T
            # We use simplified logic here using Z directly as Z = B^-1 D^-1 Y
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(B, z_w)
            norm_ps = np.linalg.norm(ps)
            
            # hsig threshold
            hsig = (norm_ps / np.sqrt(1 - (1 - cs)**(2 * gen))) / chiN < (1.4 + 2 / (dim + 1))
            
            # 6. Covariance Matrix Adaptation
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
            
            # Rank-1 update
            rank_1 = np.outer(pc, pc)
            
            # Rank-mu update
            # Estimate C_mu from selected steps
            # C_mu = sum(w_i * y_i * y_i.T)
            # More efficient: (Y.T * weights) @ Y
            weighted_y = y_selected.T * weights
            rank_mu = weighted_y.dot(y_selected)
            
            # Update C
            old_C = C
            C = (1 - c1 - cmu) * C \
                + c1 * (rank_1 + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * rank_mu
                
            # Update sigma
            sigma = sigma * np.exp((cs / damps) * (norm_ps / chiN - 1))
            
            # 7. Decomposition (Maintenance)
            # We don't decompose every generation, it's O(N^3)
            # Do it if condition number is bad or every few iterations
            if gen % max(1, int(1 / (10 * dim * (c1 + cmu)))) == 0:
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                
                try:
                    # Eigen decomposition
                    eig_vals, eig_vecs = np.linalg.eigh(C)
                    
                    # Numerical stability checks
                    if np.any(eig_vals <= 0):
                        eig_vals = np.maximum(eig_vals, 1e-16)
                    
                    D = np.sqrt(eig_vals)
                    B = eig_vecs
                    eigensystem_uptodate = True
                except np.linalg.LinAlgError:
                    stop_run = True # Restart if matrix is corrupted
            
            # 8. Termination / Restart Criteria
            
            # Condition 1: Sigma too small (converged precision)
            if sigma < 1e-8:
                stop_run = True
            
            # Condition 2: Condition number too high (ill-conditioned)
            if np.max(D) / np.min(D) > 1e7:
                stop_run = True
            
            # Condition 3: No improvement (Stagnation)
            # In a full implementation we track history. 
            # Simplified: Stop if sigma collapses or explodes without logic
            if sigma > 1e5 * np.max(bounds_range):
                stop_run = True
                
        # --- End of Generation Loop ---
        
        # Prepare for next restart (Increase Population)
        pop_size_factor *= 2
        restart_count += 1
        
        # Cap population size to prevent extreme slowdowns
        if lam * 2 > 200 * dim: 
            pop_size_factor = 1 # Reset if too huge

    return best_fitness
