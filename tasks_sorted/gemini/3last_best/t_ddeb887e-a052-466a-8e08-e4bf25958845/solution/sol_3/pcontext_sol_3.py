#Here is an improved algorithm. It implements **IPO-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy)**.
#
#**Key Improvements over previous algorithms:**
#1.  **Covariance Matrix Adaptation:** Unlike Differential Evolution (which operates on axes or random differences), CMA-ES learns the correlation between variables. It creates a "search ellipse" that aligns with the topology of the function (e.g., following a narrow valley in the fitness landscape).
#2.  **Restart Mechanism (IPO):** The algorithm detects convergence (stagnation). When it gets stuck, it restarts with a **doubled population size**. This theoretically guarantees finding the global optimum on multimodal functions given sufficient time, preventing the "getting stuck" issue seen in previous algorithms.
#3.  **Step-Size Control:** It adapts a global step size ($\sigma$) using cumulative path length control, allowing it to converge rapidly when close to the solution and expand search when far away.
#4.  **Boundary Handling:** It uses a soft penalty mechanism to guide the probabilistic distribution back within bounds without breaking the internal matrix adaptation logic.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using IPO-CMA-ES (Increasing Population Size 
    Covariance Matrix Adaptation Evolution Strategy).
    
    Strategies:
    1. Adaptation of the covariance matrix to learn variable dependencies (rotation).
    2. Path length control for step-size adaptation.
    3. Restart mechanism with increasing population size to escape local optima.
    """
    start_time = time.time()
    
    # --- Configuration ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # Global best tracking
    best_fitness = float('inf')
    
    # --- Restart Loop (IPO Strategy) ---
    # Start with standard population size for CMA-ES: 4 + 3*ln(dim)
    # We restart with pop_size * 2 if convergence is detected.
    pop_size = 4 + int(3 * np.log(dim))
    
    while True:
        # Check overall time budget
        if time.time() - start_time >= max_time:
            return best_fitness

        # --- Initialize CMA Parameters for this restart ---
        # Initialize mean uniformly within bounds
        mean = min_b + np.random.rand(dim) * (max_b - min_b)
        
        # Initial step size (sigma)
        # Set to 0.3 * range to cover significant space initially
        sigma = 0.3 * np.max(max_b - min_b)
        
        # Initialize Covariance Matrix components
        # C = B * D^2 * B'
        pc = np.zeros(dim) # Evolution path for C
        ps = np.zeros(dim) # Evolution path for sigma
        B = np.eye(dim)    # Eigenvectors
        D = np.ones(dim)   # Square root of eigenvalues
        C = np.eye(dim)    # Covariance matrix
        
        # Selection weights (Logarithmic profile)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights) # Normalize
        mueff = 1 / np.sum(weights**2) # Variance effective selection mass
        
        # Adaptation constants (Standard CMA-ES tuning)
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2)) # Expectation of ||N(0,I)||
        
        # Loop Variables
        gen = 0
        eigen_interval = max(1, int(dim / 10)) # Update B, D every few gens for speed
        
        # --- Generation Loop ---
        while True:
            gen += 1
            if time.time() - start_time >= max_time:
                return best_fitness
            
            # 1. Sampling
            # z ~ N(0, I)
            z = np.random.randn(pop_size, dim)
            
            # y ~ N(0, C) => y = B * D * z
            y = B @ (D[:, None] * z.T) # shape (dim, pop_size)
            y = y.T # shape (pop_size, dim)
            
            # x = mean + sigma * y
            x = mean + sigma * y
            
            # 2. Evaluation with Penalty for Bounds
            fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return best_fitness
                
                # Constrain query to bounds
                x_c = np.clip(x[i], min_b, max_b)
                
                # Compute distance from bounds for penalty
                # CMA-ES works best if the internal gaussian is guided back to valid space
                dist_sq = np.sum((x[i] - x_c)**2)
                
                f_val = func(x_c)
                
                # Update global best with the VALID (clipped) solution
                if f_val < best_fitness:
                    best_fitness = f_val
                
                # Add soft penalty to fitness used for CMA updates
                # This encourages the distribution to stay within bounds
                fitness[i] = f_val + (1e3 * dist_sq if dist_sq > 0 else 0)
                
            # 3. Selection and Recombination
            sorted_idx = np.argsort(fitness)
            z_sel = z[sorted_idx[:mu]] # Best mu vectors
            y_sel = y[sorted_idx[:mu]] 
            
            # New mean is weighted average of best samples
            mean_old = mean.copy()
            step = np.dot(weights, y_sel) # Weighted sum of steps
            mean = mean_old + sigma * step
            
            # Optional: Clip mean to bounds to prevent complete drift
            mean = np.clip(mean, min_b, max_b)
            
            # 4. Step-size Control (Evolution Path for Sigma)
            # Calculate C^-1/2 * step. Since step = B*D*z_w, 
            # C^-1/2 * step = B*D^-1*B' * B*D*z_w = B*z_w.
            z_w = np.dot(weights, z_sel)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_w)
            
            norm_ps = np.linalg.norm(ps)
            hsig = 1 if norm_ps / np.sqrt(1 - (1 - cs)**(2*gen)) / chiN < 1.4 + 2/(dim + 1) else 0
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            
            # 5. Covariance Matrix Adaptation
            # Evolution Path for C
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * step
            
            # Rank-1 Update + Rank-mu Update
            # rank_mu = sum(w_i * y_i * y_i.T)
            # vectorized: (y_sel.T * weights) @ y_sel
            rank_mu = (y_sel.T * weights) @ y_sel
            
            rank_1 = np.outer(pc, pc)
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            
            C = (1 - c1 - cmu) * C + \
                c1 * (rank_1 + delta_hsig * C) + \
                cmu * rank_mu
                
            # 6. Decomposition (Update B and D)
            # Expensive O(N^3), so we don't do it every generation
            if gen % eigen_interval == 0:
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    vals, vecs = np.linalg.eigh(C)
                    # Numerical stability: ensure positive eigenvalues
                    vals = np.maximum(vals, 1e-18)
                    D = np.sqrt(vals)
                    B = vecs
                    # Reconstruct C to be perfectly conditioned
                    C = B @ np.diag(vals) @ B.T
                except np.linalg.LinAlgError:
                    # If decomposition fails, restart
                    break
            
            # 7. Stop Criteria (Stagnation / Convergence Checks)
            
            # Range of fitness in best fraction is too small
            if np.max(fitness[sorted_idx[:mu]]) - np.min(fitness[sorted_idx[:mu]]) < 1e-12:
                break
                
            # Sigma too small (converged to precision limit)
            if sigma < 1e-15:
                break
                
            # Condition number too high (matrix singular)
            if np.max(D) > 1e14 * np.min(D):
                break
                
            # Sigma grew too large (diverged)
            if sigma > 1e10:
                break

        # End of Inner Loop (Restart Triggered)
        # Increase population size for better global exploration capability
        pop_size *= 2

    return best_fitness
