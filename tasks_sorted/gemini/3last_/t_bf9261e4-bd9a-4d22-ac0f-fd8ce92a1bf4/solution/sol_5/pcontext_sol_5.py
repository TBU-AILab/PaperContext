#The following algorithm implements **IPO-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy)**.
#
#**Improvements and Justification:**
#1.  **Covariance Matrix Adaptation (CMA)**: Unlike Differential Evolution, CMA-ES learns the correlations between variables (represented by matrix $C$). This allows it to solve non-separable (rotated) problems efficiently, which the previous results (21.4 vs 49.1) suggest is the case here.
#2.  **IPO Restart Strategy**: The algorithm detects stagnation (e.g., condition number too high, step size too small) and restarts with an **exponentially increasing population size** ($ \lambda \cdot 2^k $). This ensures that if a small population gets stuck in a local optimum, the next iteration uses a larger population to smooth the landscape and increase global exploration capability.
#3.  **Lazy Eigendecomposition**: The computationally expensive matrix decomposition (O($N^3$)) is performed only when necessary (based on the adaptation rate), maximizing the number of function evaluations within the time limit.
#4.  **Vectorized Operations**: Matrix multiplications are used for rank-$\mu$ updates and coordinate transformations, ensuring high performance in Python.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPO-CMA-ES (Increasing Population Size 
    Covariance Matrix Adaptation Evolution Strategy).
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Use 95% of allocated time to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.95)

    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Pre-process Bounds ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_range = ub - lb
    
    # Track global best
    best_fitness = float('inf')
    best_sol = None

    # --- Main Loop (Restarts) ---
    restart_idx = 0
    
    while has_time():
        # 1. Population Size Strategy (IPO)
        # Base size: 4 + 3*log(D). Double it every restart.
        lam = int((4 + 3 * np.log(dim)) * (2 ** restart_idx))
        restart_idx += 1
        
        # 2. Initialization
        # Mean: Random position within bounds
        xmean = lb + np.random.rand(dim) * bound_range
        
        # Step size (Sigma): Start with 30% of domain width
        sigma = 0.3 * np.mean(bound_range)
        
        # Covariance Matrix components (Start with Identity)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Selection Weights (Logarithmic)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # CMA-ES Adaptation Constants (Standard/Tuned)
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        counteval = 0
        eigeneval = 0
        
        # --- Generation Loop ---
        while has_time():
            # 3. Sampling
            # z ~ N(0, I)
            arz = np.random.randn(lam, dim)
            # y ~ N(0, C) => y = B * D * z
            ary = arz @ (np.diag(D) @ B.T)
            # x = m + sigma * y
            arx = xmean + sigma * ary
            
            # 4. Evaluation with Boundary Handling
            fitness = []
            
            for i in range(lam):
                if not has_time(): return best_fitness
                
                # Boundary Constraint: Clip
                # We evaluate the clipped solution but penalize the distribution 
                # based on distance to encourage the mean to stay feasible.
                x_gen = arx[i]
                x_eval = np.clip(x_gen, lb, ub)
                
                val = func(x_eval)
                counteval += 1
                
                # Update Global Best
                if val < best_fitness:
                    best_fitness = val
                    best_sol = x_eval.copy()
                
                # Quadratic Penalty for internal selection
                # This keeps the Gaussian distribution near the bounds if optimum is there.
                dist_sq = np.sum((x_gen - x_eval)**2)
                penalty = 1e8 * dist_sq
                fitness.append(val + penalty)
            
            fitness = np.array(fitness)
            
            # 5. Selection and Recombination
            sorted_idx = np.argsort(fitness)
            best_idx = sorted_idx[:mu]
            
            # Save old mean
            xold = xmean.copy()
            
            # Compute weighted mean of top mu vectors
            # zmean and ymean are used for updating paths and matrix
            zmean = weights @ arz[best_idx]
            ymean = weights @ ary[best_idx]
            
            # Update Mean
            xmean = xold + sigma * ymean
            
            # 6. Step-size Control (Evolution Path p_s)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            
            # Check for stall (h_sig)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (counteval/lam + 1))) / chiN < 1.4 + 2/(dim+1)
            
            # 7. Covariance Adaptation (Evolution Path p_c)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
            
            # Update C matrix
            # Rank-1 update (using pc) + Rank-mu update (using selected steps)
            dh_correction = (1 - hsig) * cc * (2 - cc)
            
            # Efficient Rank-mu calculation: y.T @ (y * weights)
            y_best = ary[best_idx]
            rank_mu_part = y_best.T @ (y_best * weights[:, np.newaxis])
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + dh_correction * C) + cmu * rank_mu_part
            
            # 8. Update Sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 9. Lazy Eigendecomposition
            # Update B and D only every few generations to save time (O(N^3))
            # Heuristic gap: approx 1/(c1+cmu)/N/10
            gap = max(1, int(1 / (c1 + cmu) / dim / 10))
            if (counteval // lam) % gap == 0:
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    vals, vecs = np.linalg.eigh(C)
                    
                    # Numerical stability: enforce positive eigenvalues
                    if np.min(vals) <= 0:
                        vals = np.maximum(vals, 1e-14)
                        
                    D = np.sqrt(vals)
                    B = vecs
                    
                    # Update internal decomposition tracker
                    eigeneval = counteval
                except np.linalg.LinAlgError:
                    break # Force restart on matrix error
            
            # 10. Restart Criteria
            # Escape if converged or stuck
            
            # Condition number too high (ill-conditioned)
            if np.max(D) > 1e14 * np.min(D):
                break
                
            # Sigma too small (converged to precision limit)
            if sigma < 1e-16:
                break
                
            # Sigma too large (diverged)
            if sigma > 1e10:
                break
            
            # Fitness stagnation (flat population)
            if np.std(fitness[best_idx]) < 1e-12:
                break
                
            # No effect axis (standard deviation in any direction is negligible)
            if np.max(D) * sigma < 1e-12:
                break

    return best_fitness
