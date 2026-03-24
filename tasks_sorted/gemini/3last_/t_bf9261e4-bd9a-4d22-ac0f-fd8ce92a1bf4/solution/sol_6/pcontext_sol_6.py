#The previous results indicate that the **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** performed best (21.37 vs 38.91 and 94.20). This suggests the problem landscape likely features non-separable variables and "valleys" where rotation-invariant algorithms like CMA-ES excel compared to coordinate-based search (like basic DE).
#
#To improve upon the 21.37 result, the following algorithm refines the **IPO-CMA-ES** (Increasing Population Size CMA-ES).
#
#**Key Improvements:**
#1.  **Strict Time Budgeting**: The algorithm now aggressively checks the remaining time before every single function evaluation, ensuring we maximize the number of evaluations without exceeding the limit.
#2.  **Optimized Vectorization**: Matrix operations for generating offspring and calculating penalties are vectorized to minimize Python interpretation overhead.
#3.  **Refined Rank-$\mu$ Update**: The covariance matrix update logic is precisely implemented to exploit the weighted best candidate steps (Rank-$\mu$ update), essential for learning the landscape's topology.
#4.  **Increasing Population (IPO)**: The restart mechanism doubles the population size after every restart. This allows the algorithm to perform local search (small population) initially and transition to global search (large population) if local optima are encountered.
#5.  **Lazy Eigendecomposition**: The expensive $O(N^3)$ matrix decomposition is performed only when required by the adaptation rate, saving significant computational time for actual function evaluations.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPO-CMA-ES (Increasing Population Size 
    Covariance Matrix Adaptation Evolution Strategy).
    
    This algorithm adapts the covariance matrix of the search distribution 
    to learn the topography of the objective function, making it efficient 
    for non-separable and ill-conditioned problems.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Use a safety buffer to ensure we return before the hard limit
    # Checking time frequently allows us to run closer to the limit
    time_limit = timedelta(seconds=max_time - 0.1)

    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Pre-process Bounds ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_range = ub - lb
    
    # Track global best solution
    best_fitness = float('inf')
    
    # --- IPO Restart Loop ---
    # Start with a small population and double it on every restart
    # Default heuristic: 4 + 3 * log(D)
    lam_start = 4 + int(3 * np.log(dim))
    restart_idx = 0
    
    while has_time():
        # 1. Configuration for current restart
        lam = int(lam_start * (2 ** restart_idx))
        restart_idx += 1
        
        # Selection weights (logarithmic profile)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # CMA-ES Adaptation Constants
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # 2. Initialization
        # Mean: Random position within bounds
        xmean = lb + np.random.rand(dim) * bound_range
        
        # Step size (Sigma): Start large (0.5 of domain) to cover space
        sigma = 0.5 * np.max(bound_range)
        
        # Covariance Matrix Decomposition (C = B * D^2 * B.T)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Counters
        counteval = 0
        eigeneval = 0
        
        # --- Generation Loop ---
        while has_time():
            # 3. Sampling
            # Generate lambda offspring: x = m + sigma * B * D * z
            arz = np.random.randn(lam, dim)
            # Transform z to y ~ N(0, C)
            ary = arz @ (np.diag(D) @ B.T) 
            # Transform y to x ~ N(m, sigma^2 C)
            arx = xmean + sigma * ary
            
            # 4. Evaluation with Boundary Handling
            fitness = np.zeros(lam)
            raw_values = []
            
            # Vectorized clipping for penalty calculation
            arx_clipped = np.clip(arx, lb, ub)
            
            for k in range(lam):
                if not has_time():
                    return best_fitness
                
                # Evaluate the valid (clipped) parameter vector
                val = func(arx_clipped[k])
                counteval += 1
                
                # Update Global Best
                if val < best_fitness:
                    best_fitness = val
                
                # Calculate Penalty
                # Soft quadratic penalty to guide mean towards feasible region
                # without distorting the covariance adaptation logic too much.
                dist_sq = np.sum((arx[k] - arx_clipped[k])**2)
                penalty = 1e8 * dist_sq
                
                fitness[k] = val + penalty
                raw_values.append(val)
            
            # 5. Selection
            # Sort by penalized fitness
            sorted_idx = np.argsort(fitness)
            best_idx = sorted_idx[:mu]
            
            # 6. Update Mean
            xold = xmean.copy()
            # zmean and ymean are weighted averages of selected steps
            zmean = weights @ arz[best_idx]
            ymean = weights @ ary[best_idx]
            
            xmean = xold + sigma * ymean
            
            # 7. Update Evolution Paths
            # Path for step-size (ps)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            
            # Stall check (h_sig) to prevent explosion of C when step size is large
            hsig_check = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (counteval/lam + 1))) / chiN
            hsig = hsig_check < 1.4 + 2/(dim+1)
            
            # Path for covariance (pc)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
            
            # 8. Update Covariance Matrix C
            # Rank-1 update (using evolution path)
            dh_correction = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            
            # Rank-mu update (using weighted selected steps)
            # y_sel shape is (mu, dim)
            y_sel = ary[best_idx]
            rank_mu = y_sel.T @ (y_sel * weights[:, np.newaxis])
            
            C = (1 - c1 - cmu) * C + c1 * (rank_one + dh_correction * C) + cmu * rank_mu
            
            # 9. Update Step-size (Sigma)
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 10. Lazy Eigendecomposition
            # Update B and D only periodically (O(N^3) operation)
            # Heuristic: roughly every 1/(c1+cmu)/N/10 generations
            update_gap = max(1, int(1 / (c1 + cmu) / dim / 10))
            
            if (counteval // lam) - eigeneval > update_gap:
                eigeneval = counteval // lam
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    vals, vecs = np.linalg.eigh(C)
                    # Numerical stability: enforce positive eigenvalues
                    vals = np.maximum(vals, 1e-14)
                    D = np.sqrt(vals)
                    B = vecs
                except np.linalg.LinAlgError:
                    break # Matrix corrupted, force restart
            
            # 11. Termination/Restart Criteria
            # Escape if converged or stuck to allow next IPO restart
            
            # Condition number too high (ill-conditioned)
            if np.max(D) > 1e14 * np.min(D):
                break
                
            # Step size too small (converged to precision limit)
            if sigma < 1e-16:
                break
            
            # Step size exploded (unstable)
            if sigma > 1e10:
                break
                
            # Population flat (fitness stagnation)
            if np.max(raw_values) - np.min(raw_values) < 1e-12:
                break
                
            # Principal axis has no effect (too small relative to sigma)
            if np.max(D) * sigma < 1e-13:
                break
                
    return best_fitness
