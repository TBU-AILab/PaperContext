#To improve upon the provided algorithms, I propose using the **Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with Restarts**. 
#
#### Rationale for Improvement
#The previous best result was achieved by **jDE**, a Differential Evolution variant. DE is excellent, but it relies on coordinate-aligned operations (crossover) which can be inefficient if the objective function has strong dependencies between variables (rotated valleys). 
#
#**CMA-ES** is considered the state-of-the-art for continuous black-box optimization in lower to medium dimensions ($N < 100$). 
#1.  **Rotation Invariance**: It learns the correlation between variables (the covariance matrix), allowing it to traverse diagonal valleys effectively where DE might zigzag.
#2.  **Step-Size Control**: It adapts the global step size ($\sigma$) automatically, allowing for fast convergence when wide exploration is done.
#3.  **Restarts**: A restart strategy (IPO-CMA-ES style) is implemented. If the algorithm converges to a local optimum (variance becomes too small) or stagnates, it restarts with a randomized mean and a larger population size to explore different basins of attraction.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    Includes automated restarts with increasing population size (IPO-CMA-ES regime).
    """
    start_time = datetime.now()
    # Use a safety buffer to ensure we return before the hard limit
    time_limit = timedelta(seconds=max_time * 0.98)

    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global best tracking
    best_fitness = float('inf')
    best_solution = None

    # IPO-CMA-ES Restart Loop
    restart_count = 0
    
    while True:
        # Check overall time budget
        if datetime.now() - start_time >= time_limit:
            return best_fitness

        # --- Initialization per Restart ---
        # 1. Strategy Parameters
        # Population size (lambda): start small, increase on restarts
        # Standard CMA-ES default: 4 + 3*log(N)
        pop_size = int(4 + 3 * np.log(dim))
        # Increase pop_size for subsequent restarts (IPO strategy) to broaden search
        pop_size = int(pop_size * (2 ** restart_count))
        
        # Parent number (mu)
        mu = pop_size // 2
        
        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights) # Normalize
        mueff = 1 / np.sum(weights**2)      # Variance effective selection mass
        
        # 2. Adaptation Parameters
        # Time constant for cumulation for C
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        # Time constant for cumulation for sigma control
        cs = (mueff + 2) / (dim + mueff + 5)
        # Learning rate for rank-1 update of C
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        # Learning rate for rank-mu update of C
        alpha_mu = 2
        cmu = min(1 - c1, alpha_mu * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + alpha_mu * mueff / 2))
        # Damping for sigma
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # 3. Dynamic State Initialization
        # Start in the center of bounds or random valid position
        xmean = np.random.uniform(lb, ub)
        
        # Step size (sigma): initialized relative to domain scope
        # Use 0.3 * range as a heuristic for initial coverage
        sigma = 0.3 * np.mean(ub - lb)
        
        # Evolution Paths and Covariance
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim) # Covariance matrix
        
        # Expected length of N(0,I) for sigma adaptation
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Loop variables
        gen = 0
        
        # --- Generation Loop ---
        while True:
            # Time check inside generation
            if datetime.now() - start_time >= time_limit:
                return best_fitness

            # 1. Sampling
            # Generate lambda offspring: x_i = m + sigma * B * D * z_i
            try:
                # Decompose C if needed (O(N^3), doing it every step for simplicity/stability in low dim)
                # For very high dims, this should be lazy, but for N<100 it's fast enough.
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T 
                D_squared, B = np.linalg.eigh(C)
                
                # Numerical stability: ensure positive eigenvalues
                D_squared = np.maximum(D_squared, 1e-18)
                D = np.sqrt(D_squared)
                
                # Setup transformation matrix M = B * D
                # We can just compute B * D * z vector-wise
            except np.linalg.LinAlgError:
                # If decomposition fails (numerical issues), break to restart
                break

            # z vectors ~ N(0, I)
            arz = np.random.randn(pop_size, dim)
            
            # y vectors ~ N(0, C) -> y = B * D * z
            # Optimization: Broadcast multiplication
            ary = (B @ (D[:, None] * arz.T)).T
            
            # x vectors (Candidate solutions)
            arx = xmean + sigma * ary
            
            # 2. Evaluation & Bound Handling
            fitness = np.zeros(pop_size)
            
            # We use a repair mechanism for evaluation (clipping)
            # but keep the original arx for the distribution update (CMA-ES standard)
            # Penalization could be added, but simple clipping is robust for black-box.
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_fitness

                # Clip to bounds
                x_eval = np.clip(arx[i], lb, ub)
                
                try:
                    val = func(x_eval)
                except Exception:
                    val = float('inf')
                
                # Add penalty for distance from bounds to discourage "pushing" against walls excessively
                # Penalty: 1e6 * ||x_raw - x_clipped||^2
                dist_sq = np.sum((arx[i] - x_eval)**2)
                # Scale penalty by value magnitude if possible, else fixed high constant
                # Using a mild penalty helps guide the mean back into bounds
                penalty = 1e6 * dist_sq + (1e9 if np.isnan(val) else 0)
                
                fitness[i] = val + penalty
                
                # Update global best with the VALID (clipped) solution
                if val < best_fitness:
                    best_fitness = val
                    best_solution = x_eval.copy()

            # 3. Selection and Recombination
            # Sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(fitness)
            xold = xmean.copy()
            
            # Recombination of the mu best
            best_z = arz[arindex[:mu]]
            zmean = np.dot(weights, best_z)
            
            # Update mean
            # xmean = xold + sigma * (B @ D @ zmean) 
            # Equivalently via weighted sum of selected y
            best_y = ary[arindex[:mu]]
            ymean = np.dot(weights, best_y)
            xmean = xold + sigma * ymean
            
            # 4. Step-size control (Evolution Path p_s)
            # C^(-1/2) = B D^(-1) B^T
            # We need invsqrtC * ymean. 
            # Since ymean = B * D * zmean, invsqrtC * ymean = B * zmean
            invsqrtC_ymean = B @ zmean
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC_ymean
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN < 1.4 + 2 / (dim + 1)
            
            # 5. Covariance Matrix Adaptation (Evolution Path p_c)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
            
            # 6. Update C
            # Rank-1 update
            artmp = (1 / sigma) * (arx[arindex[:mu]] - xold)
            
            # Adjust weights for Rank-mu update
            # (Standard CMA-ES update formula)
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            
            rank1 = np.outer(pc, pc)
            
            # Calculate rank-mu update efficiently
            # We need sum(w_i * y_i * y_i^T)
            # optimization: use vectorized operations
            # y_i are in 'best_y'
            # weighted covariance:
            rank_mu = np.dot(best_y.T * weights, best_y)

            C = (1 - c1 - cmu) * C \
                + c1 * (rank1 + delta_hsig * C) \
                + cmu * rank_mu
            
            # 7. Update sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            gen += 1
            
            # --- Stop/Restart Conditions ---
            # 1. Sigma too small (converged to precision limit)
            if sigma < 1e-10:
                break
                
            # 2. Condition number of C exceeds limit (collapsed or ill-conditioned)
            if np.max(D) > 1e14 * np.min(D):
                break
                
            # 3. No improvement (stagnation) over many generations
            # (Simplified check: if sigma explodes or vanishes significantly)
            if sigma > 1e10: # Unstable divergence
                break
            
            # 4. Standard deviation in all coordinates smaller than tolerance
            if np.all(sigma * np.sqrt(np.diag(C)) < 1e-12):
                break

        # End of Generation Loop -> Trigger Restart
        restart_count += 1
        
    return best_fitness
