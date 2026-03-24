#Here is an improved algorithm. It implements **IPO-CMA-ES** (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy).
#
#**Why this is an improvement:**
#1.  **Rotation Invariance:** Unlike Differential Evolution (L-SHADE), which relies on axis-aligned crossover, CMA-ES learns the correlation between variables (the covariance matrix). This allows it to solve ill-conditioned problems (long, narrow, diagonal valleys) much more efficiently.
#2.  **Restarts (IPO):** The algorithm detects if it is stuck in a local optimum (stagnation or vanishing step size). If so, it restarts with a **doubled population size**. This drastically increases the probability of finding the global minimum in multimodal landscapes compared to a single run.
#3.  **Cumulative Step-Size Adaptation (CSA):** It adapts the global step size $\sigma$ based on the history of the evolution path, ensuring the algorithm doesn't converge too fast (prematurely) or too slow.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using IPO-CMA-ES (Increasing Population Size 
    Covariance Matrix Adaptation Evolution Strategy).
    
    This algorithm excels at non-separable, ill-conditioned problems where 
    standard Differential Evolution or Random Search fails. It includes a 
    restart mechanism to escape local optima.
    """
    start_time = time.time()
    
    # --- Helper: Boundary Handling with Penalty ---
    # CMA-ES works best in an unconstrained space. We map bounds to penalties.
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    
    # Pre-calculate domain center and scale for initialization
    x_mean_init = 0.5 * (lower_b + upper_b)
    sigma_init = 0.5 * (upper_b - lower_b).min() / 3.0 # Initial step covers ~1/3rd of smallest dim
    
    def get_fitness_with_penalty(x):
        # 1. Clip to bounds for function evaluation
        x_clipped = np.clip(x, lower_b, upper_b)
        
        # 2. Evaluate
        val = func(x_clipped)
        
        # 3. Add quadratic penalty for out-of-bounds components
        # This guides the distribution mean back into the feasible region
        dist = np.abs(x - x_clipped)
        penalty_factor = 1e3  # Heuristic penalty weight
        penalty = penalty_factor * np.sum(dist**2)
        
        return val + penalty, val

    # Global Best Tracking
    best_fitness = float('inf')
    best_solution = None

    # --- Restart Loop (IPO Strategy) ---
    # Start with default population size, double it on every restart
    lam = 4 + int(3 * np.log(dim)) 
    restart_count = 0
    
    while True:
        # Check remaining time before starting a new run
        if time.time() - start_time > max_time:
            break
            
        # --- 1. Initialization for Current Restart ---
        # Strategy Parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights) # Normalize recombination weights
        mueff = 1.0 / np.sum(weights**2)    # Variance-effectiveness of sum w_i x_i
        
        # Adaptation constants
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, ((mueff - 1) / (dim + 1))**0.5 - 1) + cs
        
        # Dynamic State Variables
        # Randomize start point slightly within bounds to avoid deterministic loops
        xmean = lower_b + np.random.rand(dim) * (upper_b - lower_b)
        sigma = 0.3 * (upper_b - lower_b).max() # Initial step size
        
        B = np.eye(dim)       # Eigenvectors of C
        D = np.ones(dim)      # Square root of eigenvalues of C
        C = np.eye(dim)       # Covariance matrix
        invsqrtC = np.eye(dim)
        
        pc = np.zeros(dim)    # Evolution path for C
        ps = np.zeros(dim)    # Evolution path for sigma
        
        chiN = dim**0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim**2)) # Expectation of ||N(0,I)||
        
        # Termination for this run
        eigeneval = 0
        fit_history = []
        
        # --- 2. Generation Loop ---
        while True:
            # Time Check (Strict)
            if time.time() - start_time > max_time:
                return best_fitness

            # A. Sampling
            # Generate lambda offspring
            arz = np.random.randn(lam, dim)
            # arx = xmean + sigma * (B @ (D * arz))
            # Optimized matrix multiplication:
            arx = xmean + sigma * (B @ (D[:, None] * arz.T)).T
            
            # B. Evaluation
            fitnesses = np.zeros(lam)
            true_fitnesses = np.zeros(lam) # Without penalty, for reporting
            
            for i in range(lam):
                # Check time occasionally within population loop for slow functions
                if (i % 10 == 0) and (time.time() - start_time > max_time):
                    return best_fitness
                
                f_pen, f_true = get_fitness_with_penalty(arx[i])
                fitnesses[i] = f_pen
                true_fitnesses[i] = f_true
                
                # Update global best
                if f_true < best_fitness:
                    best_fitness = f_true
                    best_solution = arx[i]

            # C. Selection and Recombination
            # Sort by fitness and compute weighted mean into xmean
            sort_indices = np.argsort(fitnesses)
            arx_best = arx[sort_indices[:mu]]
            arz_best = arz[sort_indices[:mu]]
            
            xmean_old = xmean.copy()
            # Recombination: xmean = sum(weights * best_individuals)
            zmean = np.dot(weights, arz_best)
            xmean = np.dot(weights, arx_best)
            
            # D. Step-Size Control (CSA - Cumulative Step-size Adaptation)
            # Calculate evolution path ps
            # Conjugate evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            norm_ps = np.linalg.norm(ps)
            
            # Update sigma
            # If ps is large (moving in one direction), increase sigma. 
            # If ps is small (random walk/zigzag), decrease sigma.
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            
            # E. Covariance Matrix Adaptation
            # Hsig logic (turn off rank-1 update if sigma is increasing rapidly to prevent explosion)
            hsig = 1 if norm_ps / np.sqrt(1 - (1 - cs)**(2 * (eigeneval/lam + 1))) / chiN < 1.4 + 2 / (dim + 1) else 0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ (D[:, None] * zmean[:, None])).ravel()
            
            # Rank-1 update
            c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc)) # adjust for hsig
            # Rank-mu update
            # Calculate C_mu = sum(w_i * y_i * y_i.T)
            # Using vectorization:
            # y_s = (arx_best - xmean_old) / sigma
            # C_mu = y_s.T @ (weights[:, None] * y_s) -- but using z is more stable
            
            # Reconstruct rank-mu update using arz (more numerically stable)
            rank_mu_upd = np.zeros((dim, dim))
            for k in range(mu):
                v = B @ (D * arz_best[k])
                rank_mu_upd += weights[k] * np.outer(v, v)

            # Update C
            C = (1 - c1a - cmu) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * rank_mu_upd

            # F. Decomposition (Lazy update: O(N^3) operation)
            # Only update Eigendecomposition every few generations
            eigeneval += 1
            if eigeneval % max(1, int(1.0 / (c1 + cmu) / dim / 10)) == 0:
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                
                try:
                    # Eigen decomposition
                    D2, B = np.linalg.eigh(C)
                    
                    # Numerical stability limits
                    D2 = np.maximum(D2, 0) # Eigenvalues must be non-negative
                    D = np.sqrt(D2)
                    
                    # Condition number limit (limit axis ratio)
                    if D.min() < 1e-9:
                        # Add regularization if singular
                        C += 1e-9 * np.eye(dim)
                        D2, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D2, 0))
                except np.linalg.LinAlgError:
                    # Soft restart internal matrices if linear algebra fails
                    B = np.eye(dim)
                    D = np.ones(dim)
                    C = np.eye(dim)

            # --- 3. Convergence / Restart Criteria ---
            
            # History buffer for TolFun
            fit_history.append(fitnesses[sort_indices[0]])
            if len(fit_history) > 10 + int(30 * dim / lam):
                fit_history.pop(0)
            
            # Stop Conditions for current run:
            
            # 1. Target fitness reached (if we knew it, but here we just minimize)
            
            # 2. TolFun: Fitness range in history is too small
            if len(fit_history) > 5 and (max(fit_history) - min(fit_history) < 1e-12):
                break
                
            # 3. TolX: Sigma or axis length too small
            if sigma * np.max(D) < 1e-11:
                break
            
            # 4. Condition number of C too high (ill-conditioned beyond float precision)
            if np.min(D) > 0 and (np.max(D) / np.min(D) > 1e7):
                break
                
            # 5. NoEffectAxis: Standard deviation in any axis is 0
            if np.any(sigma * np.sqrt(np.diag(C)) < 1e-12):
                break

        # --- End of Run cleanup ---
        # Prepare for restart: Increase population size (IPO strategy)
        restart_count += 1
        lam *= 2 # Double population size to explore better
        
        # Prevent explosion of population size causing memory/speed issues
        if lam > 100 * dim:
            lam = 4 + int(3 * np.log(dim)) # Reset if too huge

    return best_fitness
