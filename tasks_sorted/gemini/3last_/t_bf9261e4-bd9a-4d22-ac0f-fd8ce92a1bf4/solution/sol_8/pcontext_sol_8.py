#The following algorithm implements **BIPOP-CMA-ES (Bi-population Covariance Matrix Adaptation Evolution Strategy)** tailored for a strict time budget.
#
#### Algorithm Description
#
#1.  **Bi-population Strategy (BIPOP)**: The algorithm alternates between two restart regimes to balance global exploration and local exploitation:
#    *   **Large Regime (IPO)**: Starts with a small population and doubles it on every restart ($ \lambda \cdot 2^k $). This ensures that if the problem requires global exploration, the population eventually grows large enough to smooth out local optima.
#    *   **Small Regime (Local)**: Uses a fixed small population with varying initial step sizes. These restarts are cheap and fast, intended to aggressively exploit local basins of attraction.
#    *   **Budget Balancing**: The algorithm tracks the time spent in both regimes. If the "Small" regime has used less time than the "Large" regime, it launches more small restarts. This prevents the algorithm from getting stuck in a single slow, massive-population run while ensuring enough local searches are performed.
#
#2.  **CMA-ES Core**:
#    *   **Covariance Adaptation**: Learns the correlation matrix of variables to handle non-separable (rotated) landscapes.
#    *   **Lazy Eigendecomposition**: The computationally expensive $O(N^3)$ matrix decomposition is performed only periodically (every few generations) to maximize the time available for function evaluations.
#    *   **Boundary Handling**: A "soft" boundary handling method is used. Solutions are clipped to bounds for evaluation, but a quadratic penalty is added to the fitness during the selection phase. This guides the search distribution to stay within valid bounds without artificially truncating the Gaussian distribution.
#
#3.  **Strict Time Management**: The algorithm checks the remaining time budget before every function evaluation, ensuring a safe return of the best solution found so far without exceeding the `max_time` limit.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using BIPOP-CMA-ES (Bi-population CMA-ES) 
    under a strict time limit.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Reserve 2% buffer to ensure safe return/cleanup
    time_limit = timedelta(seconds=max_time * 0.98)

    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Pre-processing ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    scale = ub - lb
    
    # Global Best Tracking
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # BIPOP State
    used_time_large = 0.0
    used_time_small = 0.0
    large_restart_idx = 0
    
    # Base population size: 4 + 3*ln(D)
    lam_base = 4 + int(3 * np.log(dim))

    # --- Core CMA-ES Optimizer ---
    def optimize_cma(lam, sigma_init):
        nonlocal global_best_fitness, global_best_sol
        
        # 1. Initialization
        # Center: Randomly placed within bounds
        xmean = lb + np.random.rand(dim) * scale
        sigma = sigma_init
        
        # Selection weights (logarithmic)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Adaptation constants
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Dynamic State Variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        counteval = 0
        eigeneval = 0
        
        # Stagnation detection
        fitness_history = []
        history_len = 10 + int(30 * dim / lam)
        
        # 2. Generation Loop
        while has_time():
            # Lazy Eigendecomposition (O(N^3))
            # Performed roughly every 1/(c1+cmu)/N/10 generations
            eigen_gap = max(1, int(1.0 / ((c1 + cmu) * dim) / 10.0))
            if counteval // lam > eigeneval + eigen_gap:
                eigeneval = counteval // lam
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T 
                try:
                    vals, vecs = np.linalg.eigh(C)
                    # Numerical stability
                    vals = np.maximum(vals, 1e-14)
                    D = np.sqrt(vals)
                    B = vecs
                except np.linalg.LinAlgError:
                    return # Matrix corrupted, restart
            
            # Condition number check
            if np.max(D) > 1e14 * np.min(D): 
                return 

            # 3. Sampling & Evaluation
            arz = np.random.randn(lam, dim)
            arx = np.zeros((lam, dim))
            penalties = np.zeros(lam)
            fitness_raw = np.zeros(lam)
            
            for k in range(lam):
                if not has_time(): return
                
                # Transform: y = B*D*z, x = m + sigma*y
                y_k = B @ (D * arz[k])
                arx[k] = xmean + sigma * y_k
                
                # Boundary Handling: Clip for evaluation
                x_eval = np.clip(arx[k], lb, ub)
                
                val = func(x_eval)
                counteval += 1
                fitness_raw[k] = val
                
                # Update Global Best immediately
                if val < global_best_fitness:
                    global_best_fitness = val
                    global_best_sol = x_eval.copy()
                
                # Quadratic Penalty for Selection
                # fitness = f(clip(x)) + penalty * ||x - clip(x)||^2
                # This guides the mean towards the feasible region
                dist_sq = np.sum((arx[k] - x_eval)**2)
                penalties[k] = 1e8 * dist_sq
            
            fitness_penalized = fitness_raw + penalties
            
            # 4. Selection and Recombination
            sorted_idx = np.argsort(fitness_penalized)
            best_idx = sorted_idx[:mu]
            
            xold = xmean
            zmean = weights @ arz[best_idx]
            ymean = B @ (D * zmean)
            
            xmean = xold + sigma * ymean
            
            # 5. Step-size Control (Evolution Path)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            hsig = np.linalg.norm(ps) / chiN < 1.4 + 2/(dim+1)
            
            # 6. Covariance Adaptation
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
            
            # Efficient Rank-mu update
            # Reconstruct y vectors for the selected steps
            y_best = (arx[best_idx] - xold) / sigma
            
            dh_correction = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = y_best.T @ (y_best * weights[:, None])
            
            C = (1 - c1 - cmu) * C + c1 * (rank_one + dh_correction * C) + cmu * rank_mu
            
            # Update Sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 7. Restart Criteria
            # Sigma too small or too large
            if sigma < 1e-16 or sigma > 1e10: return
            
            # Fitness Stagnation (Population flat)
            if np.max(fitness_penalized) - np.min(fitness_penalized) < 1e-12:
                return
            
            # Fitness Stagnation (History)
            current_best = fitness_penalized[sorted_idx[0]]
            fitness_history.append(current_best)
            if len(fitness_history) > history_len:
                fitness_history.pop(0)
                if max(fitness_history) - min(fitness_history) < 1e-12:
                    return

    # --- Main BIPOP Scheduler ---
    while has_time():
        # Policy: Balance time budget between Large (exploration) and Small (exploitation)
        if used_time_small < used_time_large:
            # --- Small Regime ---
            # Random restarts with default population size
            t0 = datetime.now()
            
            # Randomized step size for small restarts to cover different basin sizes
            # range: [0.01, 0.5] * scale
            s_factor = 10**np.random.uniform(-2, -0.3) 
            sigma_s = s_factor * np.mean(scale)
            
            optimize_cma(lam_base, sigma_s)
            
            used_time_small += (datetime.now() - t0).total_seconds()
        
        else:
            # --- Large Regime ---
            # IPO: Double the population size every time this regime is triggered
            t0 = datetime.now()
            
            lam_l = int(lam_base * (2 ** large_restart_idx))
            # Large step size for global coverage
            sigma_l = 0.5 * np.mean(scale) 
            
            optimize_cma(lam_l, sigma_l)
            
            used_time_large += (datetime.now() - t0).total_seconds()
            large_restart_idx += 1
            
    return global_best_fitness
