#The following is an improved algorithm implementing **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** with a **Restart Mechanism**.
#
##### Improvements Explanation:
#1.  **CMA-ES Algorithm**: Unlike Differential Evolution, CMA-ES adapts the full covariance matrix of the mutation distribution. This allows it to learn the shape of the objective function's landscape (e.g., valleys, ridges) and handle rotation and scaling (ill-conditioned problems) extremely efficiently. It is widely considered the state-of-the-art for continuous optimization in dimensions up to ~100.
#2.  **Restart Mechanism**: The algorithm detects convergence (when step size `sigma` becomes too small or condition number too high) and restarts the search from a random position. This is crucial for multimodal functions to escape local optima.
#3.  **Soft Boundary Handling**: Uses a penalty method for the internal evolution (guiding the mean into bounds) while strictly clipping input vectors for function evaluation. This prevents the algorithm from getting stuck at boundaries while ensuring valid inputs are passed to the function.
#4.  **Robust Time Management**: Checks `max_time` strictly before expensive operations (like function evaluations and matrix decompositions) to ensure the best result is returned within the limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    with a Restart mechanism for global search capabilities within a time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Problem Setup ---
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    bound_range = upper - lower
    
    # Track global best
    best_fitness = float('inf')
    
    # --- CMA-ES Parameters (Standard Configuration) ---
    # Population size (lambda) and parent number (mu)
    lambda_ = 4 + int(3 * np.log(dim))
    mu = int(lambda_ / 2)
    
    # Selection weights (logarithmic)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2) # Variance-effective selection mass
    
    # Adaptation parameters
    cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
    
    # Expected length of normal vector
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check time before starting a new run
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        # Initialize State Variables
        # Start from a random position
        xmean = np.random.uniform(lower, upper)
        # Initial step size: 30% of the domain range
        sigma = 0.3 * np.max(bound_range)
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Covariance Matrix components
        B = np.eye(dim)      # Eigenvectors
        D = np.ones(dim)     # Square root of Eigenvalues
        C = np.eye(dim)      # Covariance Matrix
        
        generation = 0
        
        # --- Generation Loop ---
        while True:
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            # 1. Sampling and Evaluation
            arx = [] # Stores dicts with 'z', 'fit'
            
            for k in range(lambda_):
                if (datetime.now() - start_time) >= time_limit:
                    return best_fitness
                
                # Sample z ~ N(0, I)
                z = np.random.standard_normal(dim)
                
                # Mutation: x = m + sigma * B * D * z
                y = np.dot(B, D * z)
                x_mut = xmean + sigma * y
                
                # Boundary Handling
                # strict clip for evaluation
                x_eval = np.clip(x_mut, lower, upper)
                
                # Evaluate
                val = func(x_eval)
                
                # Update Global Best
                if val < best_fitness:
                    best_fitness = val
                
                # Calculate Fitness with Penalty
                # Penalize solutions that drift out of bounds to guide the mean back
                # Penalty is proportional to squared distance from boundary
                dist = np.linalg.norm(x_mut - x_eval)
                penalty = 1e5 * dist**2 if dist > 0 else 0
                
                arx.append({'z': z, 'fit': val + penalty})
            
            # 2. Sort by fitness
            arx.sort(key=lambda x: x['fit'])
            
            # 3. Selection and Mean Update
            # Select best mu individuals
            z_sel = np.array([ind['z'] for ind in arx[:mu]])
            
            # Compute weighted mean of selected steps
            zmean = np.dot(weights, z_sel)
            
            # Update mean: xmean = xmean + sigma * B * D * zmean
            y_mean = np.dot(B, D * zmean)
            xmean = xmean + sigma * y_mean
            
            # 4. Update Evolution Paths
            # ps: Evolution path for sigma (conjugate evolution path)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(B, zmean)
            
            # hsig: Threshold to stall pc update if norm(ps) is too large (outlier)
            hsig_val = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN
            hsig = hsig_val < 1.4 + 2/(dim + 1)
            
            # pc: Evolution path for C
            pc = (1 - cc) * pc + (1 if hsig else 0) * np.sqrt(cc * (2 - cc) * mueff) * y_mean
            
            # 5. Update Covariance Matrix C
            # Rank-1 update
            pc_col = pc[:, np.newaxis]
            C = (1 - c1 - cmu) * C + c1 * (np.dot(pc_col, pc_col.T) + (1 - (1 if hsig else 0)) * cc * (2 - cc) * C)
            
            # Rank-mu update
            # Reconstruct y vectors for selected individuals: y_i = B * D * z_i
            # We compute weighted covariance of z_sel and project it
            Z_matrix = np.zeros((dim, dim))
            for i in range(mu):
                z_col = z_sel[i][:, np.newaxis]
                Z_matrix += weights[i] * np.dot(z_col, z_col.T)
            
            BD = B * D[None, :] # Scale columns of B by D
            C += cmu * np.dot(BD, np.dot(Z_matrix, BD.T))
            
            # 6. Update Step Size (Sigma)
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 7. Decomposition (Eigenvalue Decomposition)
            # Perform periodically to update B and D (O(dim^3))
            # Heuristic: every 1/(c1+cmu)/dim/10 generations or simpler
            decomp_interval = max(1, int(1 / (c1 + cmu) / dim / 10)) if dim > 10 else 1
            
            if generation % decomp_interval == 0:
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    vals, vecs = np.linalg.eigh(C)
                    # Numerical stability: enforce positive eigenvalues
                    vals = np.maximum(vals, 1e-16)
                    D = np.sqrt(vals)
                    B = vecs
                except np.linalg.LinAlgError:
                    # If decomposition fails, restart the run
                    break
            
            generation += 1
            
            # 8. Termination / Restart Criteria
            # Stop if sigma is too small (converged to precision limit)
            if sigma < 1e-11:
                break
            # Stop if condition number is too high (ill-conditioned beyond repair)
            if np.max(D) > 1e14 * np.min(D):
                break
            # Stop if sigma explodes (instability)
            if sigma > 1e10:
                break
            # Stop if time is almost up (checked at start of loop, but good to be safe)

    return best_fitness
