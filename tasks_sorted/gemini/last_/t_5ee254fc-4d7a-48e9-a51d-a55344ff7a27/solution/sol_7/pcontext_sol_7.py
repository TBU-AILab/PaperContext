#The following Python code implements an **IPO-CMA-ES (Iterative Population Size Covariance Matrix Adaptation Evolution Strategy)**. 
#
#**Improvements over the previous algorithm:**
#1.  **Robustness to Rotation:** CMA-ES intrinsically handles rotated landscapes (variable dependencies) via the covariance matrix update, making it mathematically superior to DE for ill-conditioned valleys (like Rosenbrock or rotated Rastrigin) without ad-hoc linear algebra patches.
#2.  **Scalability:** It includes a switch to **Separable CMA-ES** (diagonal covariance) for high dimensions ($D > 100$) to keep iteration complexity linear $O(D)$ instead of cubic $O(D^3)$.
#3.  **Global Search Capability:** The **IPO** (Restart) mechanism starts with a small population for fast local convergence. If it stalls, it restarts with a doubled population size, increasing the global exploration capability to escape local optima.
#4.  **Bound Handling:** It utilizes a "Repair" strategy where parameter values are clipped for function evaluation, but the internal distribution is allowed to push against the boundaries, ensuring convergence to optimal values located on the edges of the domain.
#5.  **Initialization:** The covariance matrix is initialized proportional to the aspect ratio of the bounds, handling rectangular search spaces (e.g., $x_1 \in [-100, 100], x_2 \in [-1, 1]$) correctly from step zero.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Algorithm: IPO-CMA-ES (Iterative Population Size CMA-ES).
    
    Strategies:
    1. CMA-ES for efficient local search and handling rotated/ill-conditioned functions.
    2. Separable-CMA-ES logic for high dimensions (>100) to maintain speed.
    3. Restart mechanism with increasing population size to handle multimodality.
    4. Aspect-ratio aware initialization for rectangular bounds.
    """
    
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    bounds = np.array(bounds)
    lower_b = bounds[:, 0]
    upper_b = bounds[:, 1]
    
    # Check for High Dimension to switch to Separable CMA-ES (O(N) vs O(N^3))
    # Standard CMA-ES is expensive for N > 100 due to eigendecomposition
    use_separable = dim > 100
    
    # Global State
    best_val = float('inf')
    
    # Helper for timeout
    def get_remaining_seconds():
        return (time_limit - (datetime.now() - start_time)).total_seconds()
    
    # IPO parameters
    # Start with standard population size recommended for CMA-ES
    pop_size = 4 + int(3 * np.log(dim))
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        if get_remaining_seconds() <= 0: return best_val
        
        # 1. Initialization per Restart
        # Center mean in the bounds
        xmean = np.random.uniform(lower_b, upper_b)
        
        # Initialize Step Size (Sigma) and Covariance Scaling (D)
        # Handle rectangular bounds by scaling D to the axis ranges
        range_w = upper_b - lower_b
        mean_range = np.mean(range_w)
        
        # D represents the coordinate-wise scaling (std dev ratios)
        # sigma represents the global step size
        D = range_w / mean_range 
        sigma = 0.3 * mean_range # Start with 30% of the domain coverage
        
        # Setup Covariance Matrix components
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        if use_separable:
            C = D**2 # Vector for diagonal
            B = np.eye(dim) # Implicit Identity
        else:
            C = np.diag(D**2)
            B = np.eye(dim)
            # D is already set above, but in Full CMA, D comes from Eigendecomp of C
            # We keep D consistent with C initially
            
        # Selection Weights (Logarithmic)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # CMA Adaptation Constants
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Stagnation tracking
        history_best = []
        gen = 0
        
        # Lazy Update limit (Full CMA-ES only)
        # Eigendecomposition is O(N^3), do it only every few generations
        lazy_gap = max(1, int(1.0 / (c1 + cmu) / dim / 10.0)) if not use_separable else 1

        # --- Generation Loop ---
        while True:
            if get_remaining_seconds() <= 0: return best_val
            gen += 1
            
            # 2. Sampling & Evaluation
            arx = []
            arz = []
            arf = []
            
            for k in range(pop_size):
                if get_remaining_seconds() <= 0: return best_val
                
                # Sample z ~ N(0, I)
                z = np.random.normal(0, 1, dim)
                arz.append(z)
                
                # Transform z to x
                if use_separable:
                    dx = D * z
                else:
                    dx = np.dot(B, D * z)
                
                x = xmean + sigma * dx
                
                # Bound Handling: Repair
                # Evaluate the clipped version, but let CMA-ES evolve the raw version.
                # This allows the mean to move towards the edge properly.
                x_repair = np.clip(x, lower_b, upper_b)
                
                val = func(x_repair)
                
                # Update Global Best
                if val < best_val:
                    best_val = val
                
                arx.append(x)
                arf.append(val)
                
            arx = np.array(arx)
            arz = np.array(arz)
            arf = np.array(arf)
            
            # 3. Selection
            # Sort by fitness
            sorted_idx = np.argsort(arf)
            z_sel = arz[sorted_idx[:mu]]
            # x_sel = arx[sorted_idx[:mu]] # Not strictly needed for update using z
            
            # 4. Update Mean
            z_w = np.dot(weights, z_sel)
            
            if use_separable:
                move = D * z_w
            else:
                move = np.dot(B, D * z_w)
                
            xmean = xmean + sigma * move
            
            # Safety clip on mean to prevent numerical explosion if it drifts too far
            xmean = np.clip(xmean, lower_b - range_w*0.5, upper_b + range_w*0.5)

            # 5. Update Evolution Paths
            # Conjugate Evolution Path (ps)
            if use_separable:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_w # B=I
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(B, z_w)
            
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen)) / chiN < 1.4 + 2 / (dim + 1)
            
            # Evolution Path (pc)
            if use_separable:
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (D * z_w)
            else:
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * np.dot(B, D * z_w)
                
            # 6. Adapt Covariance Matrix
            if use_separable:
                # Diagonal Update (Sep-CMA-ES)
                # C is a vector here
                old_C = C.copy()
                
                # Rank-1
                h_val = (1 - hsig) * cc * (2 - cc)
                C = (1 - c1 - cmu) * C + c1 * (pc**2 + h_val * C)
                
                # Rank-mu
                # In Sep-CMA, y_i = D_ii * z_i
                y_sel = D * z_sel
                y_var = np.dot(weights, y_sel**2) # Weighted variance per dimension
                C += cmu * y_var
                
                D = np.sqrt(C)
                
            else:
                # Full Matrix Update
                # Rank-1
                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) + (1-hsig) * cc*(2-cc) * C)
                
                # Rank-mu
                # Reconstruct y vectors (B * D * z)
                # Calculating sum(w * y * y.T)
                for i in range(mu):
                    y = np.dot(B, D * z_sel[i])
                    C += cmu * weights[i] * np.outer(y, y)
                
                # Eigen Decomposition (Lazy)
                if gen % lazy_gap == 0:
                    # Enforce symmetry
                    C = np.triu(C) + np.triu(C, 1).T
                    vals, vecs = np.linalg.eigh(C)
                    
                    # Numerical Stability
                    vals = np.maximum(vals, 1e-14)
                    
                    D = np.sqrt(vals)
                    B = vecs
                    
                    # Restart if condition number is too high
                    if np.max(D) / np.min(D) > 1e14:
                        break
            
            # 7. Adapt Step Size (Sigma)
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # --- Stopping Criteria for Restart ---
            
            # 1. Sigma too small or too large
            if sigma < 1e-14 * mean_range or sigma > 1e14:
                break
                
            # 2. Fitness Stagnation
            current_best = arf[sorted_idx[0]]
            history_best.append(current_best)
            
            # Keep history short
            if len(history_best) > 20:
                history_best.pop(0)
                
            if len(history_best) >= 20:
                if np.max(history_best) - np.min(history_best) < 1e-12:
                    break
                    
        # --- End of Generation Loop ---
        # Double population size for the next restart (IPO strategy)
        # This allows exploring more complex multimodal landscapes
        pop_size *= 2
        
        # Safety cap to prevent population from becoming too slow
        if pop_size > 100 * dim:
            pop_size = 4 + int(3 * np.log(dim)) # Reset

    return best_val
