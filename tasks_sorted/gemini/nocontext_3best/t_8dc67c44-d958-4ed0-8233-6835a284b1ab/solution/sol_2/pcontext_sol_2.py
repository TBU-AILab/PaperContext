#The algorithm below implements **IPOP-CMA-ES** (Restart Covariance Matrix Adaptation Evolution Strategy with Increasing Population). 
#
#**Why this is an improvement:**
#1.  **Rotation Invariance**: Unlike Differential Evolution (used in previous attempts), CMA-ES learns the correlation between variables (the covariance matrix). It can solve ill-conditioned and rotated problems where variables are highly dependent on each other.
#2.  **Step-Size Adaptation**: It optimally adapts the search step size ($\sigma$) using Evolution Paths, allowing it to converge rapidly into narrow basins.
#3.  **IPOP (Increasing Population) Restart**: If the local search stagnates or converges to a local optimum, the algorithm restarts with a doubled population size. This allows it to transition from local exploitation to global exploration, effectively handling multimodal landscapes.
#
import numpy as np
from datetime import datetime, timedelta
import math

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPOP-CMA-ES (Restart CMA-ES with Increasing Population).
    CMA-ES is state-of-the-art for continuous, non-separable, ill-conditioned optimization.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Bounds Processing ---
    bounds_np = np.array(bounds)
    lower_bounds = bounds_np[:, 0]
    upper_bounds = bounds_np[:, 1]
    bound_range = upper_bounds - lower_bounds
    
    # Global best tracking
    best_fitness = float('inf')
    best_solution = None

    # Function wrapper to handle best tracking and time
    def evaluate_wrapper(x):
        # Clip to bounds for evaluation
        x_clipped = np.clip(x, lower_bounds, upper_bounds)
        val = func(x_clipped)
        nonlocal best_fitness, best_solution
        if val < best_fitness:
            best_fitness = val
            best_solution = x_clipped.copy()
        return val

    # Helper: Check time
    def time_is_up():
        return datetime.now() >= end_time

    # Perform initial evaluation of the center
    x_start = lower_bounds + 0.5 * bound_range
    evaluate_wrapper(x_start)
    if time_is_up(): return best_fitness

    # --- IPOP-CMA-ES Main Loop ---
    # Start with default population size
    lam_start = 4 + int(3 * np.log(dim))
    restart_count = 0
    
    while not time_is_up():
        
        # 1. Restart Configuration
        # Increase population size for each restart (IPOP strategy)
        lam = int(lam_start * (2 ** restart_count))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # State Initialization
        # Initialize mean uniformly within bounds
        mean = lower_bounds + np.random.rand(dim) * bound_range
        
        # Step size (sigma): initialize relative to domain size
        sigma = 0.3 * (upper_bounds - lower_bounds).max()
        
        # Covariance paths and matrix
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim) # B * D^2 * B.T
        
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Loop for this restart
        generation = 0
        while not time_is_up():
            generation += 1
            
            # 2. Sampling (Ask)
            offspring = []
            offspring_raw = [] # unclipped for CMA update
            
            # Generate lambda offspring
            # x_i = m + sigma * (B @ (D * z_i))
            try:
                # Eigendecomposition is expensive; only do it if C changed significantly or periodically
                # For this implementation, we do it every gen but optimized
                if generation % 1 == 0:
                    # Enforce symmetry
                    C = np.triu(C) + np.triu(C, 1).T 
                    vals, vecs = np.linalg.eigh(C)
                    # Numerical stability
                    vals = np.maximum(vals, 1e-16) 
                    D = np.sqrt(vals)
                    B = vecs
            except:
                # Fallback in case of numerical error
                B = np.eye(dim)
                D = np.ones(dim)
                C = np.eye(dim)

            # Generate random noise vectors
            Z = np.random.randn(lam, dim)
            
            fitnesses = np.zeros(lam)
            
            for i in range(lam):
                if time_is_up(): return best_fitness
                
                # Mutation
                diff = B @ (D * Z[i])
                x_raw = mean + sigma * diff
                
                # Check bounds (Clip)
                # We use the fitness of the *clipped* vector, 
                # but mathematically CMA usually prefers penalties. 
                # Clipping is more robust for general black-box functions in limited time.
                val = evaluate_wrapper(x_raw)
                
                offspring_raw.append(diff) # Store displacement (not x)
                fitnesses[i] = val

            # 3. Selection (Tell)
            # Sort by fitness
            sort_indices = np.argsort(fitnesses)
            
            # Compute weighted mean of top mu vectors
            old_mean = mean.copy()
            
            # Recombination of the displacements
            # mean = old_mean + sigma * (sum(weights * mutation_steps))
            z_mean = np.zeros(dim) # This is sum(w * z)
            diff_mean = np.zeros(dim) # This is sum(w * B*D*z)
            
            for i in range(mu):
                idx = sort_indices[i]
                z_mean += weights[i] * Z[idx]
                diff_mean += weights[i] * offspring_raw[idx]
            
            mean = old_mean + sigma * diff_mean
            
            # 4. Step-size Control (Path for Sigma)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrt(C) * (mean - old_mean)/sigma
            # Note: B @ z_mean is roughly invsqrt(C) * (mean - old_mean)/sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_mean)
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            
            # 5. Covariance Adaptation (Path for C)
            # hsig check (heavyside function) to stall update if ps is large
            hsig = 0
            if norm_ps / np.sqrt(1 - (1 - cs)**(2 * generation)) / chiN < 1.4 + 2 / (dim + 1):
                hsig = 1
                
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff_mean
            
            # Update C
            # Rank-1 update (using pc) + Rank-mu update (using weighted mutations)
            
            # Create rank-mu matrix
            # Vectors for rank-mu are offspring_raw[idx] / sigma ?? 
            # Actually standard CMA uses: B*D*z.
            artmp = np.zeros((dim, dim))
            for i in range(mu):
                idx = sort_indices[i]
                # The mutation step d = B*D*z
                d = offspring_raw[idx] 
                # Outer product
                artmp += weights[i] * np.outer(d, d)
            
            c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
            C = (1 - c1a - cmu) * C 
            C += c1 * np.outer(pc, pc) 
            C += cmu * artmp

            # 6. Stop/Restart Criteria
            
            # Stop if time is up
            if time_is_up(): return best_fitness
            
            # Stop if sigma is too small (convergence)
            if sigma < 1e-12 * bound_range.max():
                break
                
            # Stop if condition number is too high
            if np.max(D) / np.min(D) > 1e7:
                break
            
            # Stop if no improvement in fitness for some time (stagnation)
            # (Simplified check: if range of fitness in population is tiny)
            if np.max(fitnesses) - np.min(fitnesses) < 1e-12:
                break
            
            # Safety clamp for sigma
            if sigma > 1e4 * bound_range.max():
                break
        
        # Prepare for next restart
        restart_count += 1

    return best_fitness
