#Here is a fully operational, self-contained implementation of **IPOP-CMA-ES (Restart Covariance Matrix Adaptation Evolution Strategy with Increasing Population Size)**.
#
#### Algorithm Description
#This algorithm is fundamentally different from the previous Differential Evolution (DE) submission. The previous result (~2.5) suggests the problem landscape is likely **ill-conditioned** (narrow valleys) or **rotated**, which DE struggles with, or **multimodal**, requiring restarts.
#
#**CMA-ES** is the gold standard for continuous black-box optimization. It learns the covariance matrix of the search distribution to adapt to the landscape geometry.
#1.  **Adaptation**: It automatically scales the search steps and rotation (via the Covariance Matrix $C$) to navigate narrow valleys efficiently.
#2.  **Restart Strategy (IPOP)**: If the algorithm converges (step size $\sigma$ becomes too small) or stagnates, it restarts. Crucially, **IPOP** (Increasing POPulation) doubles the population size with each restart. This transitions the search from "local exploitation" (small pop) to "global exploration" (large pop), solving the multimodal trap.
#3.  **Boundary Handling**: It implements a quadratic penalty method to guide the internal Gaussian distribution to stay within bounds, while evaluating the valid clipped solution.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPOP-CMA-ES (Restart CMA-ES with Increasing Population).
    
    Strategies:
    1. Covariance Matrix Adaptation (CMA) to handle ill-conditioned/rotated problems.
    2. Cumulative Step-size Adaptation (CSA) for dynamic step-size control.
    3. IPOP (Increasing Population): Restarts with doubled population size to escape local optima.
    4. Quadratic Boundary Penalty: Softly guides the distribution back to feasible space.
    """
    
    # --- Configuration ---
    start_time = time.time()
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # Global best tracking
    best_fitness = float('inf')
    best_solution = None

    def get_remaining_time():
        return max_time - (time.time() - start_time)

    # --- IPOP Restart Loop ---
    # We start with a small population and double it on every restart.
    restart_count = 0
    pop_size_multiplier = 1
    
    # Base population size (will be scaled)
    # 4 + 3 * log(N) is the standard heuristic for CMA-ES
    base_lambda = 4 + int(3 * np.log(dim))
    
    while get_remaining_time() > 0.05: # Keep restarting until time runs out
        
        # --- Initialization for this Epoch ---
        # 1. Selection parameters
        lambda_ = base_lambda * pop_size_multiplier # Offspring number
        mu = lambda_ // 2                           # Parent number
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)) # Recombination weights
        weights = weights / np.sum(weights)         # Normalize
        mueff = 1.0 / np.sum(weights**2)            # Variance-effectiveness
        
        # 2. Adaptation parameters
        # Time constant for cumulation for C
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        # Time constant for cumulation for sigma control
        cs = (mueff + 2) / (dim + mueff + 5)
        # Learning rate for rank-1 update of C
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        # Learning rate for rank-mu update of C
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        # Damping for sigma
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # 3. Dynamic State Variables
        # Start in the middle of bounds with some random noise
        xmean = lb + (ub - lb) * np.random.rand(dim)
        
        # Step size (sigma) initialization: relative to domain size
        sigma = 0.5 * (ub - lb).max() / 5.0 
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Covariance Matrix (B @ D @ B.T = C)
        B = np.eye(dim)
        D = np.ones(dim)
        C = B @ np.diag(D**2) @ B.T
        
        # Tracking for updates
        eigen_eval = 0
        chiN = dim**0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # History for stagnation check
        history_best = []
        
        # --- Generation Loop ---
        generation = 0
        stop_epoch = False
        
        while not stop_epoch:
            # Check time strictly
            if get_remaining_time() < 0.02:
                return best_fitness

            # 1. Sample Offspring
            # arx: raw generated points (population)
            # arx_valid: points clipped to bounds for evaluation
            arx = np.zeros((lambda_, dim))
            arx_valid = np.zeros((lambda_, dim))
            fitness_raw = np.zeros(lambda_) # Pure function value
            fitness_pen = np.zeros(lambda_) # Penalized fitness for CMA internal logic
            
            # Sampling Loop
            for k in range(lambda_):
                # Check time occasionally within generation for large populations
                if k % 100 == 0 and get_remaining_time() < 0.01:
                    return best_fitness

                # z ~ N(0, I)
                z = np.random.randn(dim) 
                # x ~ N(m, sigma^2 C)  => x = m + sigma * (B * D * z)
                arx[k] = xmean + sigma * (B @ (D * z))
                
                # Clip to bounds
                arx_valid[k] = np.clip(arx[k], lb, ub)
                
                # Evaluate
                val = func(arx_valid[k])
                fitness_raw[k] = val
                
                # Update Global Best
                if val < best_fitness:
                    best_fitness = val
                    best_solution = arx_valid[k].copy()
                
                # Boundary Penalty Calculation (for internal CMA guidance)
                # If x is out of bounds, add penalty ||x - clipped_x||^2
                # This creates a quadratic bowl pointing back to the valid region
                dist = np.linalg.norm(arx[k] - arx_valid[k])
                penalty = 1e3 * dist**2 + 1e3 * dist # Heuristic penalty
                fitness_pen[k] = val + penalty

            # 2. Sort by penalized fitness
            arindex = np.argsort(fitness_pen)
            xold = xmean.copy()
            
            # 3. Update Mean
            # Recombination of top 'mu' offspring
            xmean = np.dot(weights, arx[arindex[:mu]])
            
            # 4. Update Evolution Paths
            # zmean (step) = (xmean - xold) / sigma
            # But we need B^-1 D^-1 zmean roughly. 
            # A more numerically stable way involves the z-vectors from generation.
            # Reconstruct z-vectors for the selected parents:
            z_selected = (arx[arindex[:mu]] - xold) / sigma
            # C^-1/2 * z_selected is approximately the sampled z. 
            # Actually, standard CMA update uses:
            invsqrtC = B @ np.diag(1.0/D) @ B.T
            y = (xmean - xold) / sigma
            
            # Update ps (Conjugate evolution path for sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y)
            
            # Update pc (Evolution path for C)
            # hsig: Heaviside switch to stall update if ||ps|| is too large (avoids expanding C too fast)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y

            # 5. Update Covariance Matrix C
            # Rank-1 update (using pc) + Rank-mu update (using selected steps)
            # Create weighted matrix of steps
            artmp = (arx[arindex[:mu]] - xold) / sigma
            
            C = (1 - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * (artmp.T @ np.diag(weights) @ artmp)

            # 6. Update Step Size sigma (CSA)
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            # 7. Decomposition (Eigen)
            # We don't do this every generation as it's O(N^3).
            # Do it every 1/(c1+cmu)/N/10 generations or if needed
            if generation - eigen_eval > 1.0 / (c1 + cmu) / dim / 10.0:
                e_eval_start = time.time()
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T 
                
                try:
                    # Eigendecomposition
                    vals, vecs = np.linalg.eigh(C)
                    
                    # Numerical stability limits
                    vals = np.maximum(vals, 1e-16) # Avoid zero/negative variance
                    D = np.sqrt(vals)
                    B = vecs
                    eigen_eval = generation
                except np.linalg.LinAlgError:
                    # If matrix is singular, soft restart internal
                    break
                    
                # Subtract eigen time from budget check to avoid timeouts during heavy calc
                if get_remaining_time() < 0.05:
                    return best_fitness

            # --- Convergence / Restart Criteria ---
            
            # Track best penalized fitness in this epoch
            current_best_pen = fitness_pen[arindex[0]]
            history_best.append(current_best_pen)
            if len(history_best) > 10 + int(30 * dim / lambda_):
                history_best.pop(0)

            # 1. Stop if sigma is too small (converged)
            if sigma < 1e-12 * (ub.max() - lb.min()):
                stop_epoch = True
                
            # 2. Stop if condition number is huge
            if np.max(D) / np.min(D) > 1e14:
                stop_epoch = True
                
            # 3. Stop if stagnant (TolFun equivalent)
            if len(history_best) > 10 and np.max(history_best) - np.min(history_best) < 1e-12:
                stop_epoch = True
            
            # 4. Stop if sigma explodes (instability)
            if sigma > 1e5 * (ub.max() - lb.min()):
                stop_epoch = True
                
            generation += 1
            
        # --- End of Epoch ---
        restart_count += 1
        pop_size_multiplier *= 2 # IPOP: Double population for next restart
        
        # Safety break if population gets absurdly large to avoid MemoryError or O(N^2) slowdown
        if lambda_ * 2 > 2000:
            pop_size_multiplier = 1 # Reset to small pop but new random start
            
    return best_fitness
