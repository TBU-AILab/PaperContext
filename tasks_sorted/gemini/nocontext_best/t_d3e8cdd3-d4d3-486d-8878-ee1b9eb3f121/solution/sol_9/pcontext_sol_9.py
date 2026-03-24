#Here is a fully operational, self-contained implementation using **IPO-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy)** combined with **MTS-LS1 (Multiple Trajectory Search - Local Search 1)** for final polishing.
#
#### Algorithm Explanation
#1.  **IPO-CMA-ES**: This is the core global search engine. It is robust for ill-conditioned, non-separable, and continuous optimization problems. The "IPO" part means that if the algorithm stagnates or converges to a local optimum, it restarts with a larger population size (doubling it), allowing for a more thorough exploration of the search space in subsequent phases.
#2.  **MTS-LS1 Polishing**: In the final remaining time (or when the global search is exhausted), the algorithm switches to a coordinate-based local search method (MTS-LS1). This method is highly effective at refining the best-found solution to high precision by dynamically adjusting search steps for each dimension.
#3.  **Boundary Handling**: A quadratic penalty method is used during the evolution to guide the search towards feasible regions, while "repairing" (clipping) solutions ensures only valid inputs are passed to the function.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPO-CMA-ES (Restart CMA-ES with increasing population)
    and MTS-LS1 for final polishing.
    """
    start_time = time.time()
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracking
    best_x = np.random.uniform(lb, ub)
    best_f = float('inf')

    # Time Management Helper
    def get_remaining_time():
        return max_time - (time.time() - start_time)

    # ---------------------------------------------------------
    # 1. MTS-LS1 Local Search (Polishing)
    # ---------------------------------------------------------
    def mts_ls1(center_x, center_f, budget):
        ls_start = time.time()
        curr_x = np.copy(center_x)
        curr_f = center_f
        
        # Initial search range relative to domain size
        search_range = (ub - lb) * 0.4
        min_search_range = 1e-15
        
        improved = True
        dims_indices = np.arange(dim)
        
        while (time.time() - ls_start) < budget:
            # If step size is too small and no improvement, break/restart LS
            if not improved and np.max(search_range) < min_search_range:
                break
                
            improved = False
            np.random.shuffle(dims_indices) # Randomize dimension order
            
            for i in dims_indices:
                if (time.time() - ls_start) >= budget: break
                
                original_val = curr_x[i]
                
                # Search Direction 1: Negative
                curr_x[i] = np.clip(original_val - search_range[i], lb[i], ub[i])
                val = func(curr_x)
                
                if val < curr_f:
                    curr_f = val
                    improved = True
                else:
                    # Search Direction 2: Positive (0.5 step for finer granularity)
                    curr_x[i] = np.clip(original_val + 0.5 * search_range[i], lb[i], ub[i])
                    val = func(curr_x)
                    
                    if val < curr_f:
                        curr_f = val
                        improved = True
                    else:
                        curr_x[i] = original_val # Revert
            
            if not improved:
                # Reduce search range (Refine)
                search_range /= 2.0
            else:
                # If improved, we maintain intensity. 
                # MTS-LS1 logic often keeps the step size upon success to exploit the gradient.
                pass
                
        return curr_x, curr_f

    # ---------------------------------------------------------
    # 2. IPO-CMA-ES Main Loop
    # ---------------------------------------------------------
    restart_count = 0
    
    # Reserve small portion of time for final local search polish
    # At least 0.2s or 5% of max_time
    final_polish_buffer = max(0.2, max_time * 0.05)
    
    while get_remaining_time() > final_polish_buffer:
        
        # --- Initialization per Restart ---
        # Population size: 4 + 3*ln(N), doubling every restart (IPO strategy)
        pop_size = int(4 + 3 * np.log(dim)) * (2 ** restart_count)
        # Cap population to maintain speed in high dimensions
        if pop_size > 200 * dim: pop_size = 200 * dim
        
        mu = pop_size // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation constants (CMA-ES standard parameters)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Initial Mean
        if restart_count == 0:
            mean = np.random.uniform(lb, ub)
        else:
            # Soft restart: Probability to restart near best or random
            if np.random.rand() < 0.5 and not np.isinf(best_f):
                mean = best_x + np.random.normal(0, 0.05, dim) * (ub - lb)
                mean = np.clip(mean, lb, ub)
            else:
                mean = np.random.uniform(lb, ub)

        # Initial Step Size
        sigma = 0.5 * np.max(ub - lb)
        
        # Covariance Matrix state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Loop variables
        gen = 0
        stagnation_counter = 0
        last_best_fit_restart = float('inf')
        stop_restart = False
        
        # --- Generation Loop ---
        while not stop_restart and get_remaining_time() > final_polish_buffer:
            
            # 1. Eigendecomposition (update B and D)
            # Done lazily (every few generations) to save time
            if gen % max(1, int(1 / (10 * c1 * dim))) == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T # Enforce symmetry
                    D_sq, B = np.linalg.eigh(C)
                    # Numerical stability for D
                    D = np.sqrt(np.where(D_sq < 1e-20, 1e-20, D_sq))
                except np.linalg.LinAlgError:
                    stop_restart = True
                    break

            # 2. Sampling
            # Generate offspring: x = m + sigma * B * D * z
            z = np.random.normal(0, 1, (pop_size, dim))
            y = z @ np.diag(D) @ B.T # Shape (pop_size, dim)
            x = mean + sigma * y
            
            # 3. Evaluation & Bound Handling
            fitness = np.zeros(pop_size)
            penalty_factor = 1e8
            
            for i in range(pop_size):
                if get_remaining_time() <= 0:
                    stop_restart = True
                    break
                
                # Repair solution to feasible domain
                x_repaired = np.clip(x[i], lb, ub)
                
                # Evaluate
                val = func(x_repaired)
                
                # Calculate penalty (quadratic distance from bounds)
                # This guides the distribution back to the feasible region
                dist_sq = np.sum((x[i] - x_repaired)**2)
                fitness[i] = val + penalty_factor * dist_sq
                
                # Update Global Best
                if val < best_f:
                    best_f = val
                    best_x = np.copy(x_repaired)
            
            if stop_restart: break
            
            # 4. Selection and Recombination
            sort_idx = np.argsort(fitness)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            z_sorted = z[sort_idx]
            
            # Update Mean
            mean_old = mean
            mean = mean + sigma * np.dot(weights, y_sorted[:mu])
            
            # Update Evolution Paths
            y_w = np.dot(weights, y_sorted[:mu]) # Weighted mean of y
            z_w = np.dot(weights, z_sorted[:mu]) # Weighted mean of z
            
            # Conjugate Evolution Path (ps)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_w)
            
            # Check hsig to stall update if step size is large
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN < 1.4 + 2 / (dim + 1)
            
            # Evolution Path (pc)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - mean_old) / sigma
            
            # 5. Covariance Matrix Adaptation
            rank_1 = np.outer(pc, pc)
            # Rank-mu update: sum(w_i * y_i * y_i.T)
            # Vectorized computation for efficiency:
            weighted_Y = y_sorted[:mu].T * weights # Broadcasting weights
            rank_mu = weighted_Y @ y_sorted[:mu]
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu + c1 * delta_hsig) * C + c1 * rank_1 + cmu * rank_mu
            
            # 6. Step Size Control
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 7. Convergence/Restart Criteria
            
            # Condition Number Limit
            if np.max(D) > 1e7 * np.min(D):
                stop_restart = True
            
            # Sigma too small
            if sigma < 1e-15:
                stop_restart = True
                
            # Fitness Stagnation (on the penalized fitness)
            current_best_fit = fitness[0]
            if abs(current_best_fit - last_best_fit_restart) < 1e-12:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fit_restart = current_best_fit
                
            if stagnation_counter > 15 + np.sqrt(dim):
                stop_restart = True
            
            # Sigma explosion
            if sigma > (np.max(ub - lb) * 5):
                stop_restart = True
                
            gen += 1
            
        restart_count += 1

    # ---------------------------------------------------------
    # 3. Final Polishing Phase
    # ---------------------------------------------------------
    # Use any remaining time to refine the best solution found
    remaining_time = get_remaining_time()
    if remaining_time > 0:
        best_x, best_f = mts_ls1(best_x, best_f, remaining_time)

    return best_f
