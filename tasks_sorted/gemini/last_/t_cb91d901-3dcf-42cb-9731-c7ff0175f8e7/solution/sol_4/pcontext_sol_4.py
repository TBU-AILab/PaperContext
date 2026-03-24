#Here is a fully operational, self-contained Python algorithm. It implements **IPOP-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy)**.
#
#### Improvement Logic
#The previous algorithm (jSO/DE) is excellent for separable or weakly linked variables but can struggle with:
#1.  **Rotated Landscapes:** If variables are highly correlated (the "valley" is diagonal to the axes), DE's crossover operator is less efficient. CMA-ES learns the correlation matrix to align the search distribution with the landscape.
#2.  **Local Optima (Restart Strategy):** The previous restart mechanism was static. **IPOP-CMA-ES** systematically increases the population size after every restart (doubling it). This changes the search behavior from local (exploitation) with small populations to global (exploration) with large populations, significantly improving the chance of finding the global optimum in complex multimodal functions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements IPOP-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy).
    
    This algorithm adapts the covariance matrix of the multivariate normal distribution
    to learn the topography of the function (handling ill-conditioned and rotated problems).
    It uses a restart mechanism with increasing population size to escape local optima.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)

    def check_time():
        return (datetime.now() - start_time) >= limit

    # --- Problem Normalization ---
    # We map the specific bounds to a normalized space [0, 10] internally to
    # standardize the sigma (step size) logic.
    bounds = np.array(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    bound_range = upper_bounds - lower_bounds
    
    # Scale helper functions
    def to_pheno(x_geno):
        # Map [0, 10] back to [lower, upper]
        # We perform simple scaling. 
        # Note: CMA-ES can generate values outside, we clip for evaluation.
        unit = x_geno / 10.0
        val = lower_bounds + unit * bound_range
        return np.clip(val, lower_bounds, upper_bounds)

    def get_penalty(x_geno, x_pheno):
        # If CMA generates out of bounds, x_pheno is clipped.
        # We calculate distance in the normalized space for penalty.
        # This guides the mean back towards the feasible region.
        # Map pheno back to geno space to compare
        back_mapped = (x_pheno - lower_bounds) / bound_range * 10.0
        dist_sq = np.sum((x_geno - back_mapped)**2)
        return dist_sq

    # --- Global Best Tracking ---
    best_fitness = float('inf')
    best_sol = None # Stored in real (phenotype) coordinates

    # --- IPOP-CMA-ES Loop ---
    restart_count = 0
    
    while not check_time():
        
        # 1. Initialize Strategy Parameters for this Restart
        # Default population size for CMA-ES: 4 + 3 * log(N)
        # Increase by factor of 2^restart_count
        pop_size = int((4 + 3 * np.log(dim)) * (2 ** restart_count))
        
        # Initialize Mean (center of normalized space)
        xmean = np.full(dim, 5.0) 
        
        # Initialize Step-size (sigma)
        sigma = 2.0 # Covers a good chunk of [0, 10]
        
        # Initialize Covariance Matrix components
        # C = B * D * (B * D).T
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim) # Covariance matrix
        invsqrtC = np.eye(dim)
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Selection Weights (logarithmic)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights) # Normalize
        mueff = 1.0 / np.sum(weights**2) # Variance effective selection mass
        
        # Adaptation Constants (Standard CMA-ES settings)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Safety for eigen update frequency (O(N^3) is heavy)
        eigeneval = 0
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Inner Loop: Generations
        gen = 0
        stop_restart = False
        
        while not stop_restart and not check_time():
            gen += 1
            
            # 2. Sampling
            # Generate offspring: x_i = m + sigma * B * D * z_i
            # z ~ N(0, I)
            arz = np.random.standard_normal((pop_size, dim))
            ary = B @ (D * arz).T # (dim, pop_size)
            arx = xmean[:, None] + sigma * ary
            arx = arx.T # (pop_size, dim)
            
            # 3. Evaluation
            fitness = np.zeros(pop_size)
            
            for k in range(pop_size):
                if check_time(): return best_fitness
                
                # Boundary Handling
                real_params = to_pheno(arx[k])
                
                # Function Call
                val = func(real_params)
                
                # Save Best
                if val < best_fitness:
                    best_fitness = val
                    best_sol = real_params.copy()
                
                # Penalty for out-of-bounds (soft constraints)
                # Adds a gradient pointing back to the feasible region
                penalty = get_penalty(arx[k], real_params)
                fitness[k] = val + penalty * 1e4 # High penalty weight
            
            # 4. Selection and Recombination
            arindex = np.argsort(fitness)
            # Pick best mu
            arindex = arindex[:mu]
            
            xold = xmean.copy()
            # Recombination: Weighted mean of selected
            # (dim,) = ((dim, mu) @ (mu,)) 
            z_sel = arz[arindex]
            xmean = xmean + sigma * (ary[:, arindex] @ weights)
            
            # 5. Step-size Control (Evolution Path)
            # Calculate z_m (weighted mean of z)
            zmean = z_sel.T @ weights
            
            # Update evolution path ps
            # Conjugate evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            
            # HSA (Heaviside Step Function logic for pc update)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen+1))) / chiN < 1.4 + 2 / (dim + 1))
            
            # Update evolution path pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (ary[:, arindex] @ weights)
            
            # Update Sigma
            # exp( (norm(ps)/chiN - 1) * cs / damps )
            sigma *= np.exp((np.linalg.norm(ps) / chiN - 1) * cs / damps)
            
            # 6. Covariance Matrix Adaptation
            # Rank-1 update (pc * pc.T)
            # Rank-mu update (weighted variance of selected steps)
            
            # Calculate Rank-mu part: sum( w_i * y_i * y_i.T )
            # Optimized vector calc
            y_sel = ary[:, arindex] # (dim, mu)
            # We need matrix multiplication Y * W * Y.T where W is diag(weights)
            # Equivalent to (Y * weights) @ Y.T ?? No.
            # Just loop or specialized dot.
            # r_mu = y_sel @ np.diag(weights) @ y_sel.T # Expensive creation of diag
            # Use broadcasting:
            weighted_y = y_sel * weights[None, :]
            rank_mu_update = weighted_y @ y_sel.T
            
            pc_matrix = np.outer(pc, pc)
            
            # Combine updates
            # Adapt covariance matrix C
            # Old C * decay + Rank1 * c1 + RankMu * cmu
            C = (1 - c1 - cmu) * C \
                + c1 * (pc_matrix + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * rank_mu_update
            
            # 7. Decomposition (Eigen)
            # Update B and D from C every few generations (diagonalization)
            # O(N^3), so don't do it every step.
            if (gen - eigeneval) > (1.0 / (c1 + cmu) / dim / 10.0):
                eigeneval = gen
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                
                try:
                    eig_vals, eig_vecs = np.linalg.eigh(C)
                    # Numerical stability
                    eig_vals = np.maximum(eig_vals, 0)
                    D = np.sqrt(eig_vals)
                    B = eig_vecs
                except np.linalg.LinAlgError:
                    # If C is corrupted, clean it
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)

            # 8. Check Stop/Restart Criteria
            
            # A. Range of fitness in population is too small (Convergence)
            if fitness[arindex[-1]] - fitness[arindex[0]] < 1e-12:
                stop_restart = True
            
            # B. Sigma is too small (Stagnation)
            if sigma < 1e-15:
                stop_restart = True
                
            # C. Sigma is too large (Divergence / Exploring useless space)
            if sigma > 1e5:
                stop_restart = True
                
            # D. Condition number of C exceeds limit
            if np.min(D) > 0:
                cond_num = np.max(D) / np.min(D)
                if cond_num > 1e14:
                    stop_restart = True
            else:
                 stop_restart = True

            # E. No improvement over last 10 + 30*N/lambda generations (Stagnation)
            # (Simplified check: if best fitness hasn't moved significantly in last 50 gens)
            # Handled implicitly by sigma reduction usually, but explicit check:
            # Note: For time efficiency in this template, we rely mostly on A and B.

        # End of Inner Loop
        restart_count += 1
        
    return best_fitness
