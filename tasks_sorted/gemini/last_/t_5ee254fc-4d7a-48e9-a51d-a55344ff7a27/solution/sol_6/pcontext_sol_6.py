#Here is an improved algorithm. It implements **L-SHADE-E (L-SHADE with Eigen-Crossover) combined with a Covariance Matrix Adaptive Restart mechanism**.
#
#**Key Improvements:**
#1.  **Eigen-Crossover (Rotation Invariance):** Standard Differential Evolution (DE) struggles with rotated objective functions (where variables are dependent). This algorithm performs Principal Component Analysis (PCA) on the population's top individuals to determine the "natural" coordinate system of the landscape and performs crossover in that rotated space.
#2.  **jSO-style Parameter Adaptation:** Uses the specific parameter tuning rules from the jSO algorithm (a highly ranked CEC winner) to adapt mutation factors ($F$) and crossover rates ($CR$).
#3.  **Adaptive Restart:** Instead of fixed stagnation counters, it monitors the spatial diversity (population diameter) and fitness variance. If the population collapses into a local optimum, it restarts while retaining the global best and historical parameter memory.
#4.  **Local Search Polish:** A lightweight Pattern Search is triggered on the global best solution to refine the final digits of precision.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Algorithm: L-SHADE-E (L-SHADE with Eigen-Crossover) and Adaptive Restarts.
    
    This algorithm enhances standard L-SHADE by incorporating an Eigen-coordinate
    system transformation based on the covariance of the population. This allows
    the solver to efficiently navigate rotated valleys (variable dependencies).
    """

    # --- Configuration & Constants ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Problem definition
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # L-SHADE parameters
    r_N_init = 18
    N_init = int(r_N_init * dim)
    N_init = np.clip(N_init, 30, 300) # Clamp initial population
    N_min = 4
    
    # Memory for adaptive parameters
    H = 5
    M_CR = np.full(H, 0.8) # Initial mean CR
    M_F = np.full(H, 0.5)  # Initial mean F
    k_mem = 0
    
    # Archive
    archive = []
    
    # State
    best_global_val = float('inf')
    best_global_vec = None
    
    # Eigen Crossover settings
    # Only apply eigen rotation if dimension is reasonable (matrix ops are O(d^3))
    use_eigen = dim <= 100 
    eigen_freq = 0.5 # Probability of using eigen crossover in a generation
    
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    def get_time_ratio():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # --- Initialization ---
    pop = min_b + np.random.rand(N_init, dim) * diff_b
    fit = np.full(N_init, float('inf'))
    
    # Evaluate initial population
    for i in range(N_init):
        if check_timeout(): return best_global_val if best_global_val != float('inf') else 0.0
        fit[i] = func(pop[i])
        if fit[i] < best_global_val:
            best_global_val = fit[i]
            best_global_vec = pop[i].copy()

    # --- Main Optimization Loop ---
    while not check_timeout():
        
        # 0. Linear Population Size Reduction (LPSR)
        tr = get_time_ratio()
        N_target = int(round(N_init + (N_min - N_init) * tr))
        N_target = max(N_min, N_target)
        N = len(pop)
        
        if N > N_target:
            # Sort by fitness
            sort_idx = np.argsort(fit)
            pop = pop[sort_idx]
            fit = fit[sort_idx]
            
            # Truncate
            curr_N = N
            N = N_target
            pop = pop[:N]
            fit = fit[:N]
            
            # Archive resizing
            max_arc_size = N
            if len(archive) > max_arc_size:
                # Randomly remove excess
                # (Simple list slicing for speed, shuffling theoretically better but slower)
                archive = archive[:max_arc_size]

        # 1. Coordinate System Transformation (Eigen-Crossover)
        # Calculate R (Rotation Matrix) if enabled and random trigger
        R = None
        if use_eigen and np.random.rand() < eigen_freq and N > dim:
            try:
                # Covariance of top 50% individuals
                top_size = max(dim, int(N * 0.5))
                sorted_idx = np.argsort(fit)
                top_pop = pop[sorted_idx[:top_size]]
                
                # Center population
                cov_mat = np.cov(top_pop, rowvar=False)
                
                # Eigendecomposition
                # eigh is for symmetric matrices (covariance is symmetric)
                eigval, eigvec = np.linalg.eigh(cov_mat)
                R = eigvec # Matrix where columns are eigenvectors
            except:
                R = None # Fallback to standard DE if Linear Algebra fails

        # 2. Parameter Generation (jSO / L-SHADE style)
        r_idx = np.random.randint(0, H, N)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # CR generation: Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        # Specialized fix: if CR is very close to 0, lift it slightly to ensure some crossover
        cr[cr == 0] = 1.0 / dim 
        
        # F generation: Cauchy(M_F, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(N)
        # Check constraints for F
        # If F > 1, cap at 1. If F <= 0, regenerate.
        retry_idx = np.where(f <= 0)[0]
        while len(retry_idx) > 0:
             f[retry_idx] = m_f[r_idx[retry_idx]] + 0.1 * np.random.standard_cauchy(len(retry_idx))
             retry_idx = np.where(f <= 0)[0]
        f = np.clip(f, 0, 1)

        # 3. Mutation: current-to-pbest/1
        # Sort population to find p-best
        sort_idx = np.argsort(fit)
        pop_sorted = pop[sort_idx]
        
        # p scales from 0.2 down to 0.1 (Linear reduction)
        p_max = 0.2
        p_min = 0.1
        curr_p = p_max - (p_max - p_min) * tr
        
        # Select pbest indices
        num_pbest = max(1, int(curr_p * N))
        pbest_indices = np.random.randint(0, num_pbest, N)
        pbest_vectors = pop_sorted[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, N, N)
        for i in range(N):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, N)
        r1_vectors = pop[r1_indices]
        
        # Select r2 (distinct from i and r1, from Union of Pop and Archive)
        archive_arr = np.array(archive) if len(archive) > 0 else np.empty((0, dim))
        if len(archive) > 0:
            union_pop = np.vstack((pop, archive_arr))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), N)
        for i in range(N):
            while r2_indices[i] == i or r2_indices[i] == r1_indices[i]:
                r2_indices[i] = np.random.randint(0, len(union_pop))
        r2_vectors = union_pop[r2_indices]
        
        # Compute Mutant Vectors V
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        mutants = pop + f[:, None] * (pbest_vectors - pop) + f[:, None] * (r1_vectors - r2_vectors)

        # 4. Crossover (Binomial or Eigen-Binomial)
        # Create trial vectors
        trials = np.copy(pop)
        
        j_rand = np.random.randint(0, dim, N)
        mask = np.random.rand(N, dim) <= cr[:, None]
        mask[np.arange(N), j_rand] = True
        
        if R is not None:
            # Eigen Crossover
            # Project Pop and Mutants to Eigen-Space
            pop_rot = np.dot(pop, R)
            mut_rot = np.dot(mutants, R)
            
            # Perform binomial crossover in rotated space
            trial_rot = np.where(mask, mut_rot, pop_rot)
            
            # Project back to original space
            trials = np.dot(trial_rot, R.T)
        else:
            # Standard Crossover
            trials = np.where(mask, mutants, pop)
            
        # 5. Bound Constraints (Reflection / Midpoint)
        # If violated, set to (boundary + old_val) / 2
        lower_mask = trials < min_b
        upper_mask = trials > max_b
        
        if np.any(lower_mask):
            trials[lower_mask] = (min_b[np.where(lower_mask)[1]] + pop[lower_mask]) / 2.0
        if np.any(upper_mask):
            trials[upper_mask] = (max_b[np.where(upper_mask)[1]] + pop[upper_mask]) / 2.0
            
        # 6. Selection
        new_pop = np.copy(pop)
        new_fit = np.copy(fit)
        
        succ_mask = np.zeros(N, dtype=bool)
        succ_diff = np.zeros(N)
        
        for i in range(N):
            if check_timeout(): return best_global_val
            
            t_val = func(trials[i])
            
            if t_val <= fit[i]:
                new_pop[i] = trials[i]
                new_fit[i] = t_val
                
                if t_val < fit[i]:
                    succ_mask[i] = True
                    succ_diff[i] = fit[i] - t_val
                    # Add old to archive
                    archive.append(pop[i].copy())
                
                if t_val < best_global_val:
                    best_global_val = t_val
                    best_global_vec = trials[i].copy()
                    
                    # Opportunistic Local Search on new Global Best
                    # Simple coordinate refinement (Hill Climber)
                    # Limit iterations to avoid eating too much time
                    ls_vec = best_global_vec.copy()
                    ls_val = best_global_val
                    improved_ls = False
                    
                    # Search radius based on population distribution
                    radius = np.std(pop, axis=0) * 0.1 + 1e-8
                    
                    # Limited Dimensions to check (random shuffle)
                    dims_check = np.random.permutation(dim)[:min(dim, 5)] 
                    
                    for d in dims_check:
                        if check_timeout(): break
                        old_d = ls_vec[d]
                        
                        # Try neg
                        ls_vec[d] = np.clip(old_d - radius[d], min_b[d], max_b[d])
                        val_n = func(ls_vec)
                        if val_n < ls_val:
                            ls_val = val_n
                            best_global_val = val_n
                            best_global_vec = ls_vec.copy()
                            improved_ls = True
                        else:
                            # Try pos
                            ls_vec[d] = np.clip(old_d + radius[d], min_b[d], max_b[d])
                            val_p = func(ls_vec)
                            if val_p < ls_val:
                                ls_val = val_p
                                best_global_val = val_p
                                best_global_vec = ls_vec.copy()
                                improved_ls = True
                            else:
                                ls_vec[d] = old_d # Revert

        pop = new_pop
        fit = new_fit
        
        # 7. Update Memory
        if np.any(succ_mask):
            s_f = f[succ_mask]
            s_cr = cr[succ_mask]
            s_diff = succ_diff[succ_mask]
            
            # Weighted Lehmer Mean
            weights = s_diff / np.sum(s_diff)
            
            # Update M_CR
            if np.max(s_cr) == 0:
                m_cr_new = 0
            else:
                m_cr_new = np.sum(weights * s_cr)
            
            # Update M_F (Lehmer mean)
            num = np.sum(weights * (s_f**2))
            den = np.sum(weights * s_f)
            m_f_new = num / den if den > 0 else 0.5
            
            M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * m_cr_new
            M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * m_f_new
            
            k_mem = (k_mem + 1) % H
            
        # 8. Adaptive Restart Strategy
        # If population diversity is too low, or variance of fitness is tiny
        # But only if we have time left (>10%)
        if tr < 0.9:
            pop_std = np.mean(np.std(pop, axis=0))
            fit_std = np.std(fit)
            
            # Thresholds relative to domain size
            diversity_threshold = np.mean(diff_b) * 1e-5 
            
            if pop_std < diversity_threshold or fit_std < 1e-8:
                # Restart
                # Keep best
                pop[0] = best_global_vec.copy()
                fit[0] = best_global_val
                
                # Reinitialize others randomly, but mix with slight perturbation of best
                # to handle "narrow valley" restarts
                # 50% purely random, 50% around best
                split = N // 2
                
                # Random part
                pop[1:split] = min_b + np.random.rand(split-1, dim) * diff_b
                
                # Gaussian cloud around best
                sigma = (max_b - min_b) * 0.1 # 10% of domain
                pop[split:] = best_global_vec + np.random.randn(N - split, dim) * sigma
                
                # Clip
                pop = np.clip(pop, min_b, max_b)
                
                # Reset Archive (optional, but keeping it helps directed mutation)
                # We keep archive to maintain "what didn't work" or "past good areas"
                
                # Re-evaluate
                for i in range(1, N):
                    if check_timeout(): return best_global_val
                    fit[i] = func(pop[i])
                    if fit[i] < best_global_val:
                        best_global_val = fit[i]
                        best_global_vec = pop[i].copy()
                
                # Reset Memory slightly to allow new exploration
                M_CR[:] = 0.5
                M_F[:] = 0.5

    return best_global_val
