#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 9.510001392407787
#2. algorithm code is: [Restarting Memetic SHADE with OBL]
#3. output value is: 41.458591949705784
#4. algorithm code is: [Restarting Memetic SHADE]
#5. output value is: 10.210149714609102
#6. algorithm code is: [Restarting JADE]
#
#The following Python code implements a **Restarting Memetic SHADE with Eigen-Crossover and MTS Local Search**.
#
#**Key Improvements:**
#1.  **Eigen-Coordinate Crossover**: Standard Differential Evolution (DE) struggles with rotated landscapes (variable correlations). This algorithm periodically performs an Eigen-decomposition of the population's covariance matrix to establish a coordinate system aligned with the landscape's topology. Crossover is performed in this rotated space with 50% probability, significantly improving efficiency on non-separable functions.
#2.  **MTS-LS1 Local Search**: Replaces the simple coordinate descent with a Multiple Trajectory Search (MTS) local search strategy. This method uses asymmetric step sizes and is more effective at traversing valleys to polish the solution before a restart.
#3.  **Opposition-Based Initialization**: Retained from the best-performing algorithm to ensure high-quality initial diversity.
#4.  **SHADE Adaptation**: Uses historical memory to adapt mutation and crossover rates.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Memetic SHADE with Eigen-Crossover 
    and MTS Local Search.
    
    Combines:
    - SHADE (Success-History Adaptive DE)
    - Eigen-Crossover (Handles rotated landscapes via PCA)
    - OBL (Opposition-Based Learning initialization)
    - MTS-LS (MTS Local Search for polishing)
    - Time-based Restarts
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Helper: Check Time
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Configuration & Pre-processing
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracker
    global_best_val = float('inf')
    
    # -------------------------------------------------------------------------
    # Local Search: MTS-LS1 (Simplified)
    # -------------------------------------------------------------------------
    def mts_local_search(curr_vec, curr_val):
        """
        Performs a local search based on MTS-LS1 strategy.
        Uses asymmetric search steps to refine the solution.
        """
        x = curr_vec.copy()
        f_x = curr_val
        
        # Initial search range (step size)
        # Start with a large range to escape immediate basins, then shrink
        sr = (max_b - min_b) * 0.4
        
        # Iteration limit for LS to conserve time
        max_iter = 15
        
        for _ in range(max_iter):
            if check_time(): break
            
            improved = False
            # Random dimension order
            dims = np.random.permutation(dim)
            
            for i in dims:
                if check_time(): break
                
                original_xi = x[i]
                
                # 1. Try moving in negative direction
                x[i] = np.clip(original_xi - sr[i], min_b[i], max_b[i])
                val = func(x)
                
                if val < f_x:
                    f_x = val
                    improved = True
                    # Keep improvement
                else:
                    # 2. Try moving in positive direction (0.5 * step)
                    x[i] = np.clip(original_xi + 0.5 * sr[i], min_b[i], max_b[i])
                    val = func(x)
                    
                    if val < f_x:
                        f_x = val
                        improved = True
                        # Keep improvement
                    else:
                        # Revert
                        x[i] = original_xi
            
            # If no improvement in any dimension, shrink search range
            if not improved:
                sr *= 0.5
                # Terminate if steps are too small
                if np.max(sr) < 1e-8:
                    break
                    
        return x, f_x

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while not check_time():
        
        # --- Initialization ---
        # Population Size: Adaptive, clipped for performance
        NP = int(np.clip(18 * dim, 40, 100))
        
        # SHADE Memory
        H = 6
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Archive
        archive = []
        
        # OBL Initialization
        pop = min_b + np.random.rand(NP, dim) * diff_b
        pop_opp = min_b + max_b - pop
        
        fit = np.zeros(NP)
        fit_opp = np.zeros(NP)
        
        # Evaluate Pop
        for i in range(NP):
            if check_time(): return global_best_val
            fit[i] = func(pop[i])
            if fit[i] < global_best_val: global_best_val = fit[i]
            
        # Evaluate Pop_Opp
        for i in range(NP):
            if check_time(): return global_best_val
            fit_opp[i] = func(pop_opp[i])
            if fit_opp[i] < global_best_val: global_best_val = fit_opp[i]
            
        # Select best from OBL
        mask = fit_opp < fit
        pop = np.where(mask[:, None], pop_opp, pop)
        fitness = np.where(mask, fit_opp, fit)
        
        # --- Eigen Adaptation Setup ---
        # Only use Eigen crossover if dimensions are manageable to save time
        use_eigen = (dim <= 100)
        eigen_freq = 10 if dim < 20 else 50
        B = np.eye(dim) # Eigenbasis
        gen_count = 0
        
        # Stagnation tracking
        no_improv_count = 0
        last_best = np.min(fitness)
        
        # --- Evolutionary Loop ---
        while not check_time():
            gen_count += 1
            
            # Sort by fitness (needed for current-to-pbest)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Convergence Check
            curr_best = fitness[0]
            if np.abs(curr_best - last_best) < 1e-10:
                no_improv_count += 1
            else:
                no_improv_count = 0
                last_best = curr_best
            
            # Trigger Restart if converged or stagnant
            spread = fitness[-1] - fitness[0]
            if spread < 1e-8 or no_improv_count > 25:
                # Polish best solution before restarting
                ls_vec, ls_val = mts_local_search(pop[0], fitness[0])
                if ls_val < global_best_val:
                    global_best_val = ls_val
                break 
            
            # Update Eigenbasis (PCA on top 50% population)
            if use_eigen and gen_count % eigen_freq == 0:
                top_k = max(2, int(NP * 0.5))
                cov_mat = np.cov(pop[:top_k].T)
                try:
                    # eigh is for symmetric matrices (covariance)
                    _, vecs = np.linalg.eigh(cov_mat)
                    B = vecs
                except:
                    # Fallback if singular
                    B = np.eye(dim)
            
            # --- Parameter Generation ---
            r_idx = np.random.randint(0, H, NP)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            f_p = mu_f + 0.1 * np.random.standard_cauchy(NP)
            f_p = np.where(f_p > 1.0, 1.0, f_p)
            f_p = np.where(f_p <= 0.0, 0.5, f_p)
            
            # --- Mutation (DE/current-to-pbest/1) ---
            p_min = 2/NP
            p_val = np.random.uniform(p_min, 0.2)
            top_p = int(max(2, NP * p_val))
            
            pbest_idxs = np.random.randint(0, top_p, NP)
            x_pbest = pop[pbest_idxs]
            
            r1_idxs = np.random.randint(0, NP, NP)
            x_r1 = pop[r1_idxs]
            
            # Archive handling for r2
            if len(archive) > 0:
                arc_arr = np.array(archive)
                pop_all = np.vstack((pop, arc_arr))
            else:
                pop_all = pop
                
            r2_idxs = np.random.randint(0, len(pop_all), NP)
            x_r2 = pop_all[r2_idxs]
            
            f_col = f_p[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial with Eigen Rotation) ---
            # 50% chance to do crossover in rotated coordinate system if enabled
            rand_u = np.random.rand(NP, dim)
            cross_mask = rand_u < cr[:, None]
            # Guarantee 1 parameter
            j_rand = np.random.randint(0, dim, NP)
            cross_mask[np.arange(NP), j_rand] = True
            
            trial = np.zeros_like(pop)
            
            if use_eigen and gen_count > eigen_freq:
                do_rot = np.random.rand(NP) < 0.5
                
                # Standard Crossover indices
                std_idx = ~do_rot
                if np.any(std_idx):
                    trial[std_idx] = np.where(cross_mask[std_idx], mutant[std_idx], pop[std_idx])
                
                # Rotated Crossover indices
                rot_idx = do_rot
                if np.any(rot_idx):
                    # Project to Eigen-space: X_rot = X @ B
                    # B columns are eigenvectors
                    p_rot = pop[rot_idx] @ B
                    m_rot = mutant[rot_idx] @ B
                    
                    # Crossover in rotated space
                    t_rot = np.where(cross_mask[rot_idx], m_rot, p_rot)
                    
                    # Project back: X = X_rot @ B.T
                    trial[rot_idx] = t_rot @ B.T
            else:
                # Always standard crossover
                trial = np.where(cross_mask, mutant, pop)
            
            # --- Bound Handling (Midpoint) ---
            low_viol = trial < min_b
            if np.any(low_viol):
                trial[low_viol] = (min_b + pop)[low_viol] * 0.5
                
            high_viol = trial > max_b
            if np.any(high_viol):
                trial[high_viol] = (max_b + pop)[high_viol] * 0.5
                
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            succ_diff = []
            succ_f = []
            succ_cr = []
            
            for i in range(NP):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    diff = fitness[i] - f_trial
                    
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update pop
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    succ_diff.append(diff)
                    succ_f.append(f_p[i])
                    succ_cr.append(cr[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        no_improv_count = 0
            
            # Maintain Archive Size
            while len(archive) > NP:
                archive.pop(np.random.randint(0, len(archive)))
            
            # --- Update SHADE Memory ---
            if len(succ_diff) > 0:
                diffs = np.array(succ_diff)
                total_diff = np.sum(diffs)
                
                if total_diff > 0:
                    w = diffs / total_diff
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    # Weighted Lehmer Mean for F
                    mean_f = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                    # Weighted Arithmetic Mean for CR
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H

    return global_best_val
