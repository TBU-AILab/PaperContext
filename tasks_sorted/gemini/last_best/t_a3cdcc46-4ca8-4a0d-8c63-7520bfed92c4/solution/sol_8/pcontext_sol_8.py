#The following algorithm implements **L-SHADE-EIG-MTS** (L-SHADE with Eigen-Crossover and Multiple Trajectory Search).
#
#**Key Improvements:**
#1.  **Eigen-Crossover (Rotation Invariance)**: Standard DE (and previous jSO) handles separable problems well but struggles with rotated (non-separable) landscapes. This algorithm computes the covariance matrix of the population and applies a coordinate rotation (Eigenvector transformation) during crossover with 50% probability. This allows the search to adapt to the landscape's orientation.
#2.  **MTS-LS1 Local Search**: Instead of the previous Hooke-Jeeves pattern search, it utilizes **MTS-LS1** (Multiple Trajectory Search - Local Search 1). This is a highly efficient coordinate descent method used in top-ranking CEC algorithms (like L-SHADE-SPCNE) to refine the best solution. It is triggered when the population stagnates or periodically to ensure convergence.
#3.  **Refined jSO Adaptation**: Accurately implements the jSO weighted mutation ($F_w$) rules based on search progress, damping exploration in the early stages to prevent premature convergence to poor basins, and ensuring robust convergence later.
#4.  **Hybrid Restart Strategy**: Upon convergence, a restart occurs that preserves the elite solution but generates the rest of the population using a mix of global uniform sampling and local Gaussian sampling around the best found solution. This balances exploration and exploitation better than purely random restarts.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE-EIG-MTS Algorithm
    Combines:
    1. L-SHADE (Linear Pop Size Reduction, History Memory)
    2. jSO Parameter Adaptation (Weighted F, Linear p)
    3. Eigen Crossover (Rotation Invariance for non-separable functions)
    4. MTS-LS1 (Local Search for refinement)
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_timeout():
        return datetime.now() - start_time >= time_limit
    
    def get_progress():
        """Returns 0.0 to 1.0 representing time usage."""
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max(max_time, 1e-9), 1.0)

    # --- Boundaries & Init ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing (jSO formula: 25 * log(D) * sqrt(D))
    # Clipped to reasonable limits to balance exploration speed
    initial_pop_size = int(np.clip(25 * np.log(dim) * np.sqrt(dim), 50, 400))
    min_pop_size = 4
    pop_size = initial_pop_size
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_pos = None
    
    # --- Evaluation Helper with Boundary Reflection ---
    def evaluate(x):
        nonlocal best_fitness, best_pos
        
        # 1. Reflection (Bounce back logic)
        lb_vio = x < min_b
        ub_vio = x > max_b
        
        if np.any(lb_vio):
            x[lb_vio] = 2 * min_b[lb_vio] - x[lb_vio]
        if np.any(ub_vio):
            x[ub_vio] = 2 * max_b[ub_vio] - x[ub_vio]
            
        # 2. Final Clip (Safety)
        x = np.clip(x, min_b, max_b)
        
        # 3. Evaluate
        val = func(x)
        
        if val < best_fitness:
            best_fitness = val
            best_pos = x.copy()
            
        return val, x

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if check_timeout(): return best_fitness
        fitness[i], pop[i] = evaluate(pop[i])

    # --- Memory & State ---
    H = 5 # Memory size
    mem_F = np.full(H, 0.5)
    mem_CR = np.full(H, 0.8)
    mem_k = 0
    archive = []
    
    # Eigen-Crossover Variables
    # Disable Eigen for high dimensions to save computation time (O(D^3))
    use_eigen = dim <= 100 
    B = np.eye(dim)
    
    # MTS-LS1 Variables
    mts_sr = diff_b * 0.4 # Search range for local search
    last_best_fit = best_fitness
    stagnation_counter = 0
    
    # --- Main Loop ---
    while not check_timeout():
        prog = get_progress()
        
        # -----------------------------------------------------------
        # 1. Linear Population Size Reduction (LPSR)
        # -----------------------------------------------------------
        plan_pop_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * prog))
        plan_pop_size = max(min_pop_size, plan_pop_size)
        
        if pop_size > plan_pop_size:
            n_reduce = pop_size - plan_pop_size
            # Remove worst individuals
            idx_sort = np.argsort(fitness)
            pop = pop[idx_sort[:-n_reduce]]
            fitness = fitness[idx_sort[:-n_reduce]]
            pop_size = plan_pop_size
            
            # Reduce Archive
            if len(archive) > pop_size:
                import random
                random.shuffle(archive)
                archive = archive[:pop_size]

        # -----------------------------------------------------------
        # 2. Parameter Adaptation (jSO / SHADE)
        # -----------------------------------------------------------
        # p (top % for mutation) reduces linearly from 0.25 to 2/N
        p_max, p_min = 0.25, 2.0 / pop_size
        p = p_max - (p_max - p_min) * prog
        
        # Select memory slots
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_CR[r_idx]
        m_f = mem_F[r_idx]
        
        # Generate CR ~ Normal
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F ~ Cauchy
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints
        F[F > 1.0] = 1.0
        neg_mask = F <= 0
        while np.any(neg_mask):
            F[neg_mask] = m_f[neg_mask] + 0.1 * np.random.standard_cauchy(np.sum(neg_mask))
            neg_mask = F <= 0
            
        # jSO Weighted Mutation Factor (Fw)
        # Damps exploration in early stages (Progress < 0.4)
        F_w = F.copy()
        if prog < 0.2:
            F_w *= 0.7
        elif prog < 0.4:
            F_w *= 0.8
            
        # -----------------------------------------------------------
        # 3. Eigen Covariance Update
        # -----------------------------------------------------------
        # Update coordinate basis B using top 50% individuals
        if use_eigen and np.random.rand() < 0.5: 
            if pop_size > dim: # Need enough samples
                top_idx = np.argsort(fitness)[:int(pop_size * 0.5)]
                cov_mat = np.cov(pop[top_idx].T)
                try:
                    _, vecs = np.linalg.eigh(cov_mat)
                    B = vecs
                except:
                    pass # Keep previous B if fail

        # -----------------------------------------------------------
        # 4. Mutation & Crossover
        # -----------------------------------------------------------
        # Sort for p-best selection
        sorted_idx = np.argsort(fitness)
        n_pbest = max(1, int(p * pop_size))
        
        # Indices
        pbest_idx = sorted_idx[np.random.randint(0, n_pbest, pop_size)]
        x_pbest = pop[pbest_idx]
        
        # r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        conflict = (r1 == np.arange(pop_size))
        r1[conflict] = (r1[conflict] + 1) % pop_size
        x_r1 = pop[r1]
        
        # r2 != r1, r2 != i (Union Pop U Archive)
        pool = pop if len(archive) == 0 else np.vstack((pop, np.array(archive)))
        r2 = np.random.randint(0, len(pool), pop_size)
        conflict2 = (r2 == r1) | (r2 == np.arange(pop_size))
        r2[conflict2] = (r2[conflict2] + 1) % len(pool)
        x_r2 = pool[r2]
        
        # Mutation: current-to-pbest/1 (Weighted)
        # v = x + Fw * (xpbest - x) + Fw * (xr1 - xr2)
        diff_pbest = x_pbest - pop
        diff_r = x_r1 - x_r2
        mutant = pop + F_w[:, None] * diff_pbest + F_w[:, None] * diff_r
        
        # Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) <= CR[:, None]
        cross_mask[np.arange(pop_size), j_rand] = True
        
        # Determine Eigen Usage for this generation (50% prob)
        eigen_mask = np.zeros(pop_size, dtype=bool)
        if use_eigen:
            eigen_mask = np.random.rand(pop_size) < 0.5
            
        trial = np.zeros_like(pop)
        
        # A. Standard Crossover
        if np.any(~eigen_mask):
            trial[~eigen_mask] = np.where(cross_mask[~eigen_mask], mutant[~eigen_mask], pop[~eigen_mask])
            
        # B. Eigen Crossover (Rotate -> Crossover -> Rotate Back)
        if np.any(eigen_mask):
            # Rotate Pop & Mutant to Eigen-basis
            pop_rot = np.dot(pop[eigen_mask], B)
            mut_rot = np.dot(mutant[eigen_mask], B)
            
            # Crossover in rotated space
            c_mask_rot = cross_mask[eigen_mask]
            trial_rot = np.where(c_mask_rot, mut_rot, pop_rot)
            
            # Rotate back
            trial[eigen_mask] = np.dot(trial_rot, B.T)
            
        # -----------------------------------------------------------
        # 5. Selection & Memory Update
        # -----------------------------------------------------------
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        success_F = []
        success_CR = []
        diff_fit = []
        
        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            f_trial, x_trial = evaluate(trial[i])
            
            if f_trial < fitness[i]:
                new_pop[i] = x_trial
                new_fitness[i] = f_trial
                
                archive.append(pop[i].copy())
                success_F.append(F[i])
                success_CR.append(CR[i])
                diff_fit.append(fitness[i] - f_trial)
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # Update Memory (Weighted Lehmer Mean)
        if len(diff_fit) > 0:
            weights = np.array(diff_fit)
            total_imp = np.sum(weights)
            if total_imp > 0:
                weights /= total_imp
                sF = np.array(success_F)
                sCR = np.array(success_CR)
                
                mean_f = np.sum(weights * (sF**2)) / (np.sum(weights * sF) + 1e-15)
                mean_cr = np.sum(weights * sCR)
                
                mem_F[mem_k] = 0.5 * mem_F[mem_k] + 0.5 * mean_f
                mem_CR[mem_k] = 0.5 * mem_CR[mem_k] + 0.5 * mean_cr
                mem_k = (mem_k + 1) % H
        
        # -----------------------------------------------------------
        # 6. MTS-LS1 Local Search & Stagnation Check
        # -----------------------------------------------------------
        if best_fitness < last_best_fit:
            last_best_fit = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        var_fit = np.var(fitness)
        
        # Trigger Local Search if stagnant or low variance
        do_ls = (stagnation_counter > 20) or (var_fit < 1e-8)
        
        if do_ls and best_pos is not None:
            # MTS-LS1: Coordinate Descent
            current_x = best_pos.copy() # Operate on copy
            current_f = best_fitness
            improved_ls = False
            
            dims_perm = np.random.permutation(dim)
            
            for d_i in dims_perm:
                if check_timeout(): break
                
                # 1. Try Negative Step
                xt = current_x.copy()
                xt[d_i] -= mts_sr[d_i]
                f_new, x_new = evaluate(xt)
                
                if f_new < current_f:
                    current_f = f_new
                    current_x = x_new
                    improved_ls = True
                else:
                    # 2. Try Positive Step (Half size)
                    xt = current_x.copy()
                    xt[d_i] += 0.5 * mts_sr[d_i]
                    f_new, x_new = evaluate(xt)
                    
                    if f_new < current_f:
                        current_f = f_new
                        current_x = x_new
                        improved_ls = True
            
            if not improved_ls:
                mts_sr *= 0.5 # Shrink search radius
            else:
                last_best_fit = current_f # Reset stagnation
                stagnation_counter = 0

        # -----------------------------------------------------------
        # 7. Restart Strategy
        # -----------------------------------------------------------
        # If converged (variance low) and LS radius is tiny, restart
        if var_fit < 1e-9 and np.max(mts_sr) < 1e-5:
            # Preserve Elite
            new_pop = np.zeros_like(pop)
            new_pop[0] = best_pos.copy()
            fitness[0] = best_fitness
            
            # Global Random (50%)
            n_global = (pop_size - 1) // 2
            if n_global > 0:
                new_pop[1:1+n_global] = min_b + np.random.rand(n_global, dim) * diff_b
            
            # Local Gaussian around Best (50%)
            n_local = (pop_size - 1) - n_global
            if n_local > 0:
                sigma = 0.05 * diff_b
                noise = np.random.randn(n_local, dim) * sigma
                new_pop[1+n_global:] = best_pos + noise
            
            # Evaluate
            for i in range(1, pop_size):
                if check_timeout(): return best_fitness
                fitness[i], new_pop[i] = evaluate(new_pop[i])
                
            pop = new_pop
            
            # Reset adaptation partially
            mem_F.fill(0.5)
            mem_CR.fill(0.5)
            mts_sr = diff_b * 0.4
            
    return best_fitness
