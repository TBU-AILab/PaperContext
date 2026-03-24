#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 7.107091818266667
#1. algorithm code is: [L-SHADE-pAdapt code]
#...
#
#The best result so far (7.107) was achieved by **L-SHADE-pAdapt**, which uses Linear Population Size Reduction (LPSR). The second best (8.15) was a standard SHADE. The gap suggests that while L-SHADE is powerful, it might be stagnating near the global optimum due to a lack of fine-grained exploitation in the very final stages, or suboptimal boundary handling.
#
#The following algorithm implements **L-SHADE-LPSR-LS** (Linear Population Size Reduction SHADE with Local Search).
#
#**Key Improvements:**
#1.  **Gaussian Local Search (LS)**: To address the stagnation around 7.1, a lightweight Local Search strategy is interleaved during the second half of the optimization. It performs a Gaussian walk around the *current best solution* with a step size scaled by the population's standard deviation. This acts as a "focusing" mechanism to refine the best solution beyond what the crossover/mutation operators can achieve with a shrinking population.
#2.  **Midpoint-Target Constraint Handling**: Instead of standard `clip` (which accumulates points at bounds) or `reflection`, this algorithm uses the "Midpoint Target" rule: `if trial > bound: trial = (bound + parent) / 2`. This pulls the solution back into the feasible space while preserving the search direction and locality relative to the parent, leading to better behavior near boundaries.
#3.  **Tuned L-SHADE Parameters**: Based on previous runs, the initial population is set to `18 * dim` (capped at 100) to balance coverage and speed. The Archive rate is set to 2.3.
#4.  **Fully Vectorized with Safety**: The code minimizes Python loop overhead using NumPy and includes strict time checks to guarantee a result within `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-LPSR-LS (Linear Population Reduction SHADE with Local Search).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Initial Population Size: 
    # High enough for exploration, but capped to ensure enough generations run.
    # 18*dim is a good balance derived from previous attempts.
    pop_size_init = int(18 * dim)
    pop_size_init = max(30, min(pop_size_init, 100))
    
    # Minimum Population Size (for LPSR)
    pop_size_min = 4
    
    # Current Population Size
    pop_size = pop_size_init
    
    # Archive size relative to population (A = pop_size * arc_rate)
    arc_rate = 2.3
    
    # Memory Size for Adaptive Parameters (H)
    H = 5
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize Memory (M_CR, M_F)
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # External Archive
    archive = []
    
    best_val = float('inf')
    best_vec = None
    
    # Helper for time progress (0.0 to 1.0)
    def get_progress():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # -------------------------------------------------------------------------
    # Initial Evaluation
    # -------------------------------------------------------------------------
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while True:
        progress = get_progress()
        if progress >= 1.0:
            return best_val
        
        # ---------------------------------------------------------------------
        # 1. Linear Population Size Reduction (LPSR)
        # ---------------------------------------------------------------------
        # Calculate target size based on time progress
        target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Sort population by fitness
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:target_size]]
            fitness = fitness[sort_indices[:target_size]]
            pop_size = target_size
            
            # Reduce Archive to maintain ratio
            curr_arc_cap = int(pop_size * arc_rate)
            if len(archive) > curr_arc_cap:
                # Random removal preserves diversity better than truncating
                rm_count = len(archive) - curr_arc_cap
                rm_idxs = np.random.choice(len(archive), rm_count, replace=False)
                # Rebuild archive excluding removed indices
                new_archive = []
                rm_set = set(rm_idxs)
                for idx, sol in enumerate(archive):
                    if idx not in rm_set:
                        new_archive.append(sol)
                archive = new_archive
        
        # Update best from current pop (in case reduction shifted things, though unlikely)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_val:
            best_val = fitness[current_best_idx]
            best_vec = pop[current_best_idx].copy()
            
        # ---------------------------------------------------------------------
        # 2. Parameter Generation (Vectorized)
        # ---------------------------------------------------------------------
        # Pick random memory slots
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        f[f > 1] = 1.0 # Clip upper
        
        # Regenerate F <= 0
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg):
                break
            n_neg = np.sum(mask_neg)
            f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
            f[f > 1] = 1.0
            
        # ---------------------------------------------------------------------
        # 3. Mutation: current-to-pbest/1 (Vectorized)
        # ---------------------------------------------------------------------
        # Calculate p (linear reduction from 0.2 to 0.05)
        p_val = 0.2 - (0.15 * progress)
        p_val = max(0.05, p_val)
        
        # Identify p-best
        sorted_idx = np.argsort(fitness)
        n_pbest = max(2, int(p_val * pop_size))
        pbest_pool = sorted_idx[:n_pbest]
        
        # Select pbest for each individual
        pbest_choices = np.random.choice(pbest_pool, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (distinct from i)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Fix collisions r1 == i
        collisions_r1 = (r1_idxs == np.arange(pop_size))
        r1_idxs[collisions_r1] = (r1_idxs[collisions_r1] + 1) % pop_size
        x_r1 = pop[r1_idxs]
        
        # Select r2 (distinct from i, r1) from Union(Pop, Archive)
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
        
        n_all = len(pop_all)
        r2_idxs = np.random.randint(0, n_all, pop_size)
        
        # Fix collisions for r2 (iterative fix is fast for small N)
        # r2 != i and r2 != r1
        curr_idxs = np.arange(pop_size)
        for j in range(pop_size):
            while r2_idxs[j] == curr_idxs[j] or r2_idxs[j] == r1_idxs[j]:
                r2_idxs[j] = np.random.randint(0, n_all)
                
        x_r2 = pop_all[r2_idxs]
        
        # Compute Mutant Vectors
        f_v = f[:, None]
        mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # ---------------------------------------------------------------------
        # 4. Crossover (Binomial)
        # ---------------------------------------------------------------------
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        
        # Ensure at least one parameter comes from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(cross_mask, mutant, pop)
        
        # ---------------------------------------------------------------------
        # 5. Bound Constraints (Midpoint Target)
        # ---------------------------------------------------------------------
        # If trial[i,j] is out of bounds, set it to midpoint between 
        # bound and parent[i,j]. This preserves search direction better than clip.
        
        # Lower violation
        mask_l = trials < min_b
        if np.any(mask_l):
            # trials = (min_b + pop) / 2 where violation occurred
            # Note: broadcasting min_b (dim,) and pop (pop_size, dim) works
            trials = np.where(mask_l, (min_b + pop) / 2, trials)
            
        # Upper violation
        mask_u = trials > max_b
        if np.any(mask_u):
            trials = np.where(mask_u, (max_b + pop) / 2, trials)
            
        # ---------------------------------------------------------------------
        # 6. Evaluation & Selection
        # ---------------------------------------------------------------------
        succ_f = []
        succ_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            f_tri = func(trials[i])
            
            if f_tri <= fitness[i]:
                # Record success info if strictly better
                if f_tri < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - f_tri)
                    # Add parent to archive
                    archive.append(pop[i].copy())
                
                pop[i] = trials[i]
                fitness[i] = f_tri
                
                if f_tri < best_val:
                    best_val = f_tri
                    best_vec = trials[i].copy()
                    
        # ---------------------------------------------------------------------
        # 7. Memory Update
        # ---------------------------------------------------------------------
        # Enforce archive size limit
        curr_arc_cap = int(pop_size * arc_rate)
        while len(archive) > curr_arc_cap:
            # Remove random
            idx_rm = np.random.randint(0, len(archive))
            archive.pop(idx_rm)
            
        # Update Memory
        if len(succ_f) > 0:
            arr_sf = np.array(succ_f)
            arr_scr = np.array(succ_cr)
            arr_diff = np.array(diff_fitness)
            
            # Weights based on improvement
            total_diff = np.sum(arr_diff)
            if total_diff > 0:
                weights = arr_diff / total_diff
                
                # Weighted Lehmer Mean for F
                denom = np.sum(weights * arr_sf)
                if denom == 0:
                    mean_f = 0.5
                else:
                    mean_f = np.sum(weights * (arr_sf**2)) / denom
                
                # Weighted Mean for CR
                mean_cr = np.sum(weights * arr_scr)
                
                # Update memory
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                k_mem = (k_mem + 1) % H
                
        # ---------------------------------------------------------------------
        # 8. Gaussian Local Search (Exploitation Phase)
        # ---------------------------------------------------------------------
        # Only triggered in the second half of the search or if stuck.
        # Performs one extra evaluation per generation centered on best_vec.
        if progress > 0.5 and best_vec is not None:
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            # Step size based on population standard deviation
            # We add a tiny epsilon to avoid zero step size
            pop_std = np.std(pop, axis=0) + 1e-9
            
            # Generate candidate: best + Normal(0, std)
            ls_cand = best_vec + np.random.normal(0, 1, dim) * pop_std
            
            # Bound handling for LS (Clip is safe here as it's a small perturbation)
            ls_cand = np.clip(ls_cand, min_b, max_b)
            
            val_ls = func(ls_cand)
            
            if val_ls < best_val:
                best_val = val_ls
                best_vec = ls_cand
                # Replace the worst individual in population to maintain size
                worst_idx = np.argmax(fitness)
                pop[worst_idx] = ls_cand
                fitness[worst_idx] = val_ls

    return best_val
