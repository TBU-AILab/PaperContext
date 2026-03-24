#The following Python code implements an advanced version of the **L-SHADE** algorithm (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution).
#
#**Key Improvements over standard Random Search and Basic DE:**
#1.  **LPSR (Linear Population Size Reduction):** The algorithm starts with a large population to explore the global landscape and linearly reduces the population size as time progresses to focus on exploitation.
#2.  **Adaptive Parameters (SHADE):** It eliminates the need to manually tune mutation ($F$) and crossover ($CR$) rates. It learns these parameters during the run using a historical memory of successful values.
#3.  **Dynamic Selection Pressure:** The "p-best" parameter (controlling which best individuals direct the search) is dynamic, starting high to promote diversity and decreasing to enforce convergence.
#4.  **Midpoint Boundary Correction:** Instead of simple clipping, it uses midpoint reflection to prevent the population from sticking to the edges of the search space.
#5.  **Time-Aware Scheduling:** All adaptive mechanisms are mapped to the provided `max_time` budget, ensuring the algorithm utilizes the available time optimally without finishing too early or timing out.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Linear Population Size Reduction 
    and Time-Adaptive Scheduling.
    """
    start_time = time.time()
    
    # --- 1. Configuration & Constants ---
    # Bounds processing
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    
    # Population Sizing
    # Start with a sufficiently large population for exploration
    # N_init = 18 * dim is a standard heuristic in evolutionary computation
    init_pop_size = int(max(30, min(18 * dim, 250)))
    min_pop_size = 4
    
    # Archive size parameter (Archive stores A * N individuals)
    arc_rate = 2.0
    
    # Memory size for adaptive parameters
    H_mem = 6
    
    # --- 2. Initialization ---
    pop_size = init_pop_size
    
    # Initialize Population (Uniform Random)
    # pop shape: (pop_size, dim)
    pop = lower_b + (upper_b - lower_b) * np.random.rand(pop_size, dim)
    fitness = np.full(pop_size, float('inf'))
    
    # Tracking Global Best
    best_fitness = float('inf')
    
    # Evaluate Initial Population
    # We check time within the loop to be safe for slow functions
    for i in range(pop_size):
        if time.time() - start_time > max_time:
            return best_fitness if best_fitness != float('inf') else 0.0
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            
    # Sort population by fitness (required for L-SHADE rank-based logic)
    sorted_indices = np.argsort(fitness)
    pop = pop[sorted_indices]
    fitness = fitness[sorted_indices]
    
    # Initialize Adaptive Memory
    M_CR = np.full(H_mem, 0.5) # Memory for Crossover Rate
    M_F = np.full(H_mem, 0.5)  # Memory for Scaling Factor
    k_mem = 0                  # Memory index pointer
    
    # External Archive (stores successful past solutions for diversity)
    archive = [] 
    
    # --- 3. Main Optimization Loop ---
    while True:
        # Check Time Budget
        curr_time = time.time()
        elapsed = curr_time - start_time
        if elapsed >= max_time:
            break
            
        # Calculate Progress (0.0 to 1.0)
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # --- A. Linear Population Size Reduction (LPSR) ---
        # Calculate target population size based on remaining time
        target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        target_size = max(min_pop_size, target_size)
        
        # If current population is larger than target, delete worst individuals
        if pop_size > target_size:
            # Since pop is sorted, worst are at the end
            pop = pop[:target_size]
            fitness = fitness[:target_size]
            pop_size = target_size
            
            # Resize Archive: archive size scales with population size
            target_arc_size = int(pop_size * arc_rate)
            if len(archive) > target_arc_size:
                # Randomly remove excess from archive
                keep_indices = np.random.choice(len(archive), target_arc_size, replace=False)
                archive = [archive[k] for k in keep_indices]
        
        # --- B. Adaptive Parameter Generation ---
        # 1. Dynamic 'p' for current-to-pbest selection
        # Linearly decreases from 0.2 to 0.05 to increase selection pressure over time
        p_val = 0.2 - (0.15 * progress)
        p_best_count = max(2, int(pop_size * p_val))
        
        # 2. Select Memory Indices
        r_idx = np.random.randint(0, H_mem, size=pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # 3. Generate CR (Normal Distribution)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # 4. Generate F (Cauchy Distribution)
        f = m_f + 0.1 * np.random.standard_cauchy(size=pop_size)
        
        # Handling F constraints:
        # If F <= 0, regenerate until > 0
        retry_mask = f <= 0
        while np.any(retry_mask):
            f[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(size=np.sum(retry_mask))
            retry_mask = f <= 0
        # If F > 1, clip to 1
        f = np.clip(f, 0.0, 1.0)
        
        # --- C. Mutation Strategy: current-to-pbest/1 ---
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Select X_pbest (randomly from top p%)
        pbest_indices = np.random.randint(0, p_best_count, size=pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select X_r1 (random from population, r1 != i)
        r1_indices = np.random.randint(0, pop_size, size=pop_size)
        # Fix self-collision
        collision_mask = (r1_indices == np.arange(pop_size))
        r1_indices[collision_mask] = (r1_indices[collision_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select X_r2 (random from Union of Pop and Archive, r2 != r1, r2 != i)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        union_size = len(union_pop)
        r2_indices = np.random.randint(0, union_size, size=pop_size)
        
        # Simple collision fix for r2 (statistically sufficient)
        coll_r2 = (r2_indices == r1_indices) | (r2_indices == np.arange(pop_size))
        if np.any(coll_r2):
            r2_indices[coll_r2] = np.random.randint(0, union_size, size=np.sum(coll_r2))
            
        x_r2 = union_pop[r2_indices]
        
        # Calculate Mutant
        f_col = f[:, np.newaxis] # Reshape for broadcasting
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # --- D. Boundary Correction (Midpoint Reflection) ---
        # If a mutant parameter is outside bounds, place it halfway between parent and bound
        mask_l = mutant < lower_b
        if np.any(mask_l):
            mutant[mask_l] = (pop[mask_l] + lower_b[np.where(mask_l)[1]]) / 2.0
            
        mask_u = mutant > upper_b
        if np.any(mask_u):
            mutant[mask_u] = (pop[mask_u] + upper_b[np.where(mask_u)[1]]) / 2.0
            
        # --- E. Crossover (Binomial) ---
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals <= cr[:, np.newaxis]
        # Ensure at least one parameter comes from mutant
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask_cross, mutant, pop)
        
        # --- F. Selection & Evaluation ---
        success_f = []
        success_cr = []
        fit_diffs = []
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            # Strict time check inside loop to prevent timeout
            if time.time() - start_time > max_time:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                new_pop[i] = trial_pop[i]
                new_fitness[i] = f_trial
                
                # If strictly better, store success info
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    fit_diffs.append(fitness[i] - f_trial)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        
        pop = new_pop
        fitness = new_fitness
        
        # Manage Archive Size
        target_arc_size = int(pop_size * arc_rate)
        while len(archive) > target_arc_size:
            del_idx = np.random.randint(0, len(archive))
            archive.pop(del_idx)
            
        # --- G. Update Memory (Weighted Lehmer Mean) ---
        if len(fit_diffs) > 0:
            w_diff = np.array(fit_diffs)
            w = w_diff / np.sum(w_diff)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            # Lehmer Mean for F
            mean_f = np.sum(w * (s_f ** 2)) / (np.sum(w * s_f) + 1e-15)
            
            # Weighted Mean for CR
            mean_cr = np.sum(w * s_cr)
            
            # Update Memory Slots
            M_F[k_mem] = mean_f
            M_CR[k_mem] = mean_cr
            k_mem = (k_mem + 1) % H_mem
            
        # Re-sort population for next generation's rank-based logic
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
    return best_fitness
