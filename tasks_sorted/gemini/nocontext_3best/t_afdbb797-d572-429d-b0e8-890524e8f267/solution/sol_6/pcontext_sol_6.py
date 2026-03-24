#The following Python code implements **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**, adapted for time-constrained execution.
#
#**Key Improvements over previous algorithms:**
#1.  **Linear Population Size Reduction (LPSR)**: Unlike fixed-population algorithms (JADE/jDE), this algorithm starts with a larger diverse population to explore the search space and linearly reduces the population size as time elapses. This forces the algorithm to shift from global exploration to rapid local exploitation (convergence) exactly as the deadline approaches, maximizing the utility of the available time.
#2.  **Success-History Adaptation**: Instead of a single adaptive parameter set, it maintains a memory (history) of successful $F$ and $Cr$ configurations. This allows the algorithm to recall and utilize multiple distinct mutation strategies that worked in the past, adapting better to complex fitness landscapes.
#3.  **Time-Aware Execution**: The reduction schedule is dynamically calculated based on the `elapsed_time / max_time` ratio, ensuring the population reaches its minimum optimal size (exploitation phase) right before the timeout.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using L-SHADE with Time-Based Population Reduction.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial population size: Larger than standard DE to ensure initial coverage.
    # We clamp it (50-120) to ensure we get enough generations within the time limit.
    pop_size_init = int(np.clip(20 * dim, 50, 120))
    pop_size_min = 4  # Minimum population size at the end of execution
    pop_size = pop_size_init
    
    # Archive parameters (External archive to preserve diversity)
    arc_rate = 2.0
    archive = []  # Stores 1D numpy arrays
    
    # Memory for adaptive parameters (History length H=5)
    mem_size = 5
    m_cr = np.full(mem_size, 0.5)
    m_f = np.full(mem_size, 0.5)
    mem_k = 0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        # Strict time check inside loops
        if (datetime.now() - start_time) >= limit:
            return best_val if best_val != float('inf') else fitness[0]
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Time Check & Progress Calculation
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        if elapsed >= max_time:
            return best_val
            
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target size based on percentage of time used
        time_ratio = elapsed / max_time
        # Formula: N_t = round((N_min - N_init) * ratio + N_init)
        target_size = int(round((pop_size_min - pop_size_init) * time_ratio + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Reduce population: remove worst individuals
            sorted_idx = np.argsort(fitness)
            keep_idx = sorted_idx[:target_size]
            
            pop = pop[keep_idx]
            fitness = fitness[keep_idx]
            pop_size = target_size
            
            # Resize Archive proportionally
            arc_target = int(pop_size * arc_rate)
            while len(archive) > arc_target:
                del archive[np.random.randint(0, len(archive))]
        
        # 2. Parameter Adaptation
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, mem_size, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Retry F <= 0, Clip F > 1
        neg_mask = f <= 0
        retry_c = 0
        while np.any(neg_mask) and retry_c < 5:
            f[neg_mask] = mu_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(neg_mask)) - 0.5))
            neg_mask = f <= 0
            retry_c += 1
        f = np.clip(f, 0.0, 1.0)
        f[f <= 0] = 0.5  # Fallback
        
        # 3. Mutation: current-to-pbest/1
        # Sort for p-best selection (top 11%)
        sorted_indices = np.argsort(fitness)
        p_num = max(int(pop_size * 0.11), 2)
        top_p_indices = sorted_indices[:p_num]
        
        # Select p-best
        r_best = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[r_best]
        
        # Select r1 (distinct from i)
        r1 = np.random.randint(0, pop_size, pop_size)
        conflict = (r1 == np.arange(pop_size))
        while np.any(conflict):
            r1[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
            conflict = (r1 == np.arange(pop_size))
        x_r1 = pop[r1]
        
        # Select r2 (distinct from i and r1, from Pop U Archive)
        if len(archive) > 0:
            arc_np = np.array(archive)
            union_pop = np.vstack((pop, arc_np))
        else:
            union_pop = pop
        
        union_size = len(union_pop)
        r2 = np.random.randint(0, union_size, pop_size)
        
        # Check r2 collisions
        r2_in_pop = r2 < pop_size
        collision = (r2_in_pop & (r2 == np.arange(pop_size))) | (r2_in_pop & (r2 == r1))
        while np.any(collision):
            r2[collision] = np.random.randint(0, union_size, np.sum(collision))
            r2_in_pop = r2 < pop_size
            collision = (r2_in_pop & (r2 == np.arange(pop_size))) | (r2_in_pop & (r2 == r1))
            
        x_r2 = union_pop[r2]
        
        # Calculate Mutant Vector
        f_col = f[:, np.newaxis]
        v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = rand_vals < cr[:, np.newaxis]
        mask[np.arange(pop_size), j_rand] = True
        
        u = np.where(mask, v, pop)
        
        # 5. Bound Handling (Reflection)
        mask_l = u < min_b
        if np.any(mask_l):
            u[mask_l] = 2 * min_b[np.where(mask_l)[1]] - u[mask_l]
            u = np.maximum(u, min_b)
            
        mask_u = u > max_b
        if np.any(mask_u):
            u[mask_u] = 2 * max_b[np.where(mask_u)[1]] - u[mask_u]
            u = np.minimum(u, max_b)
            
        # 6. Evaluation & Selection
        success_f = []
        success_cr = []
        df = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= limit:
                return best_val
            
            val_trial = func(u[i])
            
            if val_trial <= fitness[i]:
                # Improvement
                if val_trial < fitness[i]:
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    df.append(fitness[i] - val_trial)
                    # Add parent to archive
                    archive.append(pop[i].copy())
                
                pop[i] = u[i]
                fitness[i] = val_trial
                
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = u[i].copy()
                    
        # Maintain Archive Capacity
        arc_max = int(pop_size * arc_rate)
        while len(archive) > arc_max:
            del archive[np.random.randint(0, len(archive))]
            
        # 7. Update History Memory
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w_df = np.array(df)
            
            if np.sum(w_df) > 0:
                weights = w_df / np.sum(w_df)
                
                # Update M_CR (Weighted Mean)
                m_cr[mem_k] = np.sum(weights * s_cr)
                
                # Update M_F (Weighted Lehmer Mean)
                num = np.sum(weights * (s_f**2))
                den = np.sum(weights * s_f)
                if den > 0:
                    m_f[mem_k] = num / den
                    
                mem_k = (mem_k + 1) % mem_size

    return best_val
