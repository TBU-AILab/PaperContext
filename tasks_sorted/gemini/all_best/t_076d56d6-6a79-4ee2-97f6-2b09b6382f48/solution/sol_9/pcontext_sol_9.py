#The following Python code implements an optimized **Restart-based SHADE (Success-History based Adaptive Differential Evolution) with IPOP (Increasing Population Size)** and **Reflection-based Bound Handling**.
#
##### Key Improvements:
#1.  **Reflection Bound Handling**: Unlike previous iterations that simply clipped out-of-bound solutions to the boundary, this algorithm uses a reflection strategy ($x_{new} = 2 \cdot x_{bound} - x_{trial}$). This prevents the population from accumulating on the edges of the search space, preserving the natural distribution of the evolutionary strategy.
#2.  **Expanded Archive & Memory**: The external archive size is increased to `2.0 * pop_size` and memory size to `20`. This maintains higher population diversity and allows the adaptive parameters ($F$ and $CR$) to learn from a longer history of successful updates, stabilizing convergence.
#3.  **Robust Initialization**: The population initialization and adaptive parameters ($M_{CR}=0.8$) are tuned for robustness across both separable and non-separable problems.
#4.  **Vectorized Operations**: The mutation, crossover, and reflection logic are fully vectorized using NumPy to maximize the number of function evaluations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Restart-SHADE with IPOP and Reflection Bound Handling.
    
    Algorithm Description:
    1. SHADE Adaptation: Learns F and CR parameters using a history (memory) of successful updates.
    2. IPOP Strategy: Restarts with exponentially increasing population size to escape local optima.
    3. Reflection Bound Handling: Reflects out-of-bound solutions back into the space to avoid 
       edge-stacking, improving search near boundaries.
    4. Archive: Maintains a history of diverse, recently replaced solutions to guide mutation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Global Best Tracking ---
    best_val = float('inf')
    best_vec = None

    # --- Configuration ---
    # SHADE Parameters
    memory_size = 20  # Larger memory to retain adaptation history longer
    
    # IPOP Parameters
    # Start with a robust population size (30 + 10*dim) to ensure good initial statistics
    base_pop = 30 + 10 * dim 
    restart_count = 0

    # --- Main Restart Loop ---
    while True:
        # Check overall time limit before starting a new restart
        if datetime.now() - start_time >= time_limit:
            return best_val

        # IPOP: Scale population size exponentially (1.5x) with each restart
        pop_size = int(base_pop * (1.5 ** restart_count))
        
        # --- Initialization ---
        # Initialize population uniformly
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best solution into the new population
        start_idx = 0
        if best_vec is not None:
            pop[0] = best_vec.copy()
            fitness[0] = best_val
            start_idx = 1

        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            # Strict time check during initialization
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
        
        # --- SHADE Memory Initialization ---
        # Initialize CR to 0.8 (favors mixing) and F to 0.5 (balanced exploration)
        m_cr = np.full(memory_size, 0.8)
        m_f = np.full(memory_size, 0.5)
        k_mem = 0
        
        # --- Archive Initialization ---
        # Stores decent parent vectors replaced by better offspring
        # Capacity 2.0x pop_size maintains high diversity
        archive_size = int(2.0 * pop_size)
        archive = np.empty((archive_size, dim))
        arc_count = 0

        # --- Evolution Loop ---
        while True:
            # Check time
            if datetime.now() - start_time >= time_limit:
                return best_val

            # Convergence Check: If population fitness is flat, trigger restart
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break
            
            # --- 1. Parameter Generation (Vectorized) ---
            # Randomly select memory indices
            r_idx = np.random.randint(0, memory_size, pop_size)
            r_cr = m_cr[r_idx]
            r_f = m_f[r_idx]

            # Generate CR ~ Normal(M_CR, 0.1), clipped [0, 1]
            cr = np.random.normal(r_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(M_F, 0.1)
            u_f = np.random.rand(pop_size)
            f = r_f + 0.1 * np.tan(np.pi * (u_f - 0.5))
            
            # Handle F constraints: F <= 0 (retry), F > 1 (clip)
            mask_neg = f <= 0
            while np.any(mask_neg):
                cnt = np.sum(mask_neg)
                f[mask_neg] = r_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(cnt) - 0.5))
                mask_neg = f <= 0
            f = np.clip(f, 0.0, 1.0)
            
            # --- 2. Mutation: current-to-pbest/1 ---
            # Sort population by fitness to identify p-best
            sorted_idx = np.argsort(fitness)
            
            # Select p-best (top 10% greedy selection)
            p = 0.1
            p_num = max(2, int(p * pop_size))
            pbest_pool = sorted_idx[:p_num]
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 (random from Pop, distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            mask_self = r1_idx == np.arange(pop_size)
            r1_idx[mask_self] = (r1_idx[mask_self] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (random from Union(Pop, Archive), distinct from i, r1)
            union_size = pop_size + arc_count
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Approximate conflict handling for speed
            mask_conflict = (r2_idx == r1_idx) | (r2_idx == np.arange(pop_size))
            r2_idx[mask_conflict] = (r2_idx[mask_conflict] + 1) % union_size
            
            # Construct x_r2 vector
            x_r2 = np.empty((pop_size, dim))
            mask_in_pop = r2_idx < pop_size
            x_r2[mask_in_pop] = pop[r2_idx[mask_in_pop]]
            
            # Fetch from archive if index >= pop_size
            mask_in_arc = ~mask_in_pop
            if np.any(mask_in_arc):
                arc_indices = r2_idx[mask_in_arc] - pop_size
                x_r2[mask_in_arc] = archive[arc_indices]
            
            # Compute Mutant Vector V
            f_v = f[:, None] # Reshape for broadcasting
            mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
            
            # --- 3. Crossover (Binomial) ---
            mask_cross = np.random.rand(pop_size, dim) < cr[:, None]
            
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # --- 4. Bound Handling: Reflection ---
            # Instead of clipping, reflect the value back into the bound.
            # Formula: if x < min, new_x = min + (min - x) = 2*min - x
            
            # Check Lower Bounds
            mask_l = trial < min_b
            trial[mask_l] = 2 * min_b[mask_l] - trial[mask_l]
            # If still out after reflection (rare), clip
            mask_l_2 = trial < min_b
            trial[mask_l_2] = min_b[mask_l_2]
            
            # Check Upper Bounds
            mask_u = trial > max_b
            trial[mask_u] = 2 * max_b[mask_u] - trial[mask_u]
            # If still out, clip
            mask_u_2 = trial > max_b
            trial[mask_u_2] = max_b[mask_u_2]
            
            # --- 5. Selection & Adaptation Loop ---
            succ_f = []
            succ_cr = []
            succ_diff = []
            
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_val
                
                f_trial = func(trial[i])
                
                # Greedy Selection
                if f_trial < fitness[i]:
                    diff = fitness[i] - f_trial
                    
                    # Update Archive: Add parent to archive
                    if arc_count < archive_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Random replacement if full
                        k = np.random.randint(0, archive_size)
                        archive[k] = pop[i].copy()
                        
                    # Update Population
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    # Store Success Stats
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    succ_diff.append(diff)
                    
                    # Update Global Best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial[i].copy()
            
            # --- 6. SHADE Memory Update ---
            if succ_f:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                s_diff = np.array(succ_diff)
                
                # Weighted means based on fitness improvement amount
                w = s_diff / np.sum(s_diff)
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(w * s_cr)
                
                # Weighted Lehmer Mean for F
                mean_f_num = np.sum(w * (s_f ** 2))
                mean_f_den = np.sum(w * s_f)
                mean_f = mean_f_num / mean_f_den if mean_f_den > 0 else 0.5
                
                # Update memory
                m_cr[k_mem] = np.clip(mean_cr, 0.0, 1.0)
                m_f[k_mem] = np.clip(mean_f, 0.0, 1.0)
                
                k_mem = (k_mem + 1) % memory_size
        
        # End of Inner Loop (Convergence) -> Restart
        restart_count += 1
