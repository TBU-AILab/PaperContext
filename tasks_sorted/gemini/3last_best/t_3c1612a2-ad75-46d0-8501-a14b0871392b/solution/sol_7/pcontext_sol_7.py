#The best generated algorithm is an implementation of **Restart-SHADE (Success-History based Adaptive Differential Evolution)**.
#
#### Algorithm Description
#This approach refines the previous strategies by combining the robust adaptation mechanism of SHADE with a restart strategy to handle multimodal functions effectively within a limited time.
#
#1.  **SHADE Adaptation**: Uses a history memory ($H$) to store successful $F$ (Scaling Factor) and $CR$ (Crossover Rate) values. Unlike JADE's single mean, this allows the algorithm to maintain a diverse set of successful strategies, adapting to different phases of optimization (exploration vs. exploitation).
#2.  **External Archive**: Maintains a set of inferior solutions recently replaced by better ones. This allows the `current-to-pbest` mutation strategy to use difference vectors pointing away from previous positions, maintaining diversity and preventing premature convergence.
#3.  **Linear Restart Logic**: The algorithm monitors the population's standard deviation and fitness stagnation. If the population converges to a single point (low variance) or stops improving (stagnation), it triggers a restart with a new random population (preserving the global best), allowing it to escape local optima.
#4.  **Vectorized Efficiency**: The implementation maximizes the use of NumPy vector operations (e.g., generating parameters, mutations, and crossovers for the whole population at once) to ensure the maximum number of evaluations within `max_time`.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Establish strict timing to ensure result returns within max_time
    start_time = datetime.now()
    # Reserve a 5% buffer to safely return best result
    end_time = start_time + timedelta(seconds=max_time * 0.95)

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    # Population size: 15*dim is a balanced choice for SHADE variants.
    # We clamp it between [30, 150] to ensure speed for high dims and diversity for low dims.
    pop_size = int(15 * dim)
    pop_size = max(30, min(150, pop_size))
    
    # SHADE Memory size (History length)
    H = 10
    
    # -------------------------------------------------------------------------
    # Pre-processing
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    bound_diff = ub - lb
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    
    # -------------------------------------------------------------------------
    # Main Optimization Loop (Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check time before starting a new population restart
        if datetime.now() >= end_time:
            return best_fitness
            
        # --- Initialization Phase ---
        
        # Initialize Memory (H slots initialized to 0.5)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Population randomly
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize Archive
        # Stores inferior solutions replaced by offspring to preserve diversity
        archive = np.empty((pop_size, dim))
        arc_count = 0
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
        
        # Variables for stagnation detection
        last_best_fit = np.min(fitness)
        stagnation_count = 0
        
        # --- Evolution Loop ---
        while True:
            if datetime.now() >= end_time: return best_fitness
            
            # 1. Parameter Generation (SHADE Strategy)
            # Pick random index from memory for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR: Normal(M_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(M_f, 0.1). 
            # F must be > 0. If > 1, clip to 1.
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Correct invalid F values (<= 0) by regenerating
            neg_mask = f <= 0
            retry_count = 0
            while np.any(neg_mask) and retry_count < 10:
                num_neg = np.sum(neg_mask)
                f[neg_mask] = m_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(num_neg) - 0.5))
                neg_mask = f <= 0
                retry_count += 1
            
            # Fallback if F is still invalid
            if np.any(neg_mask):
                f[neg_mask] = np.random.rand(np.sum(neg_mask))
                
            f = np.minimum(f, 1.0)
            
            # 2. Mutation: current-to-pbest/1/bin with Archive
            # Select p-best (top 5%, minimum 2 individuals)
            p_num = max(2, int(0.05 * pop_size))
            sorted_indices = np.argsort(fitness)
            pbest_indices = sorted_indices[:p_num]
            # Randomly assign a pbest for each individual
            pbest_vectors = pop[np.random.choice(pbest_indices, pop_size)]
            
            # Select r1 (distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Fix collisions r1 == i
            col_mask = (r1_idx == np.arange(pop_size))
            r1_idx[col_mask] = (r1_idx[col_mask] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (distinct from i, r1) from Union(Population, Archive)
            union_size = pop_size + arc_count
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Retrieve x_r2 vectors
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2_idx < pop_size
            mask_arc = ~mask_pop
            
            x_r2[mask_pop] = pop[r2_idx[mask_pop]]
            if np.any(mask_arc):
                # Calculate archive indices
                arc_inds = r2_idx[mask_arc] - pop_size
                x_r2[mask_arc] = archive[arc_inds]
            
            # Calculate Mutant Vector: v = x + F*(pbest - x) + F*(r1 - r2)
            F_col = f[:, None]
            mutant = pop + F_col * (pbest_vectors - pop) + F_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_j = np.random.rand(pop_size, dim)
            cross_mask = rand_j < cr[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 4. Bound Handling (Clip)
            trial = np.clip(trial, lb, ub)
            
            # 5. Selection and Memory Update
            successful_scr = []
            successful_sf = []
            fitness_diffs = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                new_val = func(trial[i])
                
                # Greedy Selection
                if new_val <= fitness[i]:
                    # Update Archive
                    if arc_count < pop_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Random replacement if full
                        replace_idx = np.random.randint(0, pop_size)
                        archive[replace_idx] = pop[i].copy()
                    
                    # Collect data for parameter update
                    diff = fitness[i] - new_val
                    fitness_diffs.append(diff)
                    successful_scr.append(cr[i])
                    successful_sf.append(f[i])
                    
                    fitness[i] = new_val
                    pop[i] = trial[i]
                    
                    if new_val < best_fitness:
                        best_fitness = new_val
            
            # Update SHADE Memory (Weighted Lehmer Mean)
            if len(fitness_diffs) > 0:
                w = np.array(fitness_diffs)
                total_w = np.sum(w)
                if total_w == 0:
                    w = np.ones(len(w)) / len(w)
                else:
                    w = w / total_w
                
                # Update M_cr (Weighted Mean)
                m_cr_new = np.sum(w * np.array(successful_scr))
                mem_cr[k_mem] = m_cr_new
                
                # Update M_f (Weighted Lehmer Mean)
                sf_arr = np.array(successful_sf)
                mean_lehmer = np.sum(w * sf_arr**2) / (np.sum(w * sf_arr) + 1e-15)
                mem_f[k_mem] = mean_lehmer
                
                k_mem = (k_mem + 1) % H
            
            # 6. Restart Logic
            current_best_in_pop = np.min(fitness)
            pop_std = np.std(fitness)
            
            # Check for Stagnation (no improvement in best fitness)
            if abs(current_best_in_pop - last_best_fit) < 1e-8:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best_in_pop
            
            # Trigger Restart if:
            # - Population variance is low (converged)
            # - No improvement for 35 generations
            if pop_std < 1e-6 or stagnation_count > 35:
                break
    
    return best_fitness
