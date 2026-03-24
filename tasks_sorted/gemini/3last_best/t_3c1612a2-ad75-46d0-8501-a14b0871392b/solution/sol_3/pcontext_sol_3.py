#The best generated algorithm code is improved below.
#
#### Algorithm Description
#The proposed algorithm is an implementation of **SHADE (Success-History based Adaptive Differential Evolution) with a Restart Mechanism**.
#
#**Key Improvements:**
#1.  **External Archive**: Unlike the previous JADE-like implementation, this algorithm maintains an **external archive** of recently inferior solutions. This archive is used during the mutation step to preserve diversity, preventing the population from converging too quickly into a local minimum.
#2.  **Success-History Adaptation**: Instead of a single adaptive parameter, it uses a **History Memory ($H$)** to store successful $F$ (Scaling Factor) and $CR$ (Crossover Rate) values from previous generations. This allows the algorithm to learn the landscape properties more effectively over time.
#3.  **Weighted Lehmer Mean**: The parameter update logic uses weighted means based on fitness improvement magnitudes, giving more weight to parameter values that produced significant improvements.
#4.  **Robust Restart Strategy**: The algorithm monitors population variance and stagnation. If the search converges or gets stuck, it triggers a restart while preserving the global best found so far.
#
#### Code
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Establish strict timing to ensure result returns within max_time
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Hyperparameters (SHADE-based configuration)
    # -------------------------------------------------------------------------
    # Population size: A value around 18*dim is robust for SHADE variants.
    # We ensure a minimum size of 30 to support statistical diversity.
    pop_size = max(30, int(18 * dim)) 
    
    # SHADE Memory parameters
    H = 5                      # History memory size
    mem_cr = np.full(H, 0.5)   # Memory for Crossover Rate (initialized to 0.5)
    mem_f = np.full(H, 0.5)    # Memory for Scaling Factor (initialized to 0.5)
    k_mem = 0                  # Memory cyclic index

    # Pre-process bounds for vectorization
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    bound_width = ub - lb
    
    # Global tracker for the best solution found across all restarts
    global_best_val = float('inf')
    
    # -------------------------------------------------------------------------
    # Main Optimization Loop (Handles Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check time before starting a new restart
        if datetime.now() >= end_time:
            return global_best_val

        # --- Initialization Phase ---
        # Initialize random population
        pop = lb + np.random.rand(pop_size, dim) * bound_width
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize Archive
        # The archive stores inferior solutions to maintain diversity.
        # Max size equals pop_size. We use a numpy array and a counter.
        archive = np.empty((pop_size, dim)) 
        arc_count = 0
        
        # Evaluate initial population
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Variables to track stagnation for restart logic
        stagnation_count = 0
        last_best_fit = np.min(fitness)
        
        # --- Generation Loop ---
        while True:
            if datetime.now() >= end_time: return global_best_val
            
            # 1. Parameter Adaptation (SHADE Strategy)
            # Select a random index from memory for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr_selected = mem_cr[r_idx]
            m_f_selected = mem_f[r_idx]
            
            # Generate CR: Normal distribution around memory value, clipped to [0, 1]
            cr = np.random.normal(m_cr_selected, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy distribution. F must be > 0.
            # If F > 1, clamp to 1. If F <= 0, regenerate.
            f = m_f_selected + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Vectorized retry for invalid F values (<= 0)
            bad_f_mask = f <= 0
            while np.any(bad_f_mask):
                num_bad = np.sum(bad_f_mask)
                f[bad_f_mask] = m_f_selected[bad_f_mask] + 0.1 * np.tan(np.pi * (np.random.rand(num_bad) - 0.5))
                bad_f_mask = f <= 0
            
            f = np.minimum(f, 1.0)
            
            # 2. Mutation: current-to-pbest/1/bin with Archive
            # Strategy: v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            
            # Select p-best individuals (top 5% of current generation)
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(0.05 * pop_size))
            pbest_indices = sorted_indices[:num_pbest]
            # Randomly assign a p-best to each individual
            pbest_vectors = pop[np.random.choice(pbest_indices, pop_size)]
            
            # Select r1: Random individuals from population, distinct from current (i)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Handle collision (r1 == i) by shifting index
            collision_mask = (r1_indices == np.arange(pop_size))
            r1_indices[collision_mask] = (r1_indices[collision_mask] + 1) % pop_size
            x_r1 = pop[r1_indices]
            
            # Select r2: Random individuals from Union(Population, Archive), distinct from i and r1
            # For efficiency, we pick randomly and ignore rare collisions
            union_size = pop_size + arc_count
            r2_indices = np.random.randint(0, union_size, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            
            # Map indices: < pop_size -> Population, >= pop_size -> Archive
            from_pop = r2_indices < pop_size
            from_arc = ~from_pop
            
            x_r2[from_pop] = pop[r2_indices[from_pop]]
            if np.any(from_arc):
                arc_idx = r2_indices[from_arc] - pop_size
                x_r2[from_arc] = archive[arc_idx]
                
            # Compute Mutant Vectors (Vectorized)
            F_col = f.reshape(-1, 1)
            mutants = pop + F_col * (pbest_vectors - pop) + F_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr.reshape(-1, 1)
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trials = np.where(cross_mask, mutants, pop)
            
            # 4. Boundary Constraint
            trials = np.clip(trials, lb, ub)
            
            # 5. Selection and Memory Update
            successful_scr = []
            successful_sf = []
            fitness_diffs = []
            parents_to_archive = []
            
            # Evaluation loop (Sequential due to func interface)
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                trial_fitness = func(trials[i])
                
                # Greedy Selection
                if trial_fitness < fitness[i]:
                    diff = fitness[i] - trial_fitness
                    
                    # Record success data
                    successful_scr.append(cr[i])
                    successful_sf.append(f[i])
                    fitness_diffs.append(diff)
                    
                    # Store old parent for archive
                    parents_to_archive.append(pop[i].copy())
                    
                    # Update Population
                    fitness[i] = trial_fitness
                    pop[i] = trials[i]
                    
                    # Update Global Best
                    if trial_fitness < global_best_val:
                        global_best_val = trial_fitness
            
            # Update Archive
            for parent in parents_to_archive:
                if arc_count < pop_size:
                    archive[arc_count] = parent
                    arc_count += 1
                else:
                    # Random replacement if archive is full
                    replace_idx = np.random.randint(0, pop_size)
                    archive[replace_idx] = parent
            
            # Update History Memory (Weighted Lehmer Mean)
            if len(fitness_diffs) > 0:
                s_scr = np.array(successful_scr)
                s_sf = np.array(successful_sf)
                w = np.array(fitness_diffs)
                
                total_w = np.sum(w)
                if total_w > 0:
                    w = w / total_w
                    
                    # Update Memory CR (Weighted Mean)
                    m_cr_new = np.sum(w * s_scr)
                    mem_cr[k_mem] = m_cr_new
                    
                    # Update Memory F (Weighted Lehmer Mean)
                    mean_f_denom = np.sum(w * s_sf)
                    if mean_f_denom > 1e-12:
                        m_f_new = np.sum(w * s_sf**2) / mean_f_denom
                        mem_f[k_mem] = m_f_new
                    
                    # Advance memory index
                    k_mem = (k_mem + 1) % H
            
            # 6. Restart Logic
            current_best = np.min(fitness)
            pop_std = np.std(fitness)
            
            # Check for stagnation (no improvement in best fitness)
            if abs(current_best - last_best_fit) < 1e-8:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best
            
            # Trigger Restart if:
            # 1. Population variance is extremely low (convergence)
            # 2. Stagnation persists for > 30 generations
            if pop_std < 1e-6 or stagnation_count > 30:
                break # Break inner loop to trigger restart in outer loop

    return global_best_val
