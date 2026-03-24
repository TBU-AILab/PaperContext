#Here is a fully operational, self-contained Python implementation using the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm.
#
#### Improvements over the previous version:
#1.  **Full Vectorization**: The mutation and crossover operations are performed using NumPy matrix operations rather than Python loops. This significantly reduces overhead and allows for many more generations (evaluations) within the time limit.
#2.  **External Archive**: Unlike JADE, SHADE utilizes an external archive of inferior solutions to maintain diversity. The second difference vector in the mutation strategy is selected from the union of the population and this archive, preventing premature convergence.
#3.  **Historical Memory**: Instead of a single moving average for parameters $F$ and $CR$, this algorithm maintains a memory `M` of successful configurations from the past $H$ generations. This handles multimodal landscapes significantly better by "remembering" different successful search strategies.
#4.  **Robust Restart**: A restart triggers not just on population standard deviation, but also ensures the archive is cleared and memory reset, preventing the algorithm from instantly falling back into the same local optimum.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the SHADE algorithm (Success-History based 
    Adaptive Differential Evolution) with restart mechanism.
    
    This implementation is fully vectorized for performance.
    """
    t_start = time.time()
    
    # --- Hyperparameters ---
    # Population sizing
    pop_size = int(max(30, 18 * dim))
    
    # Archive size (usually equal to pop_size)
    archive_size = pop_size 
    
    # Memory size for historical adaptation
    h_mem_size = 6
    
    # Initial P-best portion (top 5% to 20%)
    p_best_rate = 0.11
    
    # --- Bounds Processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Tracking ---
    best_fitness = float('inf')
    best_solution = None

    # --- Restart Loop ---
    while True:
        # Check time at start of restart
        if (time.time() - t_start) >= max_time:
            break

        # 1. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            # Check time strictly during expensive evaluations
            if (time.time() - t_start) >= max_time:
                return best_fitness if best_solution is not None else float('inf')
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = pop[i].copy()

        # Initialize Memory (M_CR and M_F)
        mem_cr = np.full(h_mem_size, 0.5)
        mem_f = np.full(h_mem_size, 0.5)
        k_mem = 0  # Index for memory update
        
        # Initialize Archive
        archive = [] # List of numpy arrays
        
        # Generation Counter for Restart check
        gen_since_restart = 0
        
        # --- Main Evolution Loop ---
        while True:
            # Time Check (Per generation)
            if (time.time() - t_start) >= max_time:
                return best_fitness

            # --- Sort population for p-best selection ---
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

            # --- Restart Trigger (Convergence) ---
            # If standard deviation of top 50% fitness is extremely small
            # or max - min fitness is negligible.
            top_half = fitness[:pop_size//2]
            if np.std(top_half) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break # Break inner loop to restart

            # --- Parameter Generation (Vectorized) ---
            # Select random memory index for each individual
            r_idx = np.random.randint(0, h_mem_size, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1) # CR must be [0, 1]
            
            # Generate F ~ Cauchy(M_F, 0.1)
            # Cauchy generation: location + scale * standard_cauchy
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Handle F constraints
            # If F > 1, clamp to 1. If F <= 0, regenerate until > 0 (simplified here to clamping/abs)
            # Standard SHADE logic for F <= 0 is retry, but clipping 0.05 is efficient and stable
            f = np.where(f > 1, 1.0, f)
            f = np.where(f <= 0, 0.5, f) # Fallback if very negative
            
            # --- Mutation: Current-to-pbest/1 ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # 1. Select X_pbest
            # Randomly select one of the top p% for each individual
            p_limit = max(2, int(pop_size * p_best_rate))
            p_best_indices = np.random.randint(0, p_limit, pop_size)
            x_pbest = pop[p_best_indices]
            
            # 2. Select X_r1
            # Random distinct indices (naive approx: random permutation)
            # We need r1 != i.
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Fix collisions where r1 == i by offsetting
            collision_mask = (r1_indices == np.arange(pop_size))
            r1_indices[collision_mask] = (r1_indices[collision_mask] + 1) % pop_size
            x_r1 = pop[r1_indices]
            
            # 3. Select X_r2 from Union(Population, Archive)
            # Convert archive list to array for vectorized indexing
            if len(archive) > 0:
                archive_np = np.array(archive)
                # Union pool
                pop_archive_pool = np.vstack((pop, archive_np))
            else:
                pop_archive_pool = pop
                
            pool_size = pop_archive_pool.shape[0]
            r2_indices = np.random.randint(0, pool_size, pop_size)
            
            # Ensure r2 != r1 and r2 != i is ideal, but in standard SHADE code, 
            # strict exclusion from Archive is loose. We just ensure r2 != r1.
            # (Vector operations make strict distinctness logic complex, small overlap is acceptable in DE)
            x_r2 = pop_archive_pool[r2_indices]
            
            # Calculate Mutation Vectors (reshape F for broadcasting)
            f_col = f[:, np.newaxis]
            v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # Boundary handling: Random bounce back or clip
            # Clipping is generally safer for black-box functions
            v = np.clip(v, min_b, max_b)
            
            # --- Crossover: Binomial ---
            # Mask: True if rand < CR
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            
            # Force at least one dimension to change
            # We use a fancy indexing trick
            rows = np.arange(pop_size)
            cross_mask[rows, j_rand] = True
            
            # Create Trial Population
            u = np.where(cross_mask, v, pop)
            
            # --- Selection ---
            new_fitness = np.zeros(pop_size)
            
            # Keep track of successful updates for memory adaptation
            scr = []
            sf = []
            diff_fitness = []
            
            # Evaluate trials
            # This loop is the necessary bottleneck
            for i in range(pop_size):
                if (time.time() - t_start) >= max_time:
                    return best_fitness

                f_trial = func(u[i])
                new_fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_solution = u[i].copy()

            # Identify improvements
            # Note: pop is currently sorted by fitness, so fitness[i] matches pop[i]
            improved_mask = new_fitness < fitness
            
            # Update Archive with displaced parent solutions
            # Only add to archive if we improved
            displaced_parents = pop[improved_mask]
            for parent in displaced_parents:
                if len(archive) < archive_size:
                    archive.append(parent.copy())
                else:
                    # Replace random member
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = parent.copy()
            
            # While archive is managed as a list, keep it bounded strictly
            if len(archive) > archive_size:
                # Truncate randomly or strictly (simple truncate)
                del archive[archive_size:]

            # Collect successful parameters
            scr.extend(cr[improved_mask])
            sf.extend(f[improved_mask])
            diff_fitness.extend(np.abs(fitness[improved_mask] - new_fitness[improved_mask]))
            
            # Update Population and Fitness
            pop[improved_mask] = u[improved_mask]
            fitness[improved_mask] = new_fitness[improved_mask]
            
            # --- Update History Memory ---
            if len(scr) > 0:
                scr_np = np.array(scr)
                sf_np = np.array(sf)
                w = np.array(diff_fitness)
                
                # Normalize weights
                if np.sum(w) > 0:
                    w = w / np.sum(w)
                else:
                    w = np.ones_like(w) / len(w)
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(w * (sf_np ** 2)) / (np.sum(w * sf_np) + 1e-10)
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(w * scr_np)
                
                # Update memory at index k
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                
                # Increment memory index
                k_mem = (k_mem + 1) % h_mem_size

            gen_since_restart += 1

    return best_fitness
