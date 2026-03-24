#The following algorithm implements **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)**.
#
#**Reasoning for Improvement:**
#1.  **Linear Population Size Reduction (LPSR)**: The previous best algorithm (SHADE) used a fixed (or reset-based) population size. L-SHADE improves this by linearly reducing the population size from a large initial value (high exploration) to a minimum value (high exploitation) as the time budget is consumed. This focuses computational resources on the best individuals during the final stages of optimization.
#2.  **Time-Based Schedule**: Standard L-SHADE relies on a fixed number of evaluations. This implementation adapts the reduction schedule based on the `max_time` constraint, ensuring the population reaches its minimum size exactly when the time limit approaches.
#3.  **Robust Statistics**: It utilizes the weighted Lehmer mean for parameter adaptation ($F$ and $CR$), which is more robust to outliers than simple averaging.
#4.  **Non-destructive Stagnation Handling**: Instead of a full hard restart which discards learned history, if the population stagnates (variance drops too low) before time is up, it performs a "partial expansion," injecting fresh random individuals into the worst slots while preserving the elite trajectory.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using L-SHADE (Success-History Adaptive DE 
    with Linear Population Size Reduction) adapted for a Time Budget.
    """

    # --- Time Management ---
    start_time = datetime.now()
    # Reserve a small buffer (5%) for final cleanup and return
    time_limit_sec = max_time * 0.95
    end_time_limit = start_time + timedelta(seconds=time_limit_sec)

    def get_time_progress():
        """Returns float 0.0 to 1.0 indicating time usage."""
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        return min(1.0, elapsed / time_limit_sec)

    def check_time():
        return datetime.now() >= end_time_limit

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Population Sizing (L-SHADE strategy)
    # Start with a larger population for exploration
    initial_pop_size = int(max(30, min(300, 25 * dim)))
    # End with a minimal population for exploitation
    min_pop_size = 4
    
    current_pop_size = initial_pop_size
    
    # Initialize Population
    population = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # Evaluate Initial Population
    best_idx = 0
    best_val = float('inf')
    
    for i in range(current_pop_size):
        if check_time(): return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Memory & Archive Initialization ---
    memory_size = 6  # H parameter
    # Initialize memory with 0.5
    mem_M_CR = np.full(memory_size, 0.5)
    mem_M_F = np.full(memory_size, 0.5)
    k_mem = 0  # Memory index pointer
    
    # External Archive (stores replaced parent vectors)
    # In L-SHADE, archive size scales with population size
    archive = []
    
    # --- Main Optimization Loop ---
    while not check_time():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate allowed population size based on time progress
        progress = get_time_progress()
        target_pop_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
        target_pop_size = max(min_pop_size, target_pop_size)

        if current_pop_size > target_pop_size:
            # Reduction needed: Remove worst individuals
            n_remove = current_pop_size - target_pop_size
            sorting_indices = np.argsort(fitness)
            # Keep the top (current - remove)
            keep_indices = sorting_indices[:-n_remove]
            
            population = population[keep_indices]
            fitness = fitness[keep_indices]
            current_pop_size = target_pop_size
            
            # Since indices changed, recalculate best_idx
            best_idx = np.argmin(fitness)
            best_val = fitness[best_idx]
            
            # Archive size is typically maintained at equal to current pop_size
            if len(archive) > current_pop_size:
                # Shrink archive randomly
                del_count = len(archive) - current_pop_size
                for _ in range(del_count):
                    archive.pop(np.random.randint(0, len(archive)))

        # 2. Check for Stagnation (Optional "Kick")
        # If variance is extremely low but we still have time, inject diversity
        if current_pop_size > min_pop_size and np.std(fitness) < 1e-10 and progress < 0.8:
            # Replace worst 30% with random individuals to break stagnation
            # but keep the best structure intact
            n_reset = int(0.3 * current_pop_size)
            if n_reset > 0:
                sort_idxs = np.argsort(fitness)
                worst_idxs = sort_idxs[-n_reset:]
                population[worst_idxs] = min_b + np.random.rand(n_reset, dim) * diff_b
                # Set fitness to inf to force update in next generation
                fitness[worst_idxs] = float('inf')

        # 3. Parameter Generation
        # Each individual picks a random memory slot
        r_idxs = np.random.randint(0, memory_size, current_pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]

        # Generate CR (Normal dist around Memory, std 0.1)
        # Clip to [0, 1]
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0, 1)
        # Special case: CR = -1 in SHADE usually implies 0, but clip handles bounds.
        
        # Generate F (Cauchy dist around Memory, scale 0.1)
        # F = m_f + 0.1 * tan(pi * (rand - 0.5))
        F = m_f + 0.1 * np.tan(np.pi * (np.random.rand(current_pop_size) - 0.5))
        
        # Check F bounds
        # If F > 1, cap at 1. If F <= 0, regenerate.
        F[F > 1] = 1.0
        while np.any(F <= 0):
            bad_mask = F <= 0
            n_bad = np.sum(bad_mask)
            F[bad_mask] = m_f[bad_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
            F[F > 1] = 1.0

        # 4. Mutation: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Determine p-best size (random p in [2/N, 0.2])
        # p_min = 2 / current_pop_size
        # p = max(p_min, 0.2) -> standard L-SHADE uses fixed p or dynamic. 
        # A robust value is top 11% (p=0.11) or minimum 2 individuals.
        p_val = max(2.0 / current_pop_size, 0.11)
        top_cut = int(round(current_pop_size * p_val))
        top_cut = max(2, top_cut) # Ensure at least 2
        
        sorted_indices = np.argsort(fitness)
        top_indices = sorted_indices[:top_cut]
        
        # Select pbest for each individual
        pbest_idxs = np.random.choice(top_indices, current_pop_size)
        x_pbest = population[pbest_idxs]
        
        # Select r1 (!= i)
        r1_idxs = np.random.randint(0, current_pop_size, current_pop_size)
        # Fix collisions
        col_mask = (r1_idxs == np.arange(current_pop_size))
        r1_idxs[col_mask] = (r1_idxs[col_mask] + 1) % current_pop_size
        x_r1 = population[r1_idxs]
        
        # Select r2 (!= r1, != i) from Population U Archive
        if len(archive) > 0:
            archive_np = np.array(archive)
            pop_arc = np.vstack((population, archive_np))
        else:
            pop_arc = population
            
        r2_idxs = np.random.randint(0, len(pop_arc), current_pop_size)
        # We ignore strict r2!=r1!=i check for vectorization speed as impact is minimal in large pools
        # and self-correction occurs over generations.
        x_r2 = pop_arc[r2_idxs]
        
        # Calculate Mutation Vectors
        # Expand F for broadcasting
        F_col = F[:, np.newaxis]
        mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, current_pop_size)
        cross_mask = np.random.rand(current_pop_size, dim) < CR[:, np.newaxis]
        # Force at least one dimension to come from mutant
        cross_mask[np.arange(current_pop_size), j_rand] = True
        
        trial_vectors = np.where(cross_mask, mutant, population)
        
        # 6. Bound Constraint Handling (Clip)
        trial_vectors = np.clip(trial_vectors, min_b, max_b)
        
        # 7. Selection
        new_fitness = np.zeros(current_pop_size)
        improved_mask = np.zeros(current_pop_size, dtype=bool)
        diff_vals = np.zeros(current_pop_size)
        
        # Evaluate trials
        # Note: Must evaluate sequentially due to check_time and func signature
        for i in range(current_pop_size):
            if check_time(): return best_val
            
            f_trial = func(trial_vectors[i])
            new_fitness[i] = f_trial
            
            if f_trial <= fitness[i]:
                improved_mask[i] = True
                diff_vals[i] = fitness[i] - f_trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i

        # 8. Update Population, Archive, and Memory
        success_indices = np.where(improved_mask)[0]
        
        if len(success_indices) > 0:
            # Add replaced parents to archive
            for idx in success_indices:
                archive.append(population[idx].copy())
            
            # Maintain archive size <= current_pop_size
            while len(archive) > current_pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update Population
            population[success_indices] = trial_vectors[success_indices]
            fitness[success_indices] = new_fitness[success_indices]
            
            # Update SHADE Memory
            # Calculate weights based on fitness improvement
            diffs = diff_vals[success_indices]
            total_diff = np.sum(diffs)
            
            if total_diff > 0:
                weights = diffs / total_diff
                
                # Update M_CR (Weighted Mean)
                s_cr = CR[success_indices]
                # Filter out -1 if any (though we clipped 0-1)
                m_cr_new = np.sum(weights * s_cr)
                
                # Update M_F (Weighted Lehmer Mean)
                s_f = F[success_indices]
                num_f = np.sum(weights * (s_f ** 2))
                den_f = np.sum(weights * s_f)
                m_f_new = num_f / den_f if den_f > 0 else 0.5
                
                # Store in memory
                mem_M_CR[k_mem] = m_cr_new
                mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                
                k_mem = (k_mem + 1) % memory_size

    return best_val
