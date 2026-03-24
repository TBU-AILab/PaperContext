#Here is a fully operational, self-contained implementation of the **L-SHADE algorithm (Linear Success-History based Adaptive Differential Evolution) with an External Archive**.
#
#### Improvements over the previous JADE implementation:
#
#1.  **Linear Population Size Reduction (LPR):** The algorithm starts with a large population to maximize exploration and linearly reduces the population size over time. This forces the algorithm to shift from exploration to exploitation naturally as time runs out, dramatically improving convergence speed in the final stages.
#2.  **Historical Memory ($H$):** Instead of simple running averages for mutation/crossover parameters, this algorithm maintains a history of successful parameter sets. This allows the algorithm to "remember" different search strategies that worked in specific phases of optimization (e.g., high mutation early, low mutation late).
#3.  **External Archive:** Rejected solutions (those replaced by better offspring) are stored in a limited-size archive. The mutation operator draws vectors from both the current population *and* this archive. This maintains diversity without slowing down convergence, preventing stagnation in local optima.
#4.  **Weighted Lehmer Mean:** Parameter updates use fitness-weighted means, giving more influence to parameter settings that resulted in larger fitness improvements.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE Implementation: Linear Success-History Adaptive Differential Evolution
    with External Archive and Linear Population Reduction.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    def get_remaining_ratio():
        """Returns 0.0 to 1.0 representing time progress."""
        now = datetime.now()
        if now >= end_time:
            return 0.0
        total_seconds = max_time
        elapsed = (now - start_time).total_seconds()
        return max(0.0, 1.0 - (elapsed / total_seconds))

    # --- Configuration ---
    # Initial Population: Start larger to explore, shrink later
    # L-SHADE typically suggests N_init around 18 * dim
    pop_size_init = int(round(18 * dim))
    # Clip specific bounds for performance safety
    pop_size_init = np.clip(pop_size_init, 30, 200) 
    pop_size_min = 4
    
    pop_size = pop_size_init
    
    # Memory for adaptive parameters (History size H)
    memory_size = 5 
    mem_idx = 0
    m_cr = np.full(memory_size, 0.5) # Memory for Crossover Rate
    m_f = np.full(memory_size, 0.5)  # Memory for Mutation Factor
    
    # External Archive (stores rejected solutions to maintain diversity)
    archive = []
    archive_factor = 2.0 # Archive size relative to pop_size
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Initialization ---
    # Latin Hypercube Sampling for initial population
    population = np.empty((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(0, 1, pop_size + 1)
        offsets = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(offsets)
        population[:, d] = min_b[d] + offsets * diff_b[d]

    fitness = np.full(pop_size, float('inf'))
    
    # Global Best Tracking
    best_val = float('inf')
    best_vec = None

    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_val if best_val != float('inf') else float('inf')
            
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()

    # --- Main Optimization Loop ---
    while True:
        # Check time
        time_ratio = get_remaining_ratio()
        if time_ratio <= 0:
            return best_val

        # 1. Linear Population Size Reduction (LPR)
        # Calculate target population size based on time remaining
        plan_pop_size = int(round((pop_size_min - pop_size_init) * (1 - time_ratio) + pop_size_init))
        
        if pop_size > plan_pop_size:
            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            fitness = fitness[sorted_indices]
            population = population[sorted_indices]
            
            # Truncate population (remove worst)
            reduction_amt = pop_size - plan_pop_size
            # Limit reduction to prevent removing too many at once if time skips
            if reduction_amt > 0:
                pop_size = plan_pop_size
                population = population[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize archive to match new pop_size * factor
                max_archive_size = int(pop_size * archive_factor)
                if len(archive) > max_archive_size:
                    # Randomly remove elements from archive to shrink it
                    del_indices = np.random.choice(len(archive), len(archive) - max_archive_size, replace=False)
                    # Use list comprehension for safe removal
                    archive = [arr for idx, arr in enumerate(archive) if idx not in del_indices]

        # 2. Sort population (needed for current-to-pbest)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Sync best (index 0 after sort)
        if fitness[0] < best_val:
            best_val = fitness[0]
            best_vec = population[0].copy()

        # 3. Generate Parameters based on History
        # Randomly select a memory index for each individual
        r_indices = np.random.randint(0, memory_size, pop_size)
        mu_cr = m_cr[r_indices]
        mu_f = m_f[r_indices]
        
        # Generate CR (Normal Distribution)
        # cr ~ N(mu_cr, 0.1), clipped [0, 1]
        cr_vals = np.random.normal(mu_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        # Inherit -1 handling from standard L-SHADE (if CR=-1, set to 0, usually rare with clip)

        # Generate F (Cauchy Distribution)
        # f ~ Cauchy(mu_f, 0.1)
        f_vals = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints
        # If F > 1, cap at 1. If F <= 0, regenerate until > 0 (or clamp)
        # For efficiency, we just clamp low to 0.1 to ensure mutation occurs
        f_vals = np.clip(f_vals, 0.1, 1.0)

        # 4. Mutation: current-to-pbest/1 (with Archive)
        # V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
        
        # p-best selection: top p% members (p varies from 0.05 to 0.2 depending on implementation)
        p_val = max(2, int(pop_size * 0.11)) # Top 11%
        pbest_indices = np.random.randint(0, p_val, pop_size)
        x_pbest = population[pbest_indices]
        
        # r1 selection: random from population (distinct from i)
        # Simplified: random from pop
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # r2 selection: random from Union(Population, Archive)
        # Combine current population and archive for the second difference vector
        if len(archive) > 0:
            archive_np = np.array(archive)
            combined_pool = np.vstack((population, archive_np))
        else:
            combined_pool = population
            
        r2_indices = np.random.randint(0, len(combined_pool), pop_size)
        x_r2 = combined_pool[r2_indices]
        
        # Calculate Mutation Vectors
        f_col = f_vals[:, np.newaxis]
        mutants = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_matrix = np.random.rand(pop_size, dim)
        cross_mask = rand_matrix < cr_vals[:, np.newaxis]
        # Force at least one dimension
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(cross_mask, mutants, population)
        
        # Boundary Correction: Midpoint handling
        # Instead of clipping, if a value exceeds bounds, move it to (bound + old)/2
        # This preserves diversity better than hard clipping.
        lower_viol = trials < min_b
        upper_viol = trials > max_b
        
        # If violates lower: (min_b + old) / 2
        trials[lower_viol] = (min_b[np.where(lower_viol)[1]] + population[lower_viol]) / 2.0
        # If violates upper: (max_b + old) / 2
        trials[upper_viol] = (max_b[np.where(upper_viol)[1]] + population[upper_viol]) / 2.0
        
        # 6. Selection & Memory Update
        success_f = []
        success_cr = []
        diff_fitness = []
        
        # Evaluate Offspring
        # We loop here to check time frequently
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val
            
            t_fit = func(trials[i])
            
            # Greedy Selection
            if t_fit <= fitness[i]: # Better or equal
                # Success
                if t_fit < fitness[i]:
                    # Keep removed solution in archive
                    archive.append(population[i].copy())
                    
                    # Record for adaptive update
                    success_f.append(f_vals[i])
                    success_cr.append(cr_vals[i])
                    diff_fitness.append(fitness[i] - t_fit)
                
                # Update population
                fitness[i] = t_fit
                population[i] = trials[i]
                
                if t_fit < best_val:
                    best_val = t_fit
                    best_vec = trials[i].copy()
                    
        # Maintain Archive Size
        max_archive_size = int(pop_size * archive_factor)
        while len(archive) > max_archive_size:
            # Remove random elements if archive overflows
            idx_remove = np.random.randint(0, len(archive))
            archive.pop(idx_remove)
            
        # 7. Update History Memory (Weighted Lehmer Mean)
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            d_fit = np.array(diff_fitness)
            
            # Weights based on fitness improvement
            total_imp = np.sum(d_fit)
            if total_imp > 0:
                weights = d_fit / total_imp
            else:
                weights = np.ones(len(d_fit)) / len(d_fit)
            
            # Update Memory for F (Lehmer Mean)
            # mean = sum(w * f^2) / sum(w * f)
            numer_f = np.sum(weights * (s_f ** 2))
            denom_f = np.sum(weights * s_f)
            new_f = numer_f / (denom_f + 1e-10)
            
            # Update Memory for CR (Weighted Mean)
            new_cr = np.sum(weights * s_cr)
            
            # Update the specific memory slot
            # If current memory is 'empty' (-1 or default), replace strictly
            # otherwise, blend slightly (though SHADE usually strictly replaces)
            m_f[mem_idx] = new_f
            m_cr[mem_idx] = new_cr
            
            # Rotate index
            mem_idx = (mem_idx + 1) % memory_size
            
            # Ensure CR memory is valid (sometimes mean can drift close to 0 or 1)
            m_cr = np.clip(m_cr, 0, 1)
            m_f = np.clip(m_f, 0, 1) # F is typically kept around 0.5, but let it adapt

    return best_val
