#The following is a fully operational, self-contained implementation of an advanced optimization algorithm.
#
#### Algorithm Explanation
#This algorithm is an improved **L-SHADE (Linear Success-History Adaptive Differential Evolution) with Restart**. It is engineered to perform exceptionally well under limited time constraints by combining the robust adaptivity of L-SHADE with an aggressive restart mechanism and efficient vectorized operations.
#
#**Key Features:**
#1.  **L-SHADE Adaptation**: It utilizes a historical memory of success parameters ($F$ and $CR$) to adapt to the objective function's landscape dynamically. This removes the need for manual parameter tuning.
#2.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases from a larger initial size (for exploration) to a minimal size (for fast exploitation) as time progresses. This ensures the algorithm does not waste computational resources on a large population when time is running out.
#3.  **Vectorized Operations**: The core evolutionary operators (mutation, crossover, boundary handling) are fully vectorized using NumPy, significantly reducing Python interpreter overhead compared to loop-based implementations.
#4.  **Midpoint-Target Bound Handling**: When a solution violates boundaries, instead of clipping (which accumulates points at the edge), it is reset to the midpoint between the parent and the bound. This maintains population diversity and valid search directions.
#5.  **Stagnation Detection & Restart**: If the population converges (variance becomes negligible) or stagnates, the algorithm triggers a soft restart. It preserves the best solution found so far but re-initializes the rest of the population to explore new basins of attraction.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Optimized L-SHADE with Linear Population Size Reduction
    and Stagnation Restart.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Initial and final population sizes
    # Scaled by dimension, but clipped to ensure speed within time limits
    min_pop_size = 4
    max_pop_size = int(np.clip(dim * 18, 30, 90))
    
    # SHADE Memory parameters
    memory_size = 6
    
    # Extract bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Helper: Boundary Constraints ---
    def apply_bound_constraints(trial_vecs, parent_vecs):
        # Midpoint-target strategy:
        # If out of bounds, place between parent and bound.
        # This preserves diversity better than simple clipping.
        
        # Lower bound violations
        viol_l = trial_vecs < min_b
        if np.any(viol_l):
            # trial = (min + parent) / 2
            target = (min_b + parent_vecs) * 0.5
            trial_vecs[viol_l] = target[viol_l]
            
        # Upper bound violations
        viol_u = trial_vecs > max_b
        if np.any(viol_u):
            # trial = (max + parent) / 2
            target = (max_b + parent_vecs) * 0.5
            trial_vecs[viol_u] = target[viol_u]
            
        return trial_vecs

    # --- Initialization ---
    pop_size = max_pop_size
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initial Evaluation
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # Sort population (required for p-best strategy)
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]
    
    # Initialize Memory
    M_CR = np.full(memory_size, 0.5)
    M_F = np.full(memory_size, 0.5)
    k_mem = 0
    
    # Initialize Archive
    archive = np.empty((0, dim))
    
    # --- Main Loop ---
    while True:
        # Check overall time
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            return best_val
            
        # 1. Linear Population Size Reduction (LPSR)
        # Reduce population size based on time elapsed
        progress = elapsed / max_time
        target_size = int(round((min_pop_size - max_pop_size) * progress + max_pop_size))
        target_size = max(min_pop_size, target_size)
        
        if pop_size > target_size:
            # Shrink population: remove worst individuals (end of sorted array)
            pop = pop[:target_size]
            fitness = fitness[:target_size]
            pop_size = target_size
            
            # Resize archive if it exceeds new population size
            if len(archive) > pop_size:
                n_del = len(archive) - pop_size
                del_idx = np.random.choice(len(archive), n_del, replace=False)
                archive = np.delete(archive, del_idx, axis=0)

        # 2. Parameter Generation
        # Randomly select memory slots
        r_idx = np.random.randint(0, memory_size, pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # Generate CR (Normal distribution, clipped [0, 1])
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F (Cauchy distribution, regenerate if <= 0, clip at 1)
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        F = np.minimum(F, 1.0)
        
        # Vectorized fix for F <= 0
        neg_mask = F <= 0
        while np.any(neg_mask):
            n_neg = np.sum(neg_mask)
            F[neg_mask] = m_f[neg_mask] + 0.1 * np.random.standard_cauchy(n_neg)
            F = np.minimum(F, 1.0)
            neg_mask = F <= 0
            
        # 3. Mutation: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Select p-best indices (top p%, where p is random in [2/N, 0.2])
        p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
        p_best_idx = (p_val * pop_size).astype(int)
        p_best_idx = np.clip(p_best_idx, 0, pop_size - 1)
        x_pbest = pop[p_best_idx]
        
        # Select r1 (random from pop, distinct from current not strictly enforced for speed but collisions rare)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        # Simple collision avoidance for r1 == i
        coll_mask = (r1_idx == np.arange(pop_size))
        r1_idx[coll_mask] = (r1_idx[coll_mask] + 1) % pop_size
        x_r1 = pop[r1_idx]
        
        # Select r2 (random from Union(Pop, Archive))
        len_arch = len(archive)
        union_size = pop_size + len_arch
        r2_idx = np.random.randint(0, union_size, pop_size)
        
        # Construct x_r2 based on indices
        x_r2 = np.empty((pop_size, dim))
        
        # Indices < pop_size take from pop
        mask_pop = r2_idx < pop_size
        x_r2[mask_pop] = pop[r2_idx[mask_pop]]
        
        # Indices >= pop_size take from archive
        mask_arch = ~mask_pop
        if np.any(mask_arch):
            arch_indices = r2_idx[mask_arch] - pop_size
            x_r2[mask_arch] = archive[arch_indices]
            
        # Compute Mutant Vectors
        F_col = F[:, None]
        mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints
        trial = apply_bound_constraints(trial, pop)
        
        # 6. Evaluation
        trial_fitness = np.empty(pop_size)
        success_list = []
        
        # Loop for function evaluation (cannot be vectorized if func is scalar)
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
                
            f_tri = func(trial[i])
            trial_fitness[i] = f_tri
            
            if f_tri < fitness[i]:
                success_list.append(i)
                if f_tri < best_val:
                    best_val = f_tri
                    best_vec = trial[i].copy()
                    
        # 7. Selection and Updates
        if len(success_list) > 0:
            s_idx = np.array(success_list)
            
            # Add replaced individuals to Archive
            new_arch_cand = pop[s_idx]
            if len(archive) == 0:
                archive = new_arch_cand
            else:
                archive = np.vstack((archive, new_arch_cand))
            
            # Trim archive
            if len(archive) > pop_size:
                n_del = len(archive) - pop_size
                del_idx = np.random.choice(len(archive), n_del, replace=False)
                archive = np.delete(archive, del_idx, axis=0)
                
            # Update Memory (M_CR, M_F)
            diff = fitness[s_idx] - trial_fitness[s_idx]
            # Weights based on fitness improvement
            weights = diff / np.sum(diff)
            
            s_cr = CR[s_idx]
            s_f = F[s_idx]
            
            # Update M_CR (Weighted Mean)
            if np.max(s_cr) > 0:
                mean_cr = np.sum(weights * s_cr)
                M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
            
            # Update M_F (Weighted Lehmer Mean)
            num_f = np.sum(weights * (s_f ** 2))
            den_f = np.sum(weights * s_f)
            if den_f > 0:
                mean_f = num_f / den_f
                M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                
            k_mem = (k_mem + 1) % memory_size
            
            # Update Population
            pop[s_idx] = trial[s_idx]
            fitness[s_idx] = trial_fitness[s_idx]
            
            # Re-sort population
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
        # 8. Stagnation Check & Restart
        # If diversity is lost, restart to use remaining time effectively
        if pop_size > 4:
            fit_std = np.std(fitness)
            # Threshold for stagnation
            if fit_std < 1e-8:
                # Only restart if we have meaningful time left (>10%)
                if (time.time() - start_time) < (max_time * 0.9):
                    # Soft Restart: Keep best, randomize others
                    # Index 0 is best due to sorting
                    n_reset = pop_size - 1
                    pop[1:] = min_b + np.random.rand(n_reset, dim) * diff_b
                    fitness[1:] = float('inf')
                    
                    # Reset Memory and Archive
                    M_CR.fill(0.5)
                    M_F.fill(0.5)
                    archive = np.empty((0, dim))
                    
                    # Re-evaluate new individuals
                    for k in range(1, pop_size):
                        if (time.time() - start_time) >= max_time:
                            return best_val
                        val = func(pop[k])
                        fitness[k] = val
                        if val < best_val:
                            best_val = val
                            best_vec = pop[k].copy()
                            
                    # Re-sort
                    sort_idx = np.argsort(fitness)
                    pop = pop[sort_idx]
                    fitness = fitness[sort_idx]

    return best_val
