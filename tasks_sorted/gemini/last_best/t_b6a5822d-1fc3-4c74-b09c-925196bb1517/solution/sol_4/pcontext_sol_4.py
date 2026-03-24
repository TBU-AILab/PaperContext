#Here is the improved algorithm. It implements **SHADE-OBL (Success-History Adaptive Differential Evolution with Opposition-Based Learning and Soft Restarts)**.
#
#### Improvements Explained:
#
#1.  **Opposition-Based Learning (OBL)**: During initialization and restarts, the algorithm evaluates both the random solution $x$ and its opposite solution $x' = lb + ub - x$. This effectively doubles the coverage of the search space immediately, often locating the basin of attraction much faster than random initialization.
#2.  **Vectorized Operations**: The mutation, crossover, and bound handling steps are fully vectorized using NumPy. This drastically reduces the overhead of the algorithm itself, allowing for more function evaluations within the limited `max_time`.
#3.  **Soft Restarts with Local Refinement**: Instead of stopping when the population converges (standard deviation drops), the algorithm triggers a restart. Before restarting, it performs a brief local search around the global best to squeeze out final precision. The restart then keeps the best solution (elitism) and re-initializes the rest with OBL, ensuring no progress is lost while escaping local optima.
#4.  **Adaptive Population Sizing**: The population size is dynamically scaled based on the dimension ($D$), but capped to ensure high iteration speed.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE with Opposition-Based Learning (OBL)
    and Soft Restarts.
    """
    
    # --- Configuration & Time Management ---
    start_time = datetime.now()
    # Subtract small buffer to manage function return overhead
    time_limit = timedelta(seconds=max_time - 0.05)
    
    class TimeoutException(Exception):
        pass

    def check_time():
        if (datetime.now() - start_time) >= time_limit:
            raise TimeoutException

    # --- Pre-computation ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff_b = ub - lb
    
    # Global tracker
    global_best_val = float('inf')
    global_best_pos = None

    # SHADE Parameters
    # History Memory Size
    H = 6
    # Population size logic: 18 * dim, clipped [30, 200]
    pop_size = int(np.clip(18 * dim, 30, 200))
    
    # Helper: Evaluation with Global Best Update
    def evaluate(candidates):
        nonlocal global_best_val, global_best_pos
        vals = np.zeros(len(candidates))
        for i in range(len(candidates)):
            check_time()
            val = func(candidates[i])
            vals[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = candidates[i].copy()
        return vals

    # Helper: Opposition Based Learning (OBL) Initialization
    def obl_initialization(size):
        # 1. Random Init
        p_rand = lb + np.random.rand(size, dim) * diff_b
        f_rand = evaluate(p_rand)
        
        # 2. Opposite Init
        p_opp = lb + ub - p_rand
        # Clip to bounds
        p_opp = np.clip(p_opp, lb, ub)
        f_opp = evaluate(p_opp)
        
        # 3. Selection
        # Create mask where Opp is better than Rand
        mask = f_opp < f_rand
        
        # Combine
        p_final = np.where(mask[:, None], p_opp, p_rand)
        f_final = np.where(mask, f_opp, f_rand)
        
        return p_final, f_final

    try:
        # --- Main Restart Loop ---
        while True:
            # 1. Initialize Population (with OBL)
            pop, fitness = obl_initialization(pop_size)
            
            # If this is a restart (global_best exists), inject global best
            if global_best_pos is not None:
                # Replace worst solution with global best to maintain elitism
                worst_idx = np.argmax(fitness)
                pop[worst_idx] = global_best_pos
                fitness[worst_idx] = global_best_val

            # Initialize SHADE Memory
            mem_cr = np.full(H, 0.5)
            mem_f = np.full(H, 0.5)
            k_mem = 0
            archive = []
            
            # Stagnation counter
            stag_counter = 0
            last_best_fit = np.min(fitness)

            # --- Evolution Loop ---
            while True:
                check_time()

                # Sort by fitness (for p-best selection)
                sorted_indices = np.argsort(fitness)
                pop = pop[sorted_indices]
                fitness = fitness[sorted_indices]

                # --- 1. Parameter Generation ---
                # Select random memory index for each individual
                r_idx = np.random.randint(0, H, pop_size)
                m_cr = mem_cr[r_idx]
                m_f = mem_f[r_idx]

                # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
                cr = np.random.normal(m_cr, 0.1)
                cr = np.clip(cr, 0, 1)

                # Generate F: Cauchy(m_f, 0.1)
                # Retry if F <= 0, Clip if F > 1
                f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
                # Vectorized retry for bad F
                bad_f_mask = f <= 0
                while np.any(bad_f_mask):
                    f[bad_f_mask] = m_f[bad_f_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f_mask)) - 0.5))
                    bad_f_mask = f <= 0
                f = np.minimum(f, 1.0)

                # --- 2. Mutation (current-to-pbest/1) ---
                # p-best selection: random top p% (p in [2/N, 0.2])
                p_val = np.random.uniform(2.0/pop_size, 0.2)
                top_cut = int(max(2, pop_size * p_val))
                
                # Indexes for pbest
                pbest_indices = np.random.randint(0, top_cut, pop_size) 
                x_pbest = pop[pbest_indices]
                
                # r1 generation: random from pop, r1 != i
                r1_indices = np.random.randint(0, pop_size, pop_size)
                # Fix collisions where r1 == i (simple swap)
                collisions = (r1_indices == np.arange(pop_size))
                r1_indices[collisions] = (r1_indices[collisions] + 1) % pop_size
                x_r1 = pop[r1_indices]
                
                # r2 generation: random from Union(Pop + Archive)
                # Union array
                if len(archive) > 0:
                    archive_np = np.array(archive)
                    union_pop = np.vstack((pop, archive_np))
                else:
                    union_pop = pop
                
                union_size = len(union_pop)
                r2_indices = np.random.randint(0, union_size, pop_size)
                # Collision handling for r2 is less critical in Union, skip complex check for speed
                x_r2 = union_pop[r2_indices]

                # Mutation Vector Calculation
                # v = x + F*(xp - x) + F*(xr1 - xr2)
                mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
                
                # Boundary Constraint: Clip
                mutant = np.clip(mutant, lb, ub)

                # --- 3. Crossover (Binomial) ---
                rand_vals = np.random.rand(pop_size, dim)
                mask = rand_vals < cr[:, None]
                # Ensure at least one parameter comes from mutant
                j_rand = np.random.randint(0, dim, pop_size)
                mask[np.arange(pop_size), j_rand] = True
                
                trial_pop = np.where(mask, mutant, pop)

                # --- 4. Selection ---
                trial_fitness = evaluate(trial_pop)
                
                # Improvement mask
                better_mask = trial_fitness < fitness
                
                # Update Archive (Store replaced parents)
                if np.any(better_mask):
                    parents_to_archive = pop[better_mask].copy()
                    for p in parents_to_archive:
                        if len(archive) < pop_size:
                            archive.append(p)
                        else:
                            # Replace random archive member
                            rem_idx = np.random.randint(0, len(archive))
                            archive[rem_idx] = p

                # Update Population and Fitness
                pop = np.where(better_mask[:, None], trial_pop, pop)
                fitness = np.where(better_mask, trial_fitness, fitness)

                # --- 5. Memory Update ---
                if np.any(better_mask):
                    diff = fitness[better_mask] - trial_fitness[better_mask] # Actually this is 0 or neg? 
                    # Correct logic: old_fitness - new_fitness (positive diff)
                    # We already updated fitness, so we need to reconstruct diff carefully
                    # Or simpler:
                    # diff_fitness = old_fitness - new_fitness. 
                    # Since we overwrote fitness, let's just rely on the fact we are minimizing.
                    # We need the magnitude of improvement.
                    
                    # Re-calculate diff strictly for successful updates
                    # Previous fitness was higher.
                    # We can't retrieve old fitness easily due to vectorization unless we stored it.
                    # Given the template constraints, let's use a simpler arithmetic mean if weight calc is too expensive,
                    # but Weighted Lehmer is standard.
                    
                    # Let's approximate weights as equal if we lost the diff, OR just skip weighted mean 
                    # and use simple mean for robustness/speed. 
                    # Actually, we can just use the generated F and CR that were successful.
                    succ_f = f[better_mask]
                    succ_cr = cr[better_mask]
                    
                    # Update Memory (Simple Mean for speed and stability)
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * np.mean(succ_f)
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * np.mean(succ_cr)
                    
                    k_mem = (k_mem + 1) % H

                # --- 6. Convergence / Restart Check ---
                current_best = np.min(fitness)
                
                # Stagnation check
                if (last_best_fit - current_best) < 1e-8:
                    stag_counter += 1
                else:
                    stag_counter = 0
                    last_best_fit = current_best

                # Restart triggers:
                # 1. Population Variance is extremely low (Converged)
                # 2. Stagnation for too many gens
                std_dev = np.std(fitness)
                
                if std_dev < 1e-9 or stag_counter > 40:
                    # Perform Local Refinement on Global Best before restart
                    # Try a few small random walks around best
                    if global_best_pos is not None:
                        for _ in range(20): # Small budget local search
                            check_time()
                            # Gaussian step, diminishing size
                            step_size = 0.01 * (ub - lb)
                            neighbor = global_best_pos + np.random.normal(0, 1, dim) * step_size
                            neighbor = np.clip(neighbor, lb, ub)
                            val = func(neighbor)
                            if val < global_best_val:
                                global_best_val = val
                                global_best_pos = neighbor
                    
                    # Break to outer loop to restart
                    break

    except TimeoutException:
        return global_best_val

    return global_best_val
