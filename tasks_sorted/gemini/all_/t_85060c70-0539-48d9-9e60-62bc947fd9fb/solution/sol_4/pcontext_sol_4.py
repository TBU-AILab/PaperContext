#The following Python code implements an optimized **L-SHADE** (Linear Population Size Reduction Success-History based Adaptive Differential Evolution) algorithm with a **Restart Mechanism**.
#
#**Key Improvements over previous attempts:**
#1.  **Fully Vectorized Operations**: Parameter generation, mutation, crossover, and bound handling are performed using NumPy array operations. This significantly reduces Python interpreter overhead, allowing for many more function evaluations within the `max_time` limit.
#2.  **Linear Population Size Reduction (LPSR)**: The population size strictly decreases over time, ensuring broad exploration at the start and focused exploitation at the end.
#3.  **Restart Mechanism**: If the population converges (low variance) or becomes too small (end of reduction schedule) while time remains, the algorithm triggers a "Soft Restart". It re-initializes the population to explore new basins of attraction but preserves the global best solution (Elitism) to ensure monotonic improvement.
#4.  **Adaptive p-Best**: The greediness of the mutation strategy adapts over time (starting with `p=0.2` and ending at `p=0.05`), balancing search pressure.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized L-SHADE with Restart Strategy.
    """
    
    # --- 1. Configuration & Constants ---
    # Population Size Heuristics
    # Start with a large population for exploration, but clamp to reasonable limits
    # to avoid excessive initialization cost on high dimensions.
    init_pop_size = int(np.clip(25 * dim, 50, 300))
    min_pop_size = 4
    
    # L-SHADE Hyperparameters
    H = 5                   # History memory size
    arc_rate = 2.0          # Archive size relative to current population
    p_start = 0.2           # Initial p-best percentage (Exploration)
    p_end = 0.05            # Final p-best percentage (Exploitation)
    
    # --- 2. Initialization ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- 3. Helper Functions ---
    def check_timeout():
        return datetime.now() >= end_time
    
    def get_time_ratio():
        """Returns 1.0 at start, decreasing to 0.0 at max_time."""
        elapsed = (datetime.now() - start_time).total_seconds()
        return max(0.0, 1.0 - (elapsed / max_time))

    # --- 4. Main Optimization Loop (Restarts) ---
    # We loop allowing restarts until time is exhausted
    while not check_timeout():
        
        # A. Initialize Population for this run
        pop_size = init_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution into the new population
        start_eval_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
        
        # B. Initialize L-SHADE Memory & Archive
        mem_cr = np.full(H, 0.5) # Crossover Memory
        mem_f = np.full(H, 0.5)  # Mutation Factor Memory
        k_mem = 0                # Memory Pointer
        archive = []             # External Archive
        
        # C. Evolutionary Generation Loop
        while pop_size > min_pop_size:
            if check_timeout(): return global_best_val
            
            # --- C1. Linear Population Size Reduction (LPSR) ---
            t_ratio = get_time_ratio()
            
            # Calculate target population size based on remaining time
            target_size = int(round(min_pop_size + (init_pop_size - min_pop_size) * t_ratio))
            
            if target_size < pop_size:
                # Reduce Population: Keep best individuals
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices]
                fitness = fitness[sort_indices]
                
                # Truncate
                pop_size = max(min_pop_size, target_size)
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive
                max_arc_size = int(pop_size * arc_rate)
                if len(archive) > max_arc_size:
                    # Randomly remove excess elements
                    # (Rebuilding list is faster than repeated pop for bulk removal)
                    keep_indices = np.random.choice(len(archive), max_arc_size, replace=False)
                    archive = [archive[ix] for ix in keep_indices]

            # --- C2. Parameter Adaptation ---
            # Linearly decrease p from p_start to p_end
            p_curr = p_end + (p_start - p_end) * t_ratio
            top_p_count = max(2, int(pop_size * p_curr))
            
            # Generate CR and F from Memory
            # Random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # numpy doesn't have a direct Cauchy with loc/scale, use tan transformation
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f = np.where(f > 1.0, 1.0, f)   # Clamp > 1 to 1
            f = np.where(f <= 0.0, 0.5, f)  # If <= 0, robust fallback to 0.5
            
            # --- C3. Mutation: DE/current-to-pbest/1 ---
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # Identify sorted indices for p-best selection
            sorted_indices = np.argsort(fitness)
            pbest_pool = sorted_indices[:top_p_count]
            
            # Select p-best indices (randomly from top p%)
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 indices (random from population, r1 != i)
            # Vectorized approach: shift index by random [1, pop_size-1]
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idx = (np.arange(pop_size) + shift) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 indices (random from Union(Pop, Archive), r2 != i, r2 != r1)
            # Prepare Union Population
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            union_size = len(union_pop)
            
            # Generate random r2 indices
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Fix collisions (r2 must not equal i (current) or r1)
            curr_idx = np.arange(pop_size)
            # Note: r1_idx refers to pop index. r2_idx refers to union index.
            # If r2_idx < pop_size, it points to pop, so we check equality.
            bad_indices = (r2_idx == curr_idx) | (r2_idx == r1_idx)
            
            # Fast rejection sampling for collisions
            while np.any(bad_indices):
                r2_idx[bad_indices] = np.random.randint(0, union_size, np.sum(bad_indices))
                bad_indices = (r2_idx == curr_idx) | (r2_idx == r1_idx)
                
            x_r2 = union_pop[r2_idx]
            
            # Calculate Mutant Vector
            # Reshape F for broadcasting (pop_size, 1)
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- C4. Crossover (Binomial) ---
            # Create mask: rand < CR
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            
            # Ensure at least one parameter comes from mutant (j_rand)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # --- C5. Bound Constraints ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- C6. Selection & Evaluation ---
            # Prepare arrays for successful updates
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            # Temporary storage for new generation
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            # Evaluate Loop (Bottleneck: Function Calls)
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                
                trial_val = func(trial[i])
                
                # Greedy Selection
                if trial_val <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fitness[i] = trial_val
                    
                    if trial_val < fitness[i]:
                        # Strict improvement found
                        succ_mask[i] = True
                        diff_fitness[i] = fitness[i] - trial_val
                        archive.append(pop[i].copy()) # Add replaced solution to archive
                        
                    # Update Global Best
                    if trial_val < global_best_val:
                        global_best_val = trial_val
                        global_best_sol = trial[i].copy()
            
            # Update Population
            pop = new_pop
            fitness = new_fitness
            
            # --- C7. Update Historical Memory ---
            if np.any(succ_mask):
                succ_f = f[succ_mask]
                succ_cr = cr[succ_mask]
                succ_diff = diff_fitness[succ_mask]
                
                # Calculate weights based on fitness improvement
                weights = succ_diff / np.sum(succ_diff)
                
                # Weighted Mean for CR
                mean_cr = np.sum(weights * succ_cr)
                mem_cr[k_mem] = mean_cr
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(weights * (succ_f**2)) / (np.sum(weights * succ_f) + 1e-10)
                mean_f = np.clip(mean_f, 0, 1)
                mem_f[k_mem] = mean_f
                
                # Advance memory pointer (circular buffer)
                k_mem = (k_mem + 1) % H
            
            # --- C8. Convergence Check (Early Restart) ---
            # If population variance is negligible, we are stuck in a local optimum
            if np.std(fitness) < 1e-8:
                break # Break inner loop to trigger restart

    return global_best_val
