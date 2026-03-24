#The following Python algorithm implements **L-SHADE (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)** with a **Dynamic Time-Based Restart Mechanism**.
#
#### Key Improvements
#1.  **Linear Population Size Reduction (L-SHADE)**: Unlike standard SHADE (fixed population), this algorithm linearly reduces the population size from a high value (exploration) to a small value (exploitation) based on the elapsed time. This significantly increases convergence pressure in the final stages of the search.
#2.  **Dynamic Restart Scheduling**: If the population converges (stagnation) before the time limit, the algorithm triggers a restart. Crucially, the **L-SHADE reduction curve is recalculated** based on the *remaining* time. This ensures that every restart cycle is perfectly optimized to fit within the leftover time budget.
#3.  **Memory & Archive Management**: Utilizes an external archive and historical memory ($M_{CR}, M_F$) to adapt control parameters to the landscape, but dynamically resizes the archive capacity as the population shrinks to maintain a consistent diversity ratio.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using L-SHADE (Linear Population Size Reduction Success-History 
    Adaptive DE) with Dynamic Time-Based Restarts.
    """
    # --- Time Management ---
    start_time = datetime.now()
    deadline = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Constants & Configuration ---
    # Initial population size (N_init) and minimum size (N_min)
    # L-SHADE recommends N_init around 18*dim
    N_init = int(np.clip(18 * dim, 40, 200))
    N_min = 4
    
    # Hyperparameters
    H_memory_size = 6
    arc_rate = 2.0  # Archive size relative to population size
    p_best_rate = 0.11 # Top 11% for current-to-pbest
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_fit = float('inf')
    global_best_sol = None
    
    # --- Restart Loop ---
    # The algorithm runs in cycles (restarts) if it converges early.
    # Each cycle treats the *remaining time* as its full duration for the L-SHADE schedule.
    
    while True:
        current_time = datetime.now()
        if current_time >= deadline:
            break
            
        # Determine duration for this specific run cycle
        # On first run, this is roughly max_time. On restarts, it's whatever is left.
        # We assume the run goes until the deadline unless convergence happens.
        cycle_start_time = current_time
        
        # --- Run Initialization ---
        pop_size = N_init
        max_archive_size = int(pop_size * arc_rate)
        
        # Memory for SHADE (reset on restart)
        mem_cr = np.full(H_memory_size, 0.5)
        mem_f = np.full(H_memory_size, 0.5)
        k_mem = 0
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Inject global best into new population if available (Exploitation of previous knowledge)
        if global_best_sol is not None:
            pop[0] = global_best_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= deadline:
                return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()
        
        # Sort for p-best selection logic
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # External Archive (Stores inferior solutions replaced by selection)
        archive = np.zeros((max_archive_size, dim))
        n_archive = 0
        
        # --- Optimization Loop (Generations) ---
        while datetime.now() < deadline:
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress ratio relative to the CURRENT cycle's allowed duration
            now = datetime.now()
            # Total available time for this cycle
            total_cycle_seconds = (deadline - cycle_start_time).total_seconds()
            if total_cycle_seconds <= 1e-6: total_cycle_seconds = 1e-6
            
            elapsed_cycle_seconds = (now - cycle_start_time).total_seconds()
            progress = min(1.0, elapsed_cycle_seconds / total_cycle_seconds)
            
            # Calculate target population size
            new_pop_size = int(round(N_init + (N_min - N_init) * progress))
            new_pop_size = max(N_min, new_pop_size)
            
            # Reduce Population if needed
            if new_pop_size < pop_size:
                # Pop is already sorted by fitness (best at 0).
                # Truncate the worst individuals (at the end).
                remove_count = pop_size - new_pop_size
                pop_size = new_pop_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive capacity
                max_archive_size = int(pop_size * arc_rate)
                if n_archive > max_archive_size:
                    # Randomly remove excess from archive
                    keep_indices = np.random.choice(n_archive, max_archive_size, replace=False)
                    archive[:max_archive_size] = archive[keep_indices]
                    n_archive = max_archive_size

            # 2. Check for Convergence (Trigger Restart)
            # If population variance is tiny or pop size reached minimum and stagnated
            if np.std(fitness) < 1e-9 or (pop_size == N_min and np.std(fitness) < 1e-5):
                # Break inner loop to trigger restart in outer loop
                break

            # 3. Parameter Adaptation
            # Select memory index
            r_idx = np.random.randint(0, H_memory_size, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            # Special case: if m_cr is close to terminal value -1 in some implementations, 
            # but standard SHADE keeps it simple.
            
            # Generate F ~ Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Clamp F
            f = np.where(f > 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.1, f) # Avoid 0 mutation
            
            # 4. Mutation: current-to-pbest/1
            # Sort is maintained from end of loop or init
            # Select p-best (top p%)
            num_pbest = max(2, int(pop_size * p_best_rate))
            pbest_indices = np.random.randint(0, num_pbest, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (distinct from i)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Collision handling
            conflict = (r1_indices == np.arange(pop_size))
            r1_indices[conflict] = (r1_indices[conflict] + 1) % pop_size
            x_r1 = pop[r1_indices]
            
            # Select r2 (from Union(Pop, Archive))
            # Construct union view (not copy if possible, but vstack is safe)
            if n_archive > 0:
                union_pop = np.vstack((pop, archive[:n_archive]))
            else:
                union_pop = pop
                
            r2_indices = np.random.randint(0, len(union_pop), pop_size)
            # Ensure r2 != r1 and r2 != i is ideal but costly to enforce strictly vectorized.
            # SHADE standard implies simple random selection from union.
            x_r2 = union_pop[r2_indices]
            
            # Calculate Mutant Vector
            # v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # Bound Constraints (Clip)
            trial = np.clip(trial, min_b, max_b)
            
            # 6. Selection
            success_f = []
            success_cr = []
            diffs = []
            
            # Iterate for evaluation
            # (Loop is necessary as func takes 1D array, but we check time frequently)
            for i in range(pop_size):
                if datetime.now() >= deadline:
                    return global_best_fit
                
                f_trial = func(trial[i])
                
                if f_trial < fitness[i]:
                    # Improvement found
                    diff = fitness[i] - f_trial
                    
                    # Update Archive
                    if n_archive < max_archive_size:
                        archive[n_archive] = pop[i].copy()
                        n_archive += 1
                    else:
                        # Replace random member
                        rep_idx = np.random.randint(0, max_archive_size)
                        archive[rep_idx] = pop[i].copy()
                        
                    # Update Pop & Fitness
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Store successful params
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diffs.append(diff)
                    
                    # Update Global Best
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial[i].copy()
            
            # 7. Update SHADE Memory
            if len(success_f) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                s_diff = np.array(diffs)
                
                # Weighted means based on fitness improvement
                total_diff = np.sum(s_diff)
                weights = s_diff / total_diff
                
                # Update M_CR (Weighted Mean)
                # If s_cr is -1 (terminal), handle it? Standard SHADE doesn't need -1 logic here.
                mean_cr = np.sum(weights * s_cr)
                
                # Update M_F (Weighted Lehmer Mean)
                num = np.sum(weights * (s_f ** 2))
                den = np.sum(weights * s_f)
                mean_f = num / den if den > 0 else 0.5
                
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = mean_f
                k_mem = (k_mem + 1) % H_memory_size
                
            # Sort population at end of generation (required for next gen p-best and reduction)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]

    return global_best_fit
