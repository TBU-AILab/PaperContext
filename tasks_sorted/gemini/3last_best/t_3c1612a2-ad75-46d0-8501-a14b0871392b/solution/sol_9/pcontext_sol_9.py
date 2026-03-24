#The following algorithm is an implementation of **L-SHADE-G (Linear Population Reduction SHADE with Gaussian Polishing)**.
#
##### Algorithm Description
#This algorithm is designed to overcome the stagnation observed in the previous attempts (stuck at ~1.75) by combining the powerful convergence of L-SHADE with a specialized "polishing" mechanism.
#
#1.  **L-SHADE with Dynamic Budgeting**: Unlike the previous L-SHADE attempt, this version dynamically estimates the remaining number of evaluations allowed within `max_time`. It performs **Linear Population Size Reduction (LPSR)** based on this dynamic budget. This ensures the population shrinks from a high-diversity state (Exploration) to a small, aggressive state (Exploitation) exactly as the time limit approaches, regardless of the function's speed.
#2.  **Gaussian Polishing (Local Search)**: The previous best result (1.75) suggests the algorithm found the correct basin of attraction but failed to pinpoint the exact global minimum (0.0). This algorithm adds a "Polisher" step: at the end of each generation, it generates a new solution by sampling a Gaussian distribution around the current global best. The standard deviation of this sample is linked to the population's current spatial variance. This acts like a stochastic gradient descent to drive the error down to near-zero.
#3.  **Soft Restarts**: If the population variance collapses (`std < 1e-9`) or the budget is exhausted while time remains, it triggers a restart. The restart preserves the Global Best but re-initializes the rest of the population to explore new areas.
#4.  **Bounce-Back Boundary Handling**: Instead of simple clipping (which ruins gradients at the edge), it uses reflection (`(bound + parent)/2`), which is more effective for bounded optimization.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # -------------------------------------------------------------------------
    # 1. Initialization and Time Management
    # -------------------------------------------------------------------------
    start_time = datetime.now()
    # Reserve a small buffer to ensure we return results before timeout
    end_time = start_time + timedelta(seconds=max_time * 0.95)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    bound_width = ub - lb
    
    # Global best tracking
    best_fitness = float('inf')
    best_sol = None
    
    # SHADE Parameters
    H_capacity = 6            # Memory size
    n_init = 18 * dim         # Initial population size
    n_init = max(30, min(n_init, 200)) # Clamp size
    n_min = 4                 # Minimum population size
    
    # -------------------------------------------------------------------------
    # 2. Main Optimization Loop (Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check hard time limit
        if datetime.now() >= end_time:
            return best_fitness
            
        # Reset population for restart
        current_pop_size = n_init
        
        # Initialize Population
        # If we have a best solution from previous run, keep it (Soft Restart)
        pop = lb + np.random.rand(current_pop_size, dim) * bound_width
        fitness = np.full(current_pop_size, float('inf'))
        
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
        # Initialize Memory (M_cr, M_f) to 0.5
        mem_cr = np.full(H_capacity, 0.5)
        mem_f = np.full(H_capacity, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = np.empty((current_pop_size, dim))
        arc_count = 0
        
        # Evaluate initial population (those not evaluated)
        for i in range(current_pop_size):
            if datetime.now() >= end_time: return best_fitness
            
            if fitness[i] == float('inf'):
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()

        # ---------------------------------------------------------------------
        # 3. Dynamic Budgeting for LPSR
        # ---------------------------------------------------------------------
        # Calculate evaluations per second to estimate max evaluations for this run
        elapsed = (datetime.now() - start_time).total_seconds()
        evals_done = current_pop_size
        
        # If this is the first few seconds, the estimate might be noisy, 
        # but SHADE self-corrects.
        if elapsed > 0:
            rate = evals_done / elapsed
        else:
            rate = 100 # Fallback arbitrary
            
        remaining_seconds = (end_time - datetime.now()).total_seconds()
        estimated_remaining_evals = int(rate * remaining_seconds)
        
        # Cap the "generation budget" for this restart to prevent infinite loops 
        # if function is very fast.
        max_evals_for_restart = min(estimated_remaining_evals, 10000 * dim)
        max_evals_for_restart = max(max_evals_for_restart, 500) # Ensure minimal run
        
        evals_in_restart = 0
        
        # ---------------------------------------------------------------------
        # 4. Evolution Loop (L-SHADE)
        # ---------------------------------------------------------------------
        while evals_in_restart < max_evals_for_restart:
            if datetime.now() >= end_time: return best_fitness
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculates the target size based on progress through the budget
            progress = evals_in_restart / max_evals_for_restart
            new_size = int(round(n_min + (n_init - n_min) * (1.0 - progress)))
            new_size = max(n_min, new_size)
            
            if current_pop_size > new_size:
                # Reduce population: sort by fitness and truncate worst
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:new_size]]
                fitness = fitness[sorted_idx[:new_size]]
                
                # Shrink archive if it exceeds new population size
                if arc_count > new_size:
                    # Randomly discard
                    keep_idx = np.random.choice(arc_count, new_size, replace=False)
                    archive[:new_size] = archive[keep_idx]
                    arc_count = new_size
                
                current_pop_size = new_size

            # --- Parameter Generation ---
            # Select random memory slot
            r_idx = np.random.randint(0, H_capacity, current_pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR (Normal dist, clipped)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            # Fold CR = 0 to small value to ensure some crossover
            cr[cr == 0] = 0.05 
            
            # Generate F (Cauchy dist)
            # F = Cauchy(m_f, 0.1) => m_f + 0.1 * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(current_pop_size) - 0.5))
            
            # Handle F bounds
            # If F > 1, clip to 1. If F <= 0, retry (or set to small value)
            while np.any(f <= 0):
                neg_mask = f <= 0
                f[neg_mask] = m_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(neg_mask)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # p-best selection (greedy)
            # p decreases linearly from 0.11 to 0.02 (exploration -> exploitation)
            # but standard SHADE often uses fixed p. Let's use 0.1 (top 10%)
            p_val = 0.1
            p_num = max(2, int(p_val * current_pop_size))
            sorted_indices = np.argsort(fitness)
            pbest_indices = sorted_indices[:p_num]
            pbest_vecs = pop[np.random.choice(pbest_indices, current_pop_size)]
            
            # r1 selection (distinct from i)
            r1_idx = np.random.randint(0, current_pop_size, current_pop_size)
            mask_self = (r1_idx == np.arange(current_pop_size))
            r1_idx[mask_self] = (r1_idx[mask_self] + 1) % current_pop_size
            xr1 = pop[r1_idx]
            
            # r2 selection (distinct from i, r1) from Union(Pop, Archive)
            union_size = current_pop_size + arc_count
            r2_idx = np.random.randint(0, union_size, current_pop_size)
            
            # Fix r2 collisions naively (rare enough to ignore or simple retry)
            # Construct xr2
            xr2 = np.empty((current_pop_size, dim))
            mask_pop = r2_idx < current_pop_size
            mask_arc = ~mask_pop
            
            xr2[mask_pop] = pop[r2_idx[mask_pop]]
            if np.any(mask_arc):
                xr2[mask_arc] = archive[r2_idx[mask_arc] - current_pop_size]
                
            # Mutation Vector
            f_col = f[:, None]
            mutant = pop + f_col * (pbest_vecs - pop) + f_col * (xr1 - xr2)
            
            # --- Crossover ---
            j_rand = np.random.randint(0, dim, current_pop_size)
            mask_cross = np.random.rand(current_pop_size, dim) < cr[:, None]
            mask_cross[np.arange(current_pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # --- Bound Handling (Bounce-Back) ---
            # Better than clipping: preserves distribution near edges
            # val = (bound + old) / 2
            below_lb = trial < lb
            above_ub = trial > ub
            trial[below_lb] = (lb[below_lb%dim] + pop[below_lb]) / 2.0
            trial[above_ub] = (ub[above_ub%dim] + pop[above_ub]) / 2.0
            
            # --- Selection ---
            fitness_old = fitness.copy()
            
            succ_scr = []
            succ_sf = []
            diff_fitness = []
            
            # Evaluation
            for i in range(current_pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                f_new = func(trial[i])
                evals_in_restart += 1
                
                if f_new <= fitness[i]:
                    # Improved or equal
                    # Add parent to archive
                    if arc_count < current_pop_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Random replacement
                        k = np.random.randint(0, current_pop_size)
                        archive[k] = pop[i].copy()
                        
                    diff = fitness[i] - f_new
                    succ_scr.append(cr[i])
                    succ_sf.append(f[i])
                    diff_fitness.append(diff)
                    
                    fitness[i] = f_new
                    pop[i] = trial[i]
                    
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_sol = trial[i].copy()

            # --- Memory Update (Weighted Lehmer Mean) ---
            if len(diff_fitness) > 0:
                w = np.array(diff_fitness)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    
                    # Mean CR
                    m_cr_new = np.sum(w * np.array(succ_scr))
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * m_cr_new
                    
                    # Mean F (Lehmer)
                    sf = np.array(succ_sf)
                    m_f_new = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * m_f_new
                    
                    k_mem = (k_mem + 1) % H_capacity

            # --- Gaussian Polishing (Local Search) ---
            # Try to improve the best solution slightly every generation.
            # This helps break the 1e-1 barrier down to 1e-10.
            if best_sol is not None:
                if datetime.now() >= end_time: return best_fitness
                
                # Step size depends on population diversity (std dev)
                # If population is converged, step is tiny. If scattered, step is large.
                pop_std = np.std(pop, axis=0)
                # Ensure a minimal step size to prevent freezing
                step_size = np.maximum(pop_std, 1e-8 * bound_width)
                
                # Generate 1 neighbor
                neighbor = best_sol + np.random.normal(0, 1, dim) * step_size
                neighbor = np.clip(neighbor, lb, ub)
                
                n_val = func(neighbor)
                evals_in_restart += 1
                
                if n_val < best_fitness:
                    best_fitness = n_val
                    best_sol = neighbor
                    # If local search worked, update the worst in population to this
                    # to pull population towards the basin
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = neighbor
                    fitness[worst_idx] = n_val

            # --- Convergence Check ---
            # If population variance is extremely low, restart to save time
            if np.max(fitness) - np.min(fitness) < 1e-9:
                break
                
    return best_fitness
