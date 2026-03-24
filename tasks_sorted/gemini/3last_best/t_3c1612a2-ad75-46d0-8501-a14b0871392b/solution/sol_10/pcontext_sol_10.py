#The following algorithm is an implementation of **L-SHADE-R (Linear Population Size Reduction SHADE with Restarts)**.
#
#### Algorithm Description
#
#This algorithm combines the state-of-the-art **L-SHADE** evolutionary strategy with a robust **Restart** mechanism to handle both multimodal landscapes and strict time constraints.
#
#1.  **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**:
#    *   **Adaptive Parameters**: Uses historical memory ($H$) to adapt Scaling Factor ($F$) and Crossover Rate ($CR$) for each individual, balancing exploration and exploitation.
#    *   **LPSR (Linear Population Size Reduction)**: Linearly reduces the population size from a large initial value ($18 \times dim$) to a minimal size ($4$) over the course of the "evaluation budget". This forces the algorithm to shift from global search (high diversity) to local refinement (high pressure) automatically.
#    *   **External Archive**: Maintains a diversity buffer of inferior solutions to prevent premature convergence during the `current-to-pbest` mutation strategy.
#
#2.  **Dynamic Budgeting & Restarts**:
#    *   Instead of a single run, the algorithm splits the available `max_time` into dynamic "runs".
#    *   It continuously estimates the cost of `func` evaluations.
#    *   For each restart, it allocates a budget of evaluations (capped at $3000 \times dim$ or the remaining time) to ensure the LPSR schedule is aggressive enough to converge but short enough to allow multiple restarts if the first run gets stuck in a local optimum.
#    *   **Restart Trigger**: A restart occurs if the population variance drops below a threshold (convergence) or if the allocated evaluation budget is exhausted.
#
#3.  **Bounce-Back Boundary Handling**:
#    *   Instead of clipping values to bounds (which can stagnate optimization on the edges), it uses a "bounce-back" approach: $x_{new} = (bound + x_{old}) / 2$. This effectively searches the region near the boundary without destroying the gradient information.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # -------------------------------------------------------------------------
    # 1. Initialization and Time Management
    # -------------------------------------------------------------------------
    start_time = datetime.now()
    # Reserve a 2% buffer to ensure strict adherence to max_time
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global best solution found across all restarts
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # L-SHADE Constants
    min_pop_size = 4
    # Initial pop size: usually 18*dim, clamped for performance safety
    init_pop_size = int(max(30, min(18 * dim, 250)))
    
    # Time estimation
    eval_times = []
    
    # -------------------------------------------------------------------------
    # 2. Main Optimization Loop (Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check hard time limit
        if datetime.now() >= end_time:
            return global_best_fitness
        
        # --- Run Initialization ---
        pop_size = init_pop_size
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize Memory (H slots)
        H = 6
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = np.empty((pop_size, dim))
        n_arc = 0
        
        # Evaluate Initial Population
        # (Check time frequently within loops)
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_fitness
            
            t0 = datetime.now()
            val = func(pop[i])
            t1 = datetime.now()
            eval_times.append((t1 - t0).total_seconds())
            
            fitness[i] = val
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = pop[i].copy()
                
        # --- Dynamic Budgeting for LPSR ---
        # Estimate remaining evaluations possible
        if len(eval_times) > 0:
            avg_t = np.mean(eval_times[-100:]) # Moving average of last 100
            if avg_t < 1e-10: avg_t = 1e-10
        else:
            avg_t = 0.0
            
        remaining_seconds = (end_time - datetime.now()).total_seconds()
        if remaining_seconds <= 0: return global_best_fitness
        
        estimated_evals_left = int(remaining_seconds / avg_t)
        
        # Define max evaluations for THIS restart run.
        # Cap at 3000*dim to force convergence and allow potential restarts.
        # But ensure we use available time if it's short.
        max_evals_run = min(estimated_evals_left, 3000 * dim)
        max_evals_run = max(max_evals_run, 500) # Minimum limit to allow evolution
        
        curr_evals_run = pop_size # Already spent on init
        
        # ---------------------------------------------------------------------
        # 3. Evolution Loop (L-SHADE)
        # ---------------------------------------------------------------------
        while curr_evals_run < max_evals_run:
            if datetime.now() >= end_time: return global_best_fitness
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculate target size based on budget consumption
            progress = curr_evals_run / max_evals_run
            target_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Reduce Population: keep best
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices[:target_size]]
                fitness = fitness[sort_indices[:target_size]]
                
                # Resize Archive: maintain size <= pop_size
                if n_arc > target_size:
                    n_arc = target_size
                    # Randomly discard excess
                    keep_idx = np.random.choice(n_arc, target_size, replace=False)
                    archive[:target_size] = archive[keep_idx]
                    
                pop_size = target_size

            # --- Parameter Generation ---
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F: Cauchy(m_f, 0.1), clipped [0, 1], regenerate if <= 0
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            while np.any(f <= 0):
                mask_neg = f <= 0
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # p-best selection (top 11%)
            p = max(2, int(0.11 * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest_indices = sorted_idx[:p]
            pbest_vectors = pop[np.random.choice(pbest_indices, pop_size)]
            
            # r1 selection (distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Simple collision handling: if r1==i, shift by 1
            mask_col = (r1_idx == np.arange(pop_size))
            r1_idx[mask_col] = (r1_idx[mask_col] + 1) % pop_size
            xr1 = pop[r1_idx]
            
            # r2 selection (distinct from i, r1) from Union(Pop, Archive)
            union_size = pop_size + n_arc
            r2_idx = np.random.randint(0, union_size, pop_size)
            # Collision handling ignored for r2 for speed/vectorization (low prob)
            
            xr2 = np.empty((pop_size, dim))
            mask_pop = r2_idx < pop_size
            xr2[mask_pop] = pop[r2_idx[mask_pop]]
            if n_arc > 0:
                mask_arc = ~mask_pop
                xr2[mask_arc] = archive[r2_idx[mask_arc] - pop_size]
            else:
                xr2[~mask_pop] = pop[r2_idx[~mask_pop] % pop_size] # Fallback
                
            # Compute Mutant
            f_col = f[:, None]
            mutant = pop + f_col * (pbest_vectors - pop) + f_col * (xr1 - xr2)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross = np.random.rand(pop_size, dim) < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # --- Bound Handling (Bounce-Back) ---
            # x_new = (bound + x_old) / 2 if out of bounds
            mask_lb = trial < lb
            mask_ub = trial > ub
            trial[mask_lb] = (lb[mask_lb % dim] + pop[mask_lb]) / 2.0
            trial[mask_ub] = (ub[mask_ub % dim] + pop[mask_ub]) / 2.0
            
            # --- Selection and Memory Update ---
            diffs = []
            succ_cr = []
            succ_f = []
            
            # Evaluate trial vectors
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_fitness
                
                # Check budget mid-loop
                if curr_evals_run >= max_evals_run:
                    break
                
                t0 = datetime.now()
                val = func(trial[i])
                t1 = datetime.now()
                eval_times.append((t1 - t0).total_seconds())
                curr_evals_run += 1
                
                if val <= fitness[i]:
                    # Improvement or Neutral
                    # Add parent to archive
                    if n_arc < pop_size:
                        archive[n_arc] = pop[i].copy()
                        n_arc += 1
                    else:
                        # Random replacement
                        k = np.random.randint(0, n_arc)
                        archive[k] = pop[i].copy()
                        
                    diffs.append(fitness[i] - val)
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    
                    fitness[i] = val
                    pop[i] = trial[i]
                    
                    if val < global_best_fitness:
                        global_best_fitness = val
                        global_best_sol = trial[i].copy()
            
            # Update Memory (Weighted Lehmer Mean)
            if len(diffs) > 0:
                w = np.array(diffs)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    
                    # M_cr update
                    m_cr_new = np.sum(w * succ_cr)
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * m_cr_new
                    
                    # M_f update
                    sf = np.array(succ_f)
                    num = np.sum(w * sf**2)
                    den = np.sum(w * sf)
                    m_f_new = num / (den + 1e-15)
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * m_f_new
                    
                    k_mem = (k_mem + 1) % H
            
            # --- Convergence Check ---
            # If population has collapsed, restart early
            if np.std(fitness) < 1e-9 and (np.max(fitness) - np.min(fitness)) < 1e-9:
                break
                
    return global_best_fitness
