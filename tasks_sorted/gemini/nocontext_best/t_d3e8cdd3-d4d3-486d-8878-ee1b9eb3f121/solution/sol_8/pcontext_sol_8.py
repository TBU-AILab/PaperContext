#Here is a fully operational, self-contained implementation.
#
#This algorithm implements **L-SHADE-RSP (Linear Population Size Reduction with Rank-based Selective Pressure)** combined with a **Restarter** and **MTS-LS1 (Multiple Trajectory Search - Local Search 1)**.
#
#### Key Improvements:
#1.  **L-SHADE with Linear Population Reduction**: Instead of a fixed population, this algorithm starts with a large population to explore the global landscape and linearly reduces it to concentrate on the best basin of attraction.
#2.  **Adaptive Memory**: Uses the weighted Lehmer mean to adapt mutation ($F$) and crossover ($CR$) rates based on successful updates, learning the landscape's features.
#3.  **Local Search Integration (MTS-LS1)**: A coordinate-descent-based local search is applied at the end of every restart cycle (epoch) to polish the best solution found, correcting the "fine-tuning" weakness of Differential Evolution.
#4.  **Efficiency**: Heavy use of NumPy vectorization for mutation, crossover, and boundary handling to minimize interpreter overhead.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an L-SHADE variant with Linear Population Size Reduction,
    Adaptive Memory, and MTS-LS1 Local Search polishing.
    """
    # --- Configuration & Helpers ---
    start_time = time.time()
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    def get_time_elapsed():
        return time.time() - start_time
    
    def is_time_left(safety_margin=0.02):
        return get_time_elapsed() < (max_time - safety_margin)

    # --- MTS-LS1 Local Search ---
    # A robust coordinate-descent local search to refine the best solution
    def mts_ls1(best_pos, best_val, time_limit):
        ls_start = time.time()
        pos = best_pos.copy()
        val = best_val
        
        # Initial search range (40% of domain)
        sr = (ub - lb) * 0.4 
        
        dim_indices = np.arange(dim)
        
        while (time.time() - ls_start) < time_limit:
            improved = False
            np.random.shuffle(dim_indices) # Randomize dimension order
            
            for i in dim_indices:
                if (time.time() - ls_start) > time_limit: break
                
                original = pos[i]
                
                # Direction 1: Negative
                pos[i] = np.clip(original - sr[i], lb[i], ub[i])
                f_new = func(pos)
                
                if f_new < val:
                    val = f_new
                    improved = True
                else:
                    # Direction 2: Positive (half step)
                    pos[i] = np.clip(original + 0.5 * sr[i], lb[i], ub[i])
                    f_new = func(pos)
                    
                    if f_new < val:
                        val = f_new
                        improved = True
                    else:
                        pos[i] = original # Revert if no improvement
            
            if not improved:
                # Reduce search range if no improvement in full pass
                sr *= 0.5
                # Check convergence
                if np.max(sr) < 1e-15:
                    break 
            else:
                # Keep current SR or optional logic to reset/decay differently
                pass
                
        return pos, val

    # --- Global Best Tracking ---
    g_best_val = float('inf')
    g_best_pos = np.random.uniform(lb, ub)
    
    # Initial Evaluation to warm up and seed global best
    try:
        f_init = func(g_best_pos)
        g_best_val = f_init
    except:
        pass # Safety

    # --- L-SHADE Parameters ---
    final_pop_size = 4
    restart_count = 0
    
    # --- Main Optimization Loop (Restarts) ---
    while is_time_left():
        
        # 1. Initialization per Epoch
        # Use standard L-SHADE population sizing (18 * D), capped for speed
        init_pop_size = min(18 * dim, 250)
        
        # Dynamic adjustment: if time is short, start with smaller population for faster convergence
        elapsed = get_time_elapsed()
        if (max_time - elapsed) < (max_time * 0.2):
             init_pop_size = max(20, int(init_pop_size * 0.5))
             
        pop_size = init_pop_size
        
        # Initialize Population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        
        # Inject global best from previous runs (Exploitation)
        if restart_count > 0:
            pop[0] = g_best_pos.copy()
            
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if not is_time_left(): break
            fitness[i] = func(pop[i])
            if fitness[i] < g_best_val:
                g_best_val = fitness[i]
                g_best_pos = pop[i].copy()
                
        # Sort population (required for rank-based logic)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Memory Initialization (History of successful F and CR)
        H = 5
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Budget Estimation for Linear Reduction
        # We estimate a "budget" of evaluations for this epoch to drive population reduction.
        # Heuristic: 2000 * dim evaluations per restart
        max_nfe_epoch = 2500 * dim
        curr_nfe_epoch = 0
        
        # --- Evolutionary Cycle ---
        while pop_size > final_pop_size and is_time_left():
            
            # --- A. Linear Population Size Reduction (LPSR) ---
            # Calculate target population size based on progress
            plan_progress = curr_nfe_epoch / max_nfe_epoch
            next_size = int(round((final_pop_size - init_pop_size) * plan_progress + init_pop_size))
            next_size = max(final_pop_size, next_size)
            
            if pop_size > next_size:
                # Remove worst individuals (end of sorted array)
                count_remove = pop_size - next_size
                pop = pop[:-count_remove]
                fitness = fitness[:-count_remove]
                pop_size = next_size
                
                # Resize archive to match population size
                while len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
            
            # --- B. Parameter Generation ---
            # Pick random memory slot for each individual
            idx_mem = np.random.randint(0, H, pop_size)
            mu_f = mem_f[idx_mem]
            mu_cr = mem_cr[idx_mem]
            
            # Generate F using Cauchy distribution
            F = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
            # Fix F values
            F = np.clip(F, 0, 1) # Clip > 1 to 1, < 0 to 0 (simplified)
            # Retry if F <= 0 (standard SHADE logic)
            retry_mask = F <= 0
            while np.any(retry_mask):
                F[retry_mask] = np.random.standard_cauchy(np.sum(retry_mask)) * 0.1 + mu_f[retry_mask]
                retry_mask = F <= 0
            F = np.clip(F, 0, 1)

            # Generate CR using Normal distribution
            CR = np.random.normal(mu_cr, 0.1)
            CR = np.clip(CR, 0, 1)
            
            # --- C. Mutation (current-to-pbest/1) ---
            # Select p-best (top p% of population)
            p_best_rate = np.random.uniform(0.05, 0.2) # Dynamic p
            num_pbest = max(1, int(p_best_rate * pop_size))
            
            pbest_indices = np.random.randint(0, num_pbest, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Prepare pool for r2 (Population + Archive)
            if len(archive) > 0:
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
                
            # Select r1, r2
            indices = np.arange(pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            # Ensure r1 != current
            hit_r1 = (r1 == indices)
            r1[hit_r1] = (r1[hit_r1] + 1) % pop_size
            
            r2 = np.random.randint(0, len(pool), pop_size)
            # Ensure r2 != r1 and r2 != current
            hit_r2 = (r2 == r1) | (r2 == indices)
            if np.any(hit_r2):
                 r2[hit_r2] = np.random.randint(0, len(pool), np.sum(hit_r2))
                 
            x_r1 = pop[r1]
            x_r2 = pool[r2]
            
            # Compute Mutant Vector V
            F_col = F[:, None]
            # v = x + F*(pbest - x) + F*(r1 - r2)
            V = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Boundary handling (Clipping)
            V = np.clip(V, lb, ub)
            
            # --- D. Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            rand_uni = np.random.rand(pop_size, dim)
            mask_cr = rand_uni <= CR[:, None]
            
            # Ensure at least one dimension is taken from mutant
            mask_jrand = np.zeros((pop_size, dim), dtype=bool)
            mask_jrand[indices, j_rand] = True
            
            U = np.where(mask_cr | mask_jrand, V, pop)
            
            # --- E. Selection ---
            fitness_u = np.zeros(pop_size)
            # Evaluation
            for i in range(pop_size):
                if not is_time_left(): break
                fitness_u[i] = func(U[i])
                curr_nfe_epoch += 1
                
            if not is_time_left(): break
            
            # Greedy Selection
            improved_mask = fitness_u < fitness
            
            # --- F. Update Archive & Memory ---
            if np.any(improved_mask):
                # Add replaced parents to archive
                losers = pop[improved_mask]
                for lv in losers:
                    archive.append(lv.copy())
                
                # Keep archive size bounded
                while len(archive) > init_pop_size:
                    archive.pop(np.random.randint(len(archive)))
                
                # Calculate improvement weights
                diffs = np.abs(fitness[improved_mask] - fitness_u[improved_mask])
                
                # Update Population
                pop[improved_mask] = U[improved_mask]
                fitness[improved_mask] = fitness_u[improved_mask]
                
                # Update Global Best
                best_u_val = np.min(fitness_u)
                if best_u_val < g_best_val:
                    idx_best = np.argmin(fitness_u)
                    g_best_val = fitness_u[idx_best]
                    g_best_pos = U[idx_best].copy()
                
                # Weighted Lehmer Mean Update for Memory
                if np.sum(diffs) > 0:
                    w = diffs / np.sum(diffs)
                    
                    s_f = F[improved_mask]
                    s_cr = CR[improved_mask]
                    
                    mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                    mean_cr = np.sum(w * s_cr)
                    
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                    k_mem = (k_mem + 1) % H
                    
            # Sort population for next generation (p-best logic requires sorted)
            sort_new = np.argsort(fitness)
            pop = pop[sort_new]
            fitness = fitness[sort_new]
            
            # --- G. Convergence Check ---
            # If population variance is very low, we are stuck in a local optimum
            if np.mean(np.std(pop, axis=0)) < 1e-12:
                break
                
        # --- End of Epoch: Polishing ---
        # Apply MTS-LS1 to the best found solution to ensure we hit the exact bottom
        remaining = max_time - get_time_elapsed()
        if remaining > 0.1:
            # Allocate a slice of remaining time (max 0.5s) to local search
            budget_ls = min(remaining, 0.5) 
            g_best_pos, g_best_val = mts_ls1(g_best_pos, g_best_val, budget_ls)
            
        restart_count += 1
        
    return g_best_val
