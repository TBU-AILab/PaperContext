#The following is a fully operational and self-contained algorithm using a **Success-History based Adaptive Differential Evolution (SHADE)** strategy with **Restarts**.
#
#### Algorithm Description
#To achieve the minimum output value within a limited time, this algorithm implements **SHADE with Linear Population Sizing and Restarts**. This is an advanced variant of Differential Evolution that self-adapts its control parameters ($F$ and $CR$) based on the success of previous generations, making it highly effective for diverse and unknown fitness landscapes.
#
#1.  **Strategy (DE/current-to-pbest/1/bin)**: The mutation strategy moves individuals towards a solution selected from the top $p\%$ best individuals ($p$-best). This balances exploitation (convergence) and exploration better than targeting the single best solution.
#2.  **External Archive**: An archive stores recently replaced inferior solutions. These are used in the mutation step (as the second difference vector) to maintain population diversity and prevent premature convergence.
#3.  **Self-Adaptive Parameters (SHADE)**: Instead of fixed or random parameters, mutation factor $F$ and crossover rate $CR$ are adapted using a historical memory ($M_F, M_{CR}$). Successful parameters that lead to fitness improvements are used to update this memory, guiding the algorithm towards optimal settings automatically.
#4.  **Restart Mechanism**: The algorithm detects stagnation (no improvement in best fitness) or convergence (low population variance). Upon detection, it restarts with a fresh population but carries over the global best solution (Elitism) to ensure monotonic improvement.
#5.  **Adaptive Population**: The population size is heuristically determined based on the dimension ($15\sqrt{D}$) to ensure adequate search density without exceeding the time budget.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Success-History based Adaptive 
    Differential Evolution (SHADE) with Restarts.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker across all restarts
    global_best_fit = float('inf')
    global_best_sol = None
    
    # Helper to check time budget
    def is_time_left():
        return (time.time() - start_time) < max_time

    # --- Main Restart Loop ---
    # Restarts the population if it converges or stagnates.
    while is_time_left():
        
        # Adaptive Population Size
        # Heuristic: 15 * sqrt(D), clamped to [30, 80] for efficiency
        pop_size = int(15 * np.sqrt(dim))
        pop_size = max(30, min(80, pop_size))
        
        # --- SHADE Memory Initialization ---
        # Memory stores successful F and CR values to guide future generation.
        # H (history size) = 5
        mem_size = 5
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous restarts
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_fit
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if not is_time_left(): return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()
                
        # External Archive for 'current-to-pbest' diversity
        # Stores inferior solutions replaced by better offspring
        archive = []
        
        # Stagnation tracking
        stag_counter = 0
        prev_min_fit = np.min(fitness)
        
        # --- Evolution Loop ---
        while is_time_left():
            # 1. Sort Population
            # Necessary for 'current-to-pbest' to easily pick top p%
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # 2. Convergence / Stagnation Check
            current_min = fitness[0]
            
            # Check if population variance is zero or no improvement seen
            if np.std(fitness) < 1e-8 or abs(current_min - prev_min_fit) < 1e-8:
                stag_counter += 1
            else:
                stag_counter = 0
                prev_min_fit = current_min
                
            if stag_counter > 30: # Trigger restart if stuck
                break
                
            # 3. Parameter Generation (SHADE Strategy)
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, mem_size, pop_size)
            mu_f = mem_F[r_idx]
            mu_cr = mem_CR[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr_g = np.random.normal(mu_cr, 0.1)
            cr_g = np.clip(cr_g, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1), clipped (0, 1]
            # Cauchy = mu + scale * tan(pi * (rand - 0.5))
            f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f_g = np.clip(f_g, 0.1, 1.0) 
            
            # 4. Mutation: DE/current-to-pbest/1
            # Select p-best (top 15% or at least 2)
            p_limit = max(2, int(0.15 * pop_size))
            p_best_idxs = np.random.randint(0, p_limit, pop_size)
            x_pbest = pop[p_best_idxs]
            
            # Select r1 (from Population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (from Population U Archive)
            if len(archive) > 0:
                arc_arr = np.array(archive)
                n_arc = len(archive)
                # Generate random indices covering both pop and archive
                r2_raw_idxs = np.random.randint(0, pop_size + n_arc, pop_size)
                
                # Construct union (concatenation is efficient for these sizes)
                union_pop = np.concatenate((pop, arc_arr), axis=0)
                x_r2 = union_pop[r2_raw_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = pop[r2_idxs]
            
            # Mutation Equation
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            # Note: 'pop' is sorted, so pop[i] is current 'x'
            mutant = pop + f_g[:, None] * (x_pbest - pop) + f_g[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover
            # Binomial crossover
            cross_points = np.random.rand(pop_size, dim) < cr_g[:, None]
            # Ensure at least one parameter comes from mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_points[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_points, mutant, pop)
            
            # 6. Evaluation & Selection
            succ_F = []
            succ_CR = []
            diff_fitness = []
            
            for i in range(pop_size):
                if not is_time_left(): return global_best_fit
                
                f_trial = func(trial[i])
                f_old = fitness[i]
                
                if f_trial <= f_old:
                    # Successful Update
                    
                    # Add parent to archive (if strictly better)
                    if f_trial < f_old:
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            # If archive full, replace random solution
                            rem = np.random.randint(0, pop_size)
                            archive[rem] = pop[i].copy()
                        
                        # Record success for memory update
                        succ_F.append(f_g[i])
                        succ_CR.append(cr_g[i])
                        diff_fitness.append(f_old - f_trial)
                    
                    # Update Population
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial[i].copy()
            
            # 7. Update SHADE Memory
            if len(succ_F) > 0:
                succ_F = np.array(succ_F)
                succ_CR = np.array(succ_CR)
                weights = np.array(diff_fitness)
                total_w = np.sum(weights)
                
                if total_w > 0:
                    weights /= total_w
                    
                    # Weighted Lehmer Mean for F
                    mean_F = np.sum(weights * (succ_F ** 2)) / np.sum(weights * succ_F)
                    
                    # Weighted Mean for CR
                    mean_CR = np.sum(weights * succ_CR)
                    
                    # Update Memory at current pointer
                    mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_F
                    mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_CR
                    
                    k_mem = (k_mem + 1) % mem_size

    return global_best_fit
