#The following algorithm implements **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution) with Time-Scheduled Restart**.
#
#**Key Improvements:**
#1.  **Linear Population Reduction**: The population size linearly decreases from a high initial value to a minimum value over the course of the execution. This allows for extensive exploration in the early phase and aggressive exploitation (fine-tuning) in the late phase.
#2.  **Time-Aware Scheduling**: Unlike standard L-SHADE which relies on a fixed number of evaluations, this implementation maps the population reduction schedule to the available `max_time`.
#3.  **Adaptive Restart**: If the population converges before time runs out, the algorithm saves the best solution and performs a restart. The new session adapts its schedule to use exactly the *remaining* time, ensuring optimal use of the computational budget.
#4.  **SHADE Adaptation**: Uses historical memory ($M_{CR}, M_F$) to adapt mutation ($F$) and crossover ($CR$) parameters based on recent successful updates.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Time-Based Population Reduction and Restart.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Bounds setup
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # L-SHADE Parameters
    r_N_init = 18       # Initial population size factor (N = r * dim)
    r_arc = 2.6         # Archive size ratio
    p_best_rate = 0.11  # Top p-best selection rate
    memory_size = 5     # Size of historical memory
    
    # Population sizing constraints
    # Cap initial population to ensure iterations are fast enough
    # but large enough for exploration.
    pop_size_init = int(r_N_init * dim)
    pop_size_init = max(30, min(pop_size_init, 200)) 
    pop_size_min = 4
    
    # Global Best Tracking
    best_val = float('inf')
    
    # --- Main Optimization Loop (Restart Mechanism) ---
    while True:
        # Check overall time budget
        current_time = time.time()
        elapsed = current_time - start_time
        remaining_time = max_time - elapsed
        
        # If remaining time is negligible (e.g., < 50ms), stop
        if remaining_time < 0.05:
            return best_val
        
        # Define current session parameters
        # We treat the remaining time as the budget for this restart session
        session_start = current_time
        session_duration = remaining_time
        
        # --- Initialization for Session ---
        pop_size = pop_size_init
        pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            # Strict time check during initialization
            if (time.time() - start_time) >= max_time:
                return best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
        
        # Initialize Memories
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        
        # Initialize Archive (stores inferior parents)
        archive = []
        
        # --- Session Generation Loop ---
        while True:
            now = time.time()
            if (now - start_time) >= max_time:
                return best_val
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress based on time consumed in this session
            progress = (now - session_start) / session_duration
            if progress > 1.0: progress = 1.0
            
            # Calculate target population size
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            # Reduce population if target is smaller than current
            if target_size < pop_size:
                # Sort by fitness (worst at the end) and truncate
                sorted_indices = np.argsort(fitness)
                keep_indices = sorted_indices[:target_size]
                
                pop = pop[keep_indices]
                fitness = fitness[keep_indices]
                pop_size = target_size
                
                # Resize Archive
                current_arc_max = int(pop_size * r_arc)
                if len(archive) > current_arc_max:
                    # Randomly remove elements to fit size
                    num_to_del = len(archive) - current_arc_max
                    del_indices = np.random.choice(len(archive), num_to_del, replace=False)
                    # Rebuild archive
                    new_archive = [archive[k] for k in range(len(archive)) if k not in del_indices]
                    archive = new_archive

            # 2. Convergence Check (Early Restart)
            # If population is extremely close or fitness is flat, restart to use remaining time better
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break
                
            # 3. Preparation for Evolution
            sorted_indices = np.argsort(fitness) # For p-best selection
            
            S_CR = [] # Successful CRs
            S_F = []  # Successful Fs
            S_df = [] # Fitness improvements
            
            new_pop = np.empty_like(pop)
            new_fitness = np.empty_like(fitness)
            
            # 4. Evolution Cycle
            for i in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_val
                
                target = pop[i]
                
                # -- Parameter Adaptation --
                r_idx = np.random.randint(0, memory_size)
                mu_cr = M_CR[r_idx]
                mu_f = M_F[r_idx]
                
                # Generate CR (Normal dist, clipped)
                if mu_cr == -1:
                    cr = 0.0
                else:
                    cr = np.random.normal(mu_cr, 0.1)
                    cr = np.clip(cr, 0.0, 1.0)
                
                # Generate F (Cauchy dist)
                while True:
                    f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                    if f > 0: break
                if f > 1.0: f = 1.0
                
                # -- Mutation: current-to-pbest/1 --
                # Select p-best
                p_count = max(2, int(pop_size * p_best_rate))
                pbest_idx = sorted_indices[np.random.randint(0, p_count)]
                x_pbest = pop[pbest_idx]
                
                # Select r1 (distinct from i)
                while True:
                    r1_idx = np.random.randint(0, pop_size)
                    if r1_idx != i: break
                x_r1 = pop[r1_idx]
                
                # Select r2 (distinct from i, r1) from Union(Pop, Archive)
                pool_size = pop_size + len(archive)
                while True:
                    r2_idx = np.random.randint(0, pool_size)
                    if r2_idx < pop_size:
                        if r2_idx != i and r2_idx != r1_idx:
                            x_r2 = pop[r2_idx]
                            break
                    else:
                        x_r2 = archive[r2_idx - pop_size]
                        break
                
                mutant = target + f * (x_pbest - target) + f * (x_r1 - x_r2)
                
                # -- Crossover --
                j_rand = np.random.randint(0, dim)
                cross_mask = np.random.rand(dim) < cr
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, target)
                
                # -- Bound Handling (Mean Correction) --
                # Instead of simple clipping, place violation between target and bound
                # This preserves distribution better than clipping to edge
                lower_mask = trial < min_b
                upper_mask = trial > max_b
                if np.any(lower_mask):
                    trial[lower_mask] = (target[lower_mask] + min_b[lower_mask]) / 2.0
                if np.any(upper_mask):
                    trial[upper_mask] = (target[upper_mask] + max_b[upper_mask]) / 2.0
                
                # -- Selection --
                f_trial = func(trial)
                
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                    
                    # Store success data
                    S_CR.append(cr)
                    S_F.append(f)
                    S_df.append(fitness[i] - f_trial)
                    
                    # Add replaced individual to archive
                    archive.append(target.copy())
                else:
                    new_pop[i] = target
                    new_fitness[i] = fitness[i]
            
            # Update Population
            pop = new_pop
            fitness = new_fitness
            
            # Maintain Archive Size
            curr_arc_max = int(pop_size * r_arc)
            while len(archive) > curr_arc_max:
                archive.pop(np.random.randint(0, len(archive)))
                
            # 5. Update Historical Memory
            if len(S_df) > 0:
                S_CR = np.array(S_CR)
                S_F = np.array(S_F)
                S_df = np.array(S_df)
                
                # Weighted Lehmer Mean based on fitness improvement
                total_imp = np.sum(S_df)
                if total_imp > 0:
                    w = S_df / total_imp
                    
                    # Update M_CR
                    m_cr_new = np.sum(w * S_CR)
                    if M_CR[k_mem] == -1 or M_CR[k_mem] == 0:
                        M_CR[k_mem] = m_cr_new
                    else:
                        M_CR[k_mem] = m_cr_new
                        
                    # Update M_F
                    num = np.sum(w * (S_F ** 2))
                    den = np.sum(w * S_F)
                    if den > 0:
                        M_F[k_mem] = num / den
                    else:
                        M_F[k_mem] = 0.5
                        
                    k_mem = (k_mem + 1) % memory_size
            
            # Check if session time completed (Progress 100%)
            if progress >= 1.0:
                break
                
    return best_val
