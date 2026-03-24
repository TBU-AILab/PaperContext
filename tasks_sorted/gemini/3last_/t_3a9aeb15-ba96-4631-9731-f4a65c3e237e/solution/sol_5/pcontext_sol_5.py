#An improved algorithm is proposed below. It implements **Vectorized L-SHADE with Time-Based Linear Population Size Reduction (LPSR) and Soft Restarts**.
#
#### Key Improvements
#1.  **Vectorization**: Instead of looping through the population to generate mutants and crossover candidates, the algorithm uses NumPy matrix operations. This significantly reduces the overhead of the algorithm itself, allowing more function evaluations within the `max_time`, which is crucial for Python-based optimization.
#2.  **Time-Driven LPSR**: The Linear Population Size Reduction strategy is mapped directly to the elapsed time. The population size shrinks from an initial exploration phase ($N \approx 18 \times D$) to an exploitation phase ($N=4$) as the time budget is consumed.
#3.  **Adaptive Session Management**: The algorithm treats the available time as a "session budget". If the population stagnates (converges) early, it triggers a **Soft Restart**:
#    *   It preserves the global best solution (Elitism).
#    *   It resets the population and historical memory to explore new basins of attraction.
#    *   It dynamically allocates the *remaining* time as the budget for the new session, rescaling the LPSR schedule accordingly.
#4.  **L-SHADE Mechanism**: Utilizes historical memory ($M_{CR}, M_F$) to adapt the Differential Evolution parameters ($F$ and $CR$) based on successful updates, with an external archive to maintain diversity.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Vectorized L-SHADE with Time-Based Linear Population Size Reduction and Soft Restarts.
    """
    start_time = time.time()
    
    # --- Configuration ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # L-SHADE Parameters
    # Initial population size: balance between exploration and iteration speed.
    # We cap N_init to ensure the loop is fast enough even in high dimensions.
    base_N = 18 * dim
    N_init_max = 300 
    N_init = int(max(30, min(base_N, N_init_max)))
    N_min = 4
    
    # Historical Memory Size
    H = 6
    
    # Archive Size Factor
    A_rate = 2.6
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol_global = None
    
    # --- Session State ---
    # We treat the run as a sequence of "sessions" (restarts).
    session_start = start_time
    # Initial budget is the full max_time
    current_session_budget = max_time
    
    # --- Initialization ---
    pop_size = N_init
    pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
    fitness = np.zeros(pop_size)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol_global = pop[i].copy()
            
    # Initialize Memory and Archive
    M_CR = np.full(H, 0.5)
    M_F = np.full(H, 0.5)
    k_mem = 0
    archive = [] 
    
    # --- Main Optimization Loop ---
    while True:
        # 1. Time & Budget Check
        current_time = time.time()
        elapsed_global = current_time - start_time
        remaining_time = max_time - elapsed_global
        
        # Buffer to ensure clean return
        if remaining_time < 0.02:
            return best_val
            
        # Calculate Progress within current session
        session_elapsed = current_time - session_start
        progress = session_elapsed / current_session_budget
        if progress > 1.0: progress = 1.0
        
        # 2. Linear Population Size Reduction (LPSR)
        target_N = int(round((N_min - N_init) * progress + N_init))
        target_N = max(N_min, target_N)
        
        if pop_size > target_N:
            # Reduce Population (keep best)
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:target_N]]
            fitness = fitness[sort_indices[:target_N]]
            pop_size = target_N
            
            # Reduce Archive
            max_arc_size = int(pop_size * A_rate)
            while len(archive) > max_arc_size:
                # Random removal
                archive.pop(np.random.randint(0, len(archive)))
        
        # 3. Convergence / Restart Check
        # Restart if:
        # a) Population has converged (std dev ~ 0)
        # b) LPSR schedule finished (progress ~ 100%) but time remains
        
        is_converged = np.std(fitness) < 1e-9
        is_schedule_done = progress > 0.95
        
        if (is_converged or is_schedule_done):
            # Only restart if there is meaningful time left (e.g. > 5% total or > 0.5s)
            if remaining_time > max(0.2, max_time * 0.05):
                # Setup New Session
                session_start = time.time()
                current_session_budget = remaining_time
                
                # Scale initial population for the new session based on remaining budget
                # If budget is tight, start smaller
                scale_factor = 1.0
                if current_session_budget < (max_time * 0.3):
                    scale_factor = 0.5
                
                N_init_restart = int(max(30, N_init * scale_factor))
                pop_size = N_init_restart
                
                # Re-initialize Population
                pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
                fitness = np.zeros(pop_size)
                
                # Inject Global Best (Elitism)
                pop[0] = best_sol_global
                fitness[0] = best_val
                
                # Reset Internals
                M_CR.fill(0.5)
                M_F.fill(0.5)
                k_mem = 0
                archive = []
                
                # Evaluate the rest
                for i in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return best_val
                    val = func(pop[i])
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_sol_global = pop[i].copy()
                
                continue
            elif is_converged:
                # If converged and no time to restart, we are done
                return best_val

        # 4. Parameter Generation (Vectorized)
        # Randomly select memory indices
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # Generate CR ~ Normal(m_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        # If CR is very close to 0, fix to ensure at least some crossover?
        # Standard SHADE relies on j_rand to ensure 1 dim changes.
        
        # Generate F ~ Cauchy(m_f, 0.1)
        # Approximate Cauchy using tan(uniform)
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F violations
        # If F <= 0, regenerate. If F > 1, clamp to 1.
        while True:
            bad_f_mask = f <= 0
            if not np.any(bad_f_mask):
                break
            # Regenerate bad ones
            n_bad = np.sum(bad_f_mask)
            f[bad_f_mask] = m_f[bad_f_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
            
        f = np.minimum(f, 1.0)
        
        # 5. Mutation: current-to-pbest/1 (Vectorized)
        # Sort population for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # Select p-best (top 11%)
        p_best_rate = 0.11
        p_num = max(2, int(pop_size * p_best_rate))
        top_p_indices = sorted_indices[:p_num]
        
        # pbest indices for each individual
        pbest_choices = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (distinct from i)
        # Approximate distinctness by random choice and simple collision fix
        r1_indices = np.random.randint(0, pop_size, pop_size)
        collision_mask = r1_indices == np.arange(pop_size)
        r1_indices[collision_mask] = (r1_indices[collision_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (distinct from i and r1, from Pop U Archive)
        if len(archive) > 0:
            # Create a pool array. Note: Creating array every iter has overhead, 
            # but archive is usually small compared to func eval.
            pool = np.vstack((pop, np.array(archive)))
        else:
            pool = pop
        
        pool_size = len(pool)
        r2_indices = np.random.randint(0, pool_size, pop_size)
        
        # Simple collision fix for r2
        # (Perfect exclusion is expensive, this is statistically sufficient)
        c2 = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
        if np.any(c2):
            r2_indices[c2] = np.random.randint(0, pool_size, np.sum(c2))
            
        x_r2 = pool[r2_indices]
        
        # Compute Mutant Vectors
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        # Reshape f for broadcasting: (N,) -> (N, 1)
        f_matrix = f[:, np.newaxis]
        mutant = pop + f_matrix * (x_pbest - pop) + f_matrix * (x_r1 - x_r2)
        
        # 6. Crossover (Binomial)
        # Mask: True if crossover happens (taking from mutant)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
        # Ensure at least one dim is from mutant
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 7. Bound Handling (Midpoint Correction)
        # If out of bounds, place between parent and bound
        lower_vio = trial < min_b
        upper_vio = trial > max_b
        
        if np.any(lower_vio):
            # Broadcast min_b to shape (N, dim)
            bound_matrix = np.tile(min_b, (pop_size, 1))
            trial[lower_vio] = (pop[lower_vio] + bound_matrix[lower_vio]) / 2.0
            
        if np.any(upper_vio):
            bound_matrix = np.tile(max_b, (pop_size, 1))
            trial[upper_vio] = (pop[upper_vio] + bound_matrix[upper_vio]) / 2.0
            
        # 8. Selection & Evaluation
        new_fitness = np.zeros(pop_size)
        
        # Storage for successful updates
        S_CR = []
        S_F = []
        S_df = []
        
        # Evaluate Trials
        # We loop here because func takes 1D array
        for i in range(pop_size):
            # Periodic time check within generation to prevent overrun
            if i % 10 == 0:
                if (time.time() - start_time) >= max_time:
                    return best_val
                    
            f_trial = func(trial[i])
            new_fitness[i] = f_trial
            
            if f_trial <= fitness[i]:
                # We will update pop after the loop
                if f_trial < fitness[i]:
                    S_CR.append(cr[i])
                    S_F.append(f[i])
                    S_df.append(fitness[i] - f_trial)
                    archive.append(pop[i].copy())
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol_global = trial[i].copy()
            else:
                # If not better, we essentially keep old. 
                # We handle this by masking updates below.
                pass
        
        # Update Population
        improve_mask = new_fitness <= fitness
        pop[improve_mask] = trial[improve_mask]
        fitness[improve_mask] = new_fitness[improve_mask]
        
        # 9. Update Historical Memory
        if len(S_df) > 0:
            S_CR = np.array(S_CR)
            S_F = np.array(S_F)
            S_df = np.array(S_df)
            
            # Weighted Lehmer Mean
            total_imp = np.sum(S_df)
            if total_imp > 0:
                w = S_df / total_imp
                
                # Update M_CR
                m_cr_new = np.sum(w * S_CR)
                M_CR[k_mem] = m_cr_new
                
                # Update M_F
                num = np.sum(w * (S_F ** 2))
                den = np.sum(w * S_F)
                m_f_new = num / den if den > 0 else 0.5
                M_F[k_mem] = m_f_new
                
                k_mem = (k_mem + 1) % H
    
    return best_val
