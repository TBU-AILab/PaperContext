#The following algorithm implements **L-SHADE (Linear Success-History based Adaptive Differential Evolution) with Time-Bound Budgeting**.
#
#**Key Improvements:**
#1.  **Time-Aware Budgeting**: Instead of running indefinitely or guessing a static budget, this algorithm calibrates the function's execution speed. It then calculates a dynamic budget (`MaxNFE`) for each restart session.
#    *   If time is plentiful, it sets a standard budget (e.g., $3000 \times D$) to ensure deep convergence, followed by restarts to explore new basins of attraction.
#    *   If time is scarce, it scales the budget to fit the remaining time, ensuring the **Linear Population Size Reduction (LPSR)** strategy completes its full cycle (exploring $\to$ exploiting) within the deadline.
#2.  **Linear Population Size Reduction (LPSR)**: The population size decreases linearly from a large initial value (exploration) to a minimal value (exploitation) based on the progress of evaluations within the session budget.
#3.  **Adaptive Memory**: Uses historical memory ($M_{CR}, M_F$) to adapt mutation and crossover rates based on successful updates, tailored to the specific problem landscape.
#4.  **Restart Mechanism**: Automatically restarts the search if the population stagnates (converges) or if the budget for the session is exhausted, ensuring the algorithm doesn't waste time in local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Time-Bound Budgeting and Restart.
    """
    start_time = time.time()
    
    # --- Configuration ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # Global best tracking
    best_val = float('inf')
    
    # L-SHADE Parameters
    r_N_init = 18       # Initial population size factor (N = r * dim)
    r_arc = 2.6         # Archive size factor
    p_best_rate = 0.11  # Top p-best selection rate
    memory_size = 5     # Historical memory size
    
    # Population sizing
    # Cap initial size to avoid overhead in high dimensions
    pop_size_init = int(r_N_init * dim)
    pop_size_init = max(30, min(pop_size_init, 200)) 
    pop_size_min = 4
    
    # --- 1. Speed Calibration ---
    # Estimate function evaluation time to plan the budget
    n_cal = 3
    t_cal_start = time.time()
    for _ in range(n_cal):
        # Safety check: don't use more than 10% of time for calibration
        if (time.time() - start_time) > max_time * 0.1:
            break
        rnd = min_b + np.random.rand(dim) * (max_b - min_b)
        val = func(rnd)
        if val < best_val:
            best_val = val
    
    elapsed_cal = time.time() - t_cal_start
    avg_eval_time = elapsed_cal / n_cal if n_cal > 0 else 0.0
    if avg_eval_time < 1e-7: avg_eval_time = 1e-7 # Prevent division by zero
    
    # --- 2. Main Optimization Loop (Restarts) ---
    while True:
        # Check remaining time
        current_time = time.time()
        remaining_time = max_time - (current_time - start_time)
        
        if remaining_time < 0.01: # Small margin to return safely
            return best_val
        
        # Calculate Session Budget (MaxNFE)
        # We aim to fit the L-SHADE reduction schedule into the remaining time.
        est_evals = remaining_time / avg_eval_time
        
        # Standard budget: ~3000*D is usually sufficient for convergence on complex problems
        standard_budget = 3000 * dim
        
        # Logic: 
        # If we have lots of time, use standard budget per run to allow convergence, then restart.
        # If we have little time, scale budget to ensure the population reduction happens.
        session_max_nfe = int(min(est_evals, standard_budget))
        
        # Ensure minimal iterations to avoid broken logic
        min_nfe = pop_size_init * 5
        if session_max_nfe < min_nfe:
            session_max_nfe = min_nfe 
            
        # --- Session Initialization ---
        pop_size = pop_size_init
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
        
        # Reset Algorithm State
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        archive = []
        nfe = pop_size
        
        # --- Session Evolution Loop ---
        while nfe < session_max_nfe:
            if (time.time() - start_time) >= max_time:
                return best_val
            
            # A. Linear Population Size Reduction (LPSR)
            progress = nfe / session_max_nfe
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduction: Keep best individuals
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:target_size]]
                fitness = fitness[sort_idx[:target_size]]
                
                # Resize Archive
                curr_arc_max = int(target_size * r_arc)
                if len(archive) > curr_arc_max:
                    # Randomly remove excess
                    n_del = len(archive) - curr_arc_max
                    del_idxs = np.random.choice(len(archive), n_del, replace=False)
                    keep_mask = np.ones(len(archive), dtype=bool)
                    keep_mask[del_idxs] = False
                    archive = [archive[k] for k in range(len(archive)) if keep_mask[k]]
                
                pop_size = target_size
            
            # B. Parameter Generation
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            rand_u = np.random.rand(pop_size)
            f = mu_f + 0.1 * np.tan(np.pi * (rand_u - 0.5))
            
            # F corrections
            bad_f = f <= 0
            while np.any(bad_f):
                cnt = np.sum(bad_f)
                f[bad_f] = mu_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(cnt) - 0.5))
                bad_f = f <= 0
            f = np.minimum(f, 1.0)
            
            # C. Evolution Step
            sorted_indices = np.argsort(fitness)
            p_limit = max(2, int(pop_size * p_best_rate))
            
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros_like(fitness)
            
            S_CR = []
            S_F = []
            S_df = []
            
            for i in range(pop_size):
                x_i = pop[i]
                
                # Mutation: current-to-pbest/1
                p_idx = sorted_indices[np.random.randint(0, p_limit)]
                x_pbest = pop[p_idx]
                
                # r1
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # r2 (from Population U Archive)
                n_total = pop_size + len(archive)
                r2 = np.random.randint(0, n_total)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, n_total)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
                
                # Crossover: Binomial
                j_rand = np.random.randint(0, dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                trial = np.where(mask, mutant, x_i)
                
                # Bound Handling: Midpoint correction
                lower_v = trial < min_b
                upper_v = trial > max_b
                trial[lower_v] = (x_i[lower_v] + min_b[lower_v]) / 2.0
                trial[upper_v] = (x_i[upper_v] + max_b[upper_v]) / 2.0
                
                # Selection
                if (time.time() - start_time) >= max_time:
                    return best_val
                
                f_trial = func(trial)
                nfe += 1
                
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        
                    if f_trial < fitness[i]:
                        S_CR.append(cr[i])
                        S_F.append(f[i])
                        S_df.append(fitness[i] - f_trial)
                        archive.append(x_i.copy())
                else:
                    new_pop[i] = x_i
                    new_fitness[i] = fitness[i]
            
            pop = new_pop
            fitness = new_fitness
            
            # Maintain Archive Size
            max_arc = int(pop_size * r_arc)
            while len(archive) > max_arc:
                archive.pop(np.random.randint(0, len(archive)))
            
            # D. Update Historical Memory
            if S_CR:
                w = np.array(S_df)
                total_imp = np.sum(w)
                if total_imp > 0:
                    w = w / total_imp
                    
                    # Update M_CR (Weighted Mean)
                    m_cr_new = np.sum(w * np.array(S_CR))
                    M_CR[k_mem] = m_cr_new
                    
                    # Update M_F (Weighted Lehmer Mean)
                    s_f_arr = np.array(S_F)
                    num = np.sum(w * (s_f_arr**2))
                    den = np.sum(w * s_f_arr)
                    m_f_new = num / den if den > 0 else 0.5
                    M_F[k_mem] = m_f_new
                    
                    k_mem = (k_mem + 1) % memory_size
            
            # E. Stagnation Check
            # If population has converged to a single point, restart immediately
            if np.std(fitness) < 1e-9:
                break
                
    return best_val
