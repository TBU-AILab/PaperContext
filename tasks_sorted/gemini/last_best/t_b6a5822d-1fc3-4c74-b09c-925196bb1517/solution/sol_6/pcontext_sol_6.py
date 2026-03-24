#An improved algorithm implementation using **SHADE with External Archive, Opposition-Based Learning (OBL), and Local Search Polishing**.
#
##### Improvements Explained:
#1.  **SHADE with External Archive**: Implements the full SHADE specification including an external archive. The archive stores inferior solutions replaced during selection, which are then used in the mutation strategy (`current-to-pbest/1`) to maintain diversity and prevent premature convergence.
#2.  **Opposition-Based Learning (OBL)**: During population initialization, the algorithm evaluates both random individuals and their "opposite" counterparts ($lb + ub - x$). This explores the search space more thoroughly at the start, often landing closer to the basin of attraction.
#3.  **Midpoint-Target Bound Handling**: Instead of simple clipping (which clumps solutions at bounds), this algorithm sets particles violating bounds to the midpoint between their parent position and the bound. This preserves the search momentum.
#4.  **Local Search Polishing**: Before restarting due to stagnation or convergence, a short local search (Random Walk with shrinking step size) is performed on the global best solution. This helps refine the solution quality ("polishing") by exploiting the local gradient-free landscape.
#5.  **Restart Mechanism**: If the population converges (low variance) or stagnates (no improvement), the algorithm restarts with a new population while preserving the global best, ensuring continuous search within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE with External Archive, Opposition-Based Learning,
    and Local Search Polishing.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Buffer to ensure clean return before hard cutoff
    time_limit = timedelta(seconds=max_time - 0.05)
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracking
    global_best_fit = float('inf')
    global_best_sol = None
    
    # SHADE Parameters
    H = 5  # History size
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_time(): return global_best_fit
        
        # 1. Population Sizing (Adaptive)
        # Standard SHADE recommendation: ~18 * dim, clipped to reasonable bounds
        pop_size = int(np.clip(18 * dim, 30, 150))
        
        # 2. Initialization with OBL
        # Generate Random Population
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if i % 20 == 0 and check_time(): return global_best_fit
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()
                
        # Opposition-Based Learning (OBL)
        # Evaluate opposite points: x' = lb + ub - x
        # This increases initial exploration
        pop_opp = lb + ub - pop
        pop_opp = np.clip(pop_opp, lb, ub)
        
        for i in range(pop_size):
            if i % 20 == 0 and check_time(): return global_best_fit
            val = func(pop_opp[i])
            # Greedy selection between random and opposite
            if val < fitness[i]:
                fitness[i] = val
                pop[i] = pop_opp[i]
                if val < global_best_fit:
                    global_best_fit = val
                    global_best_sol = pop_opp[i].copy()
                    
        # 3. Algorithm State Initialization
        # Memory for SHADE adaptation
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive
        archive = np.zeros((pop_size, dim))
        arc_count = 0
        
        # Stagnation Counter
        stag_count = 0
        
        # --- Evolutionary Generation Loop ---
        while True:
            if check_time(): return global_best_fit
            
            # --- A. Parameter Adaptation ---
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Handle F <= 0 (retry) and F > 1 (clip)
            bad_f = f <= 0
            while np.any(bad_f):
                f[bad_f] = m_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f)) - 0.5))
                bad_f = f <= 0
            f = np.minimum(f, 1.0)
            
            # --- B. Mutation: current-to-pbest/1 ---
            # Sort population to find p-best
            sorted_idx = np.argsort(fitness)
            sorted_pop = pop[sorted_idx]
            
            # p-best selection: random top p% (p in [2/N, 0.2])
            p_val = np.random.uniform(2/pop_size, 0.2)
            top_cut = int(max(2, pop_size * p_val))
            
            pbest_indices = np.random.randint(0, top_cut, pop_size)
            x_pbest = sorted_pop[pbest_indices]
            
            # r1 selection: random from pop, r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            # Resolve collision r1 == i
            col = r1 == np.arange(pop_size)
            r1[col] = (r1[col] + 1) % pop_size
            x_r1 = pop[r1]
            
            # r2 selection: random from (Pop U Archive), r2 != r1, r2 != i
            # Construct Union
            if arc_count > 0:
                union_pop = np.vstack((pop, archive[:arc_count]))
            else:
                union_pop = pop
            
            r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2]
            
            # Mutation Equation
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # --- Bound Handling: Midpoint Target ---
            # If mutant < lb, set to (parent + lb) / 2 (better than clipping)
            mask_l = mutant < lb
            if np.any(mask_l):
                rows, cols = np.where(mask_l)
                mutant[rows, cols] = (pop[rows, cols] + lb[cols]) / 2.0
            
            mask_u = mutant > ub
            if np.any(mask_u):
                rows, cols = np.where(mask_u)
                mutant[rows, cols] = (pop[rows, cols] + ub[cols]) / 2.0
            
            # --- C. Crossover: Binomial ---
            mask_c = np.random.rand(pop_size, dim) < cr[:, None]
            # Ensure at least one parameter comes from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_c[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_c, mutant, pop)
            
            # --- D. Evaluation & Selection ---
            trial_fit = np.zeros(pop_size)
            winners = np.zeros(pop_size, dtype=bool)
            fit_diff = np.zeros(pop_size)
            
            iter_improved_global = False
            
            for i in range(pop_size):
                if i % 20 == 0 and check_time(): return global_best_fit
                
                val = func(trial[i])
                trial_fit[i] = val
                
                if val <= fitness[i]:
                    winners[i] = True
                    fit_diff[i] = fitness[i] - val
                    if val < global_best_fit:
                        global_best_fit = val
                        global_best_sol = trial[i].copy()
                        iter_improved_global = True
            
            # --- E. Archive Update ---
            # Add replaced parents to archive
            if np.any(winners):
                replaced = pop[winners].copy()
                for vec in replaced:
                    if arc_count < pop_size:
                        archive[arc_count] = vec
                        arc_count += 1
                    else:
                        # Randomly replace
                        idx = np.random.randint(0, pop_size)
                        archive[idx] = vec
            
            # Update Population
            pop[winners] = trial[winners]
            fitness[winners] = trial_fit[winners]
            
            # --- F. Memory Update (Weighted Lehmer Mean) ---
            if np.any(winners):
                w_diff = fit_diff[winners]
                w_f = f[winners]
                w_cr = cr[winners]
                
                sum_diff = np.sum(w_diff)
                if sum_diff > 0:
                    weights = w_diff / sum_diff
                    
                    # Update M_CR (Weighted Arithmetic Mean)
                    m_cr_new = np.sum(weights * w_cr)
                    
                    # Update M_F (Weighted Lehmer Mean)
                    sum_f = np.sum(weights * w_f)
                    sum_f2 = np.sum(weights * (w_f ** 2))
                    if sum_f > 0:
                        m_f_new = sum_f2 / sum_f
                    else:
                        m_f_new = 0.5
                        
                    mem_cr[k_mem] = np.clip(m_cr_new, 0, 1)
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # --- G. Check Stagnation & Convergence ---
            if iter_improved_global:
                stag_count = 0
            else:
                stag_count += 1
            
            std_dev = np.std(fitness)
            
            # Restart condition: Converged (std ~ 0) or Stagnated (> 40 gens)
            if std_dev < 1e-9 or stag_count > 40:
                
                # --- H. Local Search Polishing ---
                # Before restart, try to squeeze more out of the global best
                # using a simple adaptive random walk
                ls_curr = global_best_sol.copy()
                ls_val = global_best_fit
                step_size = (ub - lb) * 0.05
                
                # Fixed budget for local search (e.g., 50 evals)
                for _ in range(50):
                    if check_time(): return global_best_fit
                    
                    perturb = np.random.normal(0, 1, dim) * step_size
                    neighbor = ls_curr + perturb
                    neighbor = np.clip(neighbor, lb, ub)
                    
                    val = func(neighbor)
                    if val < global_best_fit:
                        global_best_fit = val
                        global_best_sol = neighbor.copy()
                        ls_curr = neighbor
                        ls_val = val
                    elif val < ls_val:
                        # Accept local improvement
                        ls_curr = neighbor
                        ls_val = val
                    else:
                        # Reduce step size
                        step_size *= 0.5
                
                break # Break inner loop to trigger restart

    return global_best_fit
