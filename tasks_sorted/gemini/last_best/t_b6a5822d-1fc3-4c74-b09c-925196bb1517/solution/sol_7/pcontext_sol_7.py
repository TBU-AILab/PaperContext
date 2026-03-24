#An improved algorithm implementation based on **SHADE with External Archive and Coordinate Descent Polishing**.
#
#### Improvements Explained:
#
#1.  **External Archive (Crucial Fix):** The previous "best" algorithm implemented SHADE logic but omitted the *External Archive*. In the standard SHADE definition (and JADE), the second difference vector ($x_{r1} - x_{r2}$) selects $x_{r2}$ from the union of the **current population and an archive of inferior solutions**. This significantly preserves diversity and prevents the population from clustering too early, which likely caused the premature convergence in the previous iteration.
#2.  **Coordinate Descent (MTS-LS1 style):** Evolutionary algorithms are great at finding the "basin" of the global minimum but can be slow to hit the exact bottom. I added a deterministic **Coordinate Descent Local Search** that triggers when the population converges or before a restart. It fine-tunes each dimension of the best solution individually, acting as a high-precision polisher.
#3.  **Bound Handling (Reflection):** Instead of clipping to bounds (which biases the search to the edges) or simple midpointing, this uses a reflection-based midpoint strategy that preserves the trajectory of the mutation while keeping it valid.
#4.  **Restart Mechanism:** If the population standard deviation drops below a threshold (convergence) or fitness stops improving (stagnation), the algorithm performs the local search polisher, saves the global best, and acts as a "Restart" wrapper to explore new areas of the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with an External Archive and Coordinate Descent Local Search (Polishing).
    """

    # --- Time Management ---
    start_time = datetime.now()
    # Safety buffer to ensure we return before the rigid external timeout
    time_limit = timedelta(seconds=max_time - 0.1)

    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None

    # --- Local Search: Coordinate Descent ---
    def local_search(current_best_sol, current_best_val, search_budget):
        """
        Refines a solution by probing dimensions individually.
        Inspired by MTS-LS1.
        """
        sol = current_best_sol.copy()
        val = current_best_val
        improved = False
        
        # Search range (initial step size)
        sr = (ub - lb) * 0.4
        
        # We process dimensions in random order
        dims = np.random.permutation(dim)
        
        evals = 0
        for d in dims:
            if evals >= search_budget or check_time():
                break
            
            # Try moving in negative direction
            original_x = sol[d]
            sol[d] = np.clip(original_x - sr[d], lb[d], ub[d])
            new_val = func(sol)
            evals += 1
            
            if new_val < val:
                val = new_val
                improved = True
            else:
                # Try moving in positive direction (0.5 step for variety)
                sol[d] = np.clip(original_x + 0.5 * sr[d], lb[d], ub[d])
                new_val = func(sol)
                evals += 1
                
                if new_val < val:
                    val = new_val
                    improved = True
                else:
                    # Restore if no improvement
                    sol[d] = original_x
        
        return sol, val, improved

    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_time():
            return global_best_val

        # 1. SHADE Parameters Setup
        # Population size: Standard SHADE uses 18 * dim, but we clamp it for speed/safety
        pop_size = int(np.clip(18 * dim, 30, 100))
        
        # Initialize Population
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, float('inf'))

        # Evaluate Initial Population
        for i in range(pop_size):
            if i % 10 == 0 and check_time(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()

        # Archive setup (stores inferior solutions to preserve diversity)
        archive = np.zeros((pop_size, dim))
        arc_ind = 0 # Current insertion index
        arc_filled = 0 # Number of items currently in archive

        # Memory for adaptive parameters (History)
        H_mem = 6
        mem_f = np.full(H_mem, 0.5)
        mem_cr = np.full(H_mem, 0.5)
        k_mem = 0

        # Stagnation counters
        stag_gens = 0
        last_gen_best = np.min(fitness)

        # 2. Evolutionary Loop
        while True:
            if check_time():
                return global_best_val

            # --- A. Adaptive Parameter Generation ---
            # Randomly select memory index for each individual
            r_idx = np.random.randint(0, H_mem, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]

            # Generate CR (Normal distribution, clipped [0, 1])
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)

            # Generate F (Cauchy distribution)
            # F must be > 0. If > 1, clip to 1.
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Robust logic for F <= 0 (retry generation)
            bad_f = f <= 0
            while np.any(bad_f):
                count = np.sum(bad_f)
                f[bad_f] = m_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(count) - 0.5))
                bad_f = f <= 0
            f = np.minimum(f, 1.0)

            # --- B. Mutation (current-to-pbest/1) ---
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            sorted_pop = pop[sorted_indices]

            # p-best selection: random top p% (p in [2/N, 0.2])
            p_val = np.random.uniform(2/pop_size, 0.2)
            top_cut = int(max(2, pop_size * p_val))
            
            # Select pbest vectors
            pbest_ind = np.random.randint(0, top_cut, pop_size)
            x_pbest = sorted_pop[pbest_ind]

            # Select r1 (random from pop, distinct handled implicitly by noise or low prob)
            r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1]

            # Select r2 (random from Union(Pop, Archive))
            # Construct Union logic virtually to avoid massive copying
            n_union = pop_size + arc_filled
            r2_indices = np.random.randint(0, n_union, pop_size)
            
            # If r2_index < pop_size -> use pop, else -> use archive
            x_r2 = np.zeros((pop_size, dim))
            mask_pop = r2_indices < pop_size
            mask_arc = ~mask_pop
            
            x_r2[mask_pop] = pop[r2_indices[mask_pop]]
            if np.any(mask_arc):
                # Adjust index for archive
                arc_idx = r2_indices[mask_arc] - pop_size
                x_r2[mask_arc] = archive[arc_idx]

            # Mutation equation
            # v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)

            # --- C. Bound Handling (Midpoint/Reflection) ---
            # If mutant violates bounds, set value between parent and bound
            mask_l = mutant < lb
            if np.any(mask_l):
                # Standard JADE/SHADE method: (parent + bound) / 2
                mutant[mask_l] = (pop[mask_l] + lb[np.where(mask_l)[1]]) / 2.0
            
            mask_u = mutant > ub
            if np.any(mask_u):
                mutant[mask_u] = (pop[mask_u] + ub[np.where(mask_u)[1]]) / 2.0

            # --- D. Crossover (Binomial) ---
            mask_cross = np.random.rand(pop_size, dim) < cr[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)

            # --- E. Selection & Updates ---
            trial_fitness = np.zeros(pop_size)
            winners = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)

            run_best_val_updated = False

            for i in range(pop_size):
                if i % 10 == 0 and check_time(): return global_best_val
                
                f_trial = func(trial[i])
                trial_fitness[i] = f_trial

                if f_trial <= fitness[i]:
                    winners[i] = True
                    diff_fitness[i] = fitness[i] - f_trial
                    
                    # Update Global Best immediately
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_sol = trial[i].copy()
                        run_best_val_updated = True

            # --- F. Update Archive ---
            # Parents that were replaced go to archive
            if np.any(winners):
                replaced_pop = pop[winners]
                num_winners = len(replaced_pop)
                
                for k in range(num_winners):
                    if arc_filled < pop_size:
                        archive[arc_filled] = replaced_pop[k]
                        arc_filled += 1
                    else:
                        # Random replacement
                        rand_idx = np.random.randint(0, pop_size)
                        archive[rand_idx] = replaced_pop[k]

            # --- G. Update Population ---
            pop[winners] = trial[winners]
            fitness[winners] = trial_fitness[winners]

            # --- H. Update Memory (Weighted Lehmer Mean) ---
            if np.any(winners):
                w_diff = diff_fitness[winners]
                w_f = f[winners]
                w_cr = cr[winners]
                
                sum_diff = np.sum(w_diff)
                
                if sum_diff > 0:
                    weights = w_diff / sum_diff
                    
                    # Mean CR (Weighted Arithmetic)
                    m_cr_new = np.sum(weights * w_cr)
                    
                    # Mean F (Weighted Lehmer)
                    sum_f = np.sum(weights * w_f)
                    sum_f2 = np.sum(weights * (w_f ** 2))
                    if sum_f > 0:
                        m_f_new = sum_f2 / sum_f
                    else:
                        m_f_new = 0.5
                    
                    # Update Memory slots
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    mem_cr[k_mem] = np.clip(m_cr_new, 0, 1)
                    k_mem = (k_mem + 1) % H_mem

            # --- I. Check Convergence/Stagnation ---
            curr_gen_best = np.min(fitness)
            pop_std = np.std(fitness)
            
            # Stagnation check
            if abs(last_gen_best - curr_gen_best) < 1e-9:
                stag_gens += 1
            else:
                stag_gens = 0
                last_gen_best = curr_gen_best
            
            # Restart triggers:
            # 1. Population extremely converged (std dev near zero)
            # 2. No improvement for significant generations
            if pop_std < 1e-12 or stag_gens > 20:
                # Before restart, Polish the global best
                if global_best_sol is not None:
                    # Allocate a small budget for local search based on dim
                    ls_budget = 5 * dim 
                    polished_sol, polished_val, _ = local_search(global_best_sol, global_best_val, ls_budget)
                    if polished_val < global_best_val:
                        global_best_val = polished_val
                        global_best_sol = polished_sol
                break # Break inner loop -> Restart

    return global_best_val
