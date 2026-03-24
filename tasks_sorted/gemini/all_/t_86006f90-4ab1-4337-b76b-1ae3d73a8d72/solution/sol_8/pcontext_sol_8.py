#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 32.12476466108296
#2. output value is: 21.050545782471445
#3. output value is: 25.82729090260267
#...
#8. output value is: 105.40816267667486
#
#The best performance so far was achieved by **Algorithm 2 (SHADE-RRLS)** with a value of **21.05**. This algorithm successfully combined **SHADE** parameter adaptation, **Reflection** boundary handling, and an **MTS-based Local Search** triggered upon stagnation.
#
#The following improved algorithm, **MaSHADE-R (Memetic Adaptive SHADE with Restarts)**, builds upon the success of Algorithm 2 with specific enhancements to improve exploitation capability and reduce the value further.
#
##### Improvements Explanation:
#1.  **Memetic Polishing (Exploitation)**: While Algorithm 2 only applied Local Search (LS) when the population stagnated, MaSHADE-R applies a **light-weight LS** (low budget) to the best individual **immediately** whenever a new global best is found (or improvement is detected). This "Memetic" behavior ensures that the leader of the population is always at the local minimum of its current basin, accelerating the convergence of the rest of the population (via `current-to-pbest` mutation) towards the true valley bottom.
#2.  **Symmetric MTS-LS**: The Local Search logic is refined to be fully symmetric (checking `x-d` and `x+d` with equal weight) and more robust than the previous implementation.
#3.  **Optimized Population Size**: Based on the comparison between Algo 2 (pop ~10*dim) and Algo 7 (pop ~5*dim), a moderate population size (`10 * dim`, capped at 60) is selected to balance diversity with generation speed.
#4.  **Reflection Boundary Handling**: Retained as it proved superior to clipping in previous runs.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using MaSHADE-R (Memetic Adaptive SHADE with Restarts).
    
    MaSHADE-R integrates SHADE (Success-History Adaptive Differential Evolution) with:
    1. Memetic Strategy: Periodic light-weight local search on the leader to accelerate convergence.
    2. Deep Polishing: Heavy local search triggered upon stagnation.
    3. Restart Mechanism: Escapes local optima by Soft Restart (keeping best).
    4. Reflection: Handles boundaries by bouncing back to preserve diversity.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper: Check Time ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Pre-calc bounds for vectorized operations ---
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population Size: Based on Algo 2 success, kept moderate.
    # 10*dim provides enough diversity without slowing down generations.
    pop_size = int(max(20, 10 * dim))
    if pop_size > 60: pop_size = 60
    
    # SHADE Parameters
    H = 6               # History memory size
    p_min = 2 / pop_size
    p_max = 0.2
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Helper: Boundary Reflection ---
    def reflect(x):
        """Reflects out-of-bound values back into the domain."""
        # Lower Check
        mask_l = x < min_b
        while np.any(mask_l):
            x[mask_l] = 2 * min_b[mask_l] - x[mask_l]
            mask_l = x < min_b
        
        # Upper Check
        mask_u = x > max_b
        while np.any(mask_u):
            x[mask_u] = 2 * max_b[mask_u] - x[mask_u]
            mask_u = x > max_b
            
        return np.clip(x, min_b, max_b)

    # --- Helper: Symmetric MTS Local Search ---
    def local_search(center, val, budget):
        """
        Symmetric Multiple Trajectory Search (MTS).
        Refines 'center' by searching along coordinate axes.
        """
        x = center.copy()
        current_val = val
        
        # Initial Search Range: Start fine-grained (20% of domain)
        sr = diff_b * 0.2 
        
        evals_used = 0
        improved = True
        
        # Continue while we have budget, range is significant, and time permits
        while evals_used < budget and np.max(sr) > 1e-13 and not check_time():
            if not improved:
                # If a full pass over all dims yields no improvement, shrink range
                sr *= 0.5
            
            improved = False
            # Randomize dimension order to prevent bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if check_time() or evals_used >= budget: break
                
                original = x[d]
                
                # 1. Try Negative Direction
                x[d] = original - sr[d]
                x = reflect(x)
                v = func(x)
                evals_used += 1
                
                if v < current_val:
                    current_val = v
                    improved = True
                else:
                    # 2. Try Positive Direction
                    x[d] = original + sr[d]
                    x = reflect(x)
                    v = func(x)
                    evals_used += 1
                    
                    if v < current_val:
                        current_val = v
                        improved = True
                    else:
                        # Revert if both fail
                        x[d] = original
                        
        return x, current_val

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        # 1. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best into the new population (Soft Restart)
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_val
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_val
            
            # Skip if already evaluated (Elitism)
            if fitness[i] != float('inf'): continue 
            
            f = func(pop[i])
            fitness[i] = f
            if f < best_val:
                best_val = f
                best_sol = pop[i].copy()
                
        # 2. SHADE Memory Reset
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Stagnation Counters
        no_improv_count = 0
        last_best_fit = np.min(fitness)
        
        # --- Generation Loop ---
        while not check_time():
            # Sort Population (Best at index 0)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            current_best_fit = fitness[0]
            
            # Check for improvement
            if np.abs(current_best_fit - last_best_fit) < 1e-12:
                no_improv_count += 1
            else:
                # Improvement detected
                no_improv_count = 0
                last_best_fit = current_best_fit
                
                # --- MEMETIC STRATEGY ---
                # "Strike while the iron is hot."
                # Perform a quick, light-weight Local Search on the new leader.
                # This ensures the population is guided by a high-quality local minimum.
                ls_budget = 5 * dim 
                new_sol, new_val = local_search(pop[0], fitness[0], ls_budget)
                
                if new_val < fitness[0]:
                    fitness[0] = new_val
                    pop[0] = new_sol
                    if new_val < best_val:
                        best_val = new_val
                        best_sol = new_sol.copy()
            
            # Check Restart Conditions
            # 1. Population converged (Low Variance)
            # 2. Stagnated for too many generations
            is_converged = np.std(fitness) < 1e-9
            is_stagnant = no_improv_count > 40
            
            if is_converged or is_stagnant:
                # --- DEEP POLISHING ---
                # Before abandoning this basin, do a thorough Local Search
                # to squeeze out the final precision.
                final_budget = 50 * dim
                final_sol, final_val = local_search(pop[0], fitness[0], final_budget)
                
                if final_val < best_val:
                    best_val = final_val
                    best_sol = final_sol.copy()
                
                # Break inner loop to trigger Restart
                break
            
            # --- SHADE Parameter Generation ---
            # Select random memory slot
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Generate CR ~ Normal(M_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(M_f, 0.1)
            f = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    val = m_f[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0:
                        if val > 1: val = 1.0
                        f[i] = val
                        break
            
            # --- Evolution Step ---
            new_pop = np.zeros_like(pop)
            new_fit = np.zeros_like(fitness)
            
            succ_cr = []
            succ_f = []
            succ_df = []
            
            # p-best selection param (Randomized for robustness)
            p_val = np.random.uniform(p_min, p_max)
            top_p = int(max(2, pop_size * p_val))
            
            for i in range(pop_size):
                if check_time(): return best_val
                
                # Mutation: current-to-pbest/1
                # Select x_pbest from top p%
                p_idx = np.random.randint(0, top_p)
                x_pbest = pop[p_idx]
                
                # Select r1 != i
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                
                # Select r2 != i, r1 from Union(Population, Archive)
                combined_len = pop_size + len(archive)
                r2 = np.random.randint(0, combined_len)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, combined_len)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                # Generate Mutant
                mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (pop[r1] - x_r2)
                
                # Crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary Handling: Reflection
                trial = reflect(trial)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
                    
                    # Store success data
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    succ_df.append(fitness[i] - f_trial)
                    
                    # Add replaced parent to archive
                    archive.append(pop[i].copy())
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = fitness[i]
            
            # Update Population
            pop = new_pop
            fitness = new_fit
            
            # Maintain Archive Size
            arc_limit = pop_size * 2
            while len(archive) > arc_limit:
                # Remove random element
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update SHADE Memory (Weighted Lehmer Mean)
            if len(succ_df) > 0:
                weights = np.array(succ_df) / np.sum(succ_df)
                s_cr = np.array(succ_cr)
                s_f = np.array(succ_f)
                
                mean_cr = np.sum(weights * s_cr)
                mean_f = np.sum(weights * s_f**2) / np.sum(weights * s_f)
                
                M_cr[k_mem] = mean_cr
                M_f[k_mem] = mean_f
                k_mem = (k_mem + 1) % H

    return best_val
