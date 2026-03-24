#The following Python algorithm implements **SHADE (Success-History based Adaptive Differential Evolution)** enhanced with **Midpoint-Target Bound Handling**, a **Local Search Polishing** phase, and a **Restart Mechanism**.
#
#### Key Improvements
#1.  **SHADE Algorithm**: Uses historical memory to adapt mutation factor ($F$) and crossover rate ($CR$) based on successful individuals, learning the landscape's features.
#2.  **Local Search (Polishing)**: Before restarting the population due to stagnation, a coordinate-descent-based local search is applied to the best solution found so far. This exploits the current basin of attraction to squeeze out maximum precision, which is often missed by global search operators.
#3.  **Midpoint-Target Bound Handling**: Instead of simply clipping values to bounds (which piles points on the edge) or reflecting them, this method resets out-of-bound variables to the midpoint between the parent value and the bound. This preserves population diversity better near the boundaries.
#4.  **Restart Mechanism**: Detects convergence (low variance) and restarts the population while preserving the elite solution, ensuring continuous exploration within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using SHADE with Local Search Polishing and Restarts.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Set deadline with a small buffer to ensure safe return
    deadline = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Configuration ---
    # Population size: Adaptive to dimension but clamped for throughput
    # pop_size ~ 20*dim is standard for SHADE, but we cap it to 100 
    # to ensure enough generations run within the time limit.
    pop_size = int(np.clip(20 * dim, 40, 100))
    
    # SHADE Parameters
    H = 6                   # Size of historical memory
    mem_cr = np.full(H, 0.5)# Memory for Crossover Rate
    mem_f = np.full(H, 0.5) # Memory for Scaling Factor
    k_mem = 0               # Memory index pointer
    p_best_rate = 0.11      # Top 11% for p-best selection
    arc_rate = 1.0          # Archive size relative to population
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # External Archive (stores replaced inferior solutions to maintain diversity)
    archive_size = int(pop_size * arc_rate)
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Global Best Tracking
    best_fit = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= deadline:
            return best_fit if best_sol is not None else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while datetime.now() < deadline:
        
        # 1. Stagnation Detection & Restart Logic
        # If population diversity (std dev) is negligible, we are likely stuck.
        if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
            
            # --- Phase A: Local Search Polishing ---
            # Before throwing away the population, try to refine the best solution
            # using a simple coordinate descent (MTS-LS1 style).
            ls_sol = best_sol.copy()
            ls_fit = best_fit
            
            # Initial search step size (0.5% of domain)
            step = diff_b * 0.005 
            
            # Perform a few passes of local search
            for _ in range(10): 
                if datetime.now() >= deadline: return best_fit
                improved = False
                
                for d in range(dim):
                    old_val = ls_sol[d]
                    
                    # Try negative step
                    ls_sol[d] = np.clip(old_val - step[d], min_b[d], max_b[d])
                    val = func(ls_sol)
                    if val < ls_fit:
                        ls_fit = val
                        improved = True
                    else:
                        # Try positive step
                        ls_sol[d] = np.clip(old_val + step[d], min_b[d], max_b[d])
                        val = func(ls_sol)
                        if val < ls_fit:
                            ls_fit = val
                            improved = True
                        else:
                            # Revert if no improvement
                            ls_sol[d] = old_val
                
                # Update global best
                if ls_fit < best_fit:
                    best_fit = ls_fit
                    best_sol = ls_sol.copy()
                
                # If no improvement in this pass, refine step size
                if not improved:
                    step *= 0.5
                    if np.all(step < 1e-9): # Stop if step is too small
                        break
            
            # --- Phase B: Restart ---
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Inject the elite (best found so far) to index 0
            pop[0] = best_sol.copy()
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fit
            
            # Reset SHADE Memory and Archive
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            n_archive = 0
            
            # Evaluate new population (skipping elite)
            for i in range(1, pop_size):
                if datetime.now() >= deadline: return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_sol = pop[i].copy()
            
            # Continue to next generation immediately
            continue

        # 2. SHADE Parameter Adaptation
        # Select memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(m_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(m_f, 0.1)
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        # Clamp F
        f = np.where(f > 1.0, 1.0, f)
        f = np.where(f <= 0.0, 0.1, f)
        
        # 3. Mutation Strategy: current-to-pbest/1
        # Sort population to find p-best
        sort_idx = np.argsort(fitness)
        pop_sorted = pop[sort_idx]
        
        # Select p-best individuals
        num_pbest = max(2, int(pop_size * p_best_rate))
        pbest_idxs = np.random.randint(0, num_pbest, pop_size)
        x_pbest = pop_sorted[pbest_idxs]
        
        # Select r1 (random from pop, distinct from i)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Ensure r1 != i
        conflict = (r1_idxs == np.arange(pop_size))
        r1_idxs[conflict] = (r1_idxs[conflict] + 1) % pop_size
        x_r1 = pop[r1_idxs]
        
        # Select r2 (random from Union of Pop and Archive)
        if n_archive > 0:
            union_pop = np.vstack((pop, archive[:n_archive]))
        else:
            union_pop = pop
            
        r2_idxs = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idxs]
        
        # Compute Mutant Vector
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        f_vec = f[:, np.newaxis]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = rand_vals < cr[:, np.newaxis]
        # Force at least one dimension to come from mutant
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints (Midpoint Target)
        # Instead of clipping, set value to midpoint between parent and bound.
        # This prevents accumulation at bounds and preserves diversity.
        lower_mask = trial < min_b
        upper_mask = trial > max_b
        
        trial = np.where(lower_mask, (min_b + pop) * 0.5, trial)
        trial = np.where(upper_mask, (max_b + pop) * 0.5, trial)
        
        # 6. Selection
        success_f = []
        success_cr = []
        diffs = []
        
        for i in range(pop_size):
            if datetime.now() >= deadline: return best_fit
            
            f_trial = func(trial[i])
            
            if f_trial < fitness[i]:
                # Improvement found
                diff = fitness[i] - f_trial
                
                # Update Archive
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Randomly replace an archive member
                    rep = np.random.randint(0, archive_size)
                    archive[rep] = pop[i].copy()
                
                # Update Population
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                # Store Success Info
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(diff)
                
                # Update Global Best
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_sol = trial[i].copy()
                    
        # 7. Update SHADE History Memory
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_diff = np.array(diffs)
            
            # Weights based on fitness improvement
            weights = s_diff / np.sum(s_diff)
            
            # Weighted Mean for CR
            mean_cr = np.sum(weights * s_cr)
            
            # Weighted Lehmer Mean for F
            num = np.sum(weights * (s_f ** 2))
            den = np.sum(weights * s_f)
            mean_f = num / den if den > 0 else 0.5
            
            mem_cr[k_mem] = mean_cr
            mem_f[k_mem] = mean_f
            k_mem = (k_mem + 1) % H
            
    return best_fit
