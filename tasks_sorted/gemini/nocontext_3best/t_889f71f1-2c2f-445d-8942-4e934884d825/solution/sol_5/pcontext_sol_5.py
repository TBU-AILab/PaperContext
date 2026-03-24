#The following algorithm implements **L-SHADE-LPSR with MTS-LS1 Polish**.
#
#**Key Improvements:**
#1.  **L-SHADE with Linear Population Size Reduction (LPSR)**: The population size is linearly reduced from a high initial value to a minimum (4) based on the progress of the optimization (evaluations vs estimated max evaluations). This balances global exploration in the early phase with fast convergence in the later phase.
#2.  **Weighted Lehmer Mean**: Adaptive parameters ($F$ and $CR$) are updated using a weighted mean based on the fitness improvement magnitude, prioritizing parameters that yield significant gains.
#3.  **MTS-LS1 Polish**: A robust local search method (Multiple Trajectory Search, Local Search 1) is applied in the final phase (or upon convergence). It performs a variable-step-size coordinate descent, which is far more efficient than Gaussian random walks for high-precision refinement.
#4.  **Robust Time Management**: The algorithm strictly estimates the remaining number of evaluations and adjusts the population reduction schedule dynamically. It ensures that the computationally expensive local search has dedicated time at the end.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Linear Population Size Reduction (LPSR)
    followed by a Multiple Trajectory Search (MTS-LS1) local polish.
    """
    # --- Timing & Initialization ---
    t_start = time.time()
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_sol = None
    
    # Helper: Check remaining time
    def get_remaining():
        return max_time - (time.time() - t_start)
        
    # Helper: Safe Evaluation with Global Best Update
    def evaluate(x):
        nonlocal best_fitness, best_sol
        # Ensure bounds
        x_c = np.clip(x, min_b, max_b)
        val = func(x_c)
        if val < best_fitness:
            best_fitness = val
            best_sol = x_c.copy()
        return val

    # --- Configuration ---
    # Initial population: 18 * dim (Balanced for speed/exploration), clipped [30, 200]
    # Small enough to run many generations, large enough to explore.
    p_init = int(np.clip(18 * dim, 30, 200)) 
    p_min = 4
    
    # Initialize Population
    pop_size = p_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if get_remaining() <= 0: return best_fitness
        fitness[i] = evaluate(pop[i])
        
    # Sort population
    idx = np.argsort(fitness)
    pop = pop[idx]
    fitness = fitness[idx]
    
    # --- L-SHADE Memory ---
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    archive = []
    
    # Stats
    eval_count = pop_size
    
    # --- Main Optimization Loop ---
    # Run until 90% of time is used, saving 10% (or at least 1s) for Polish
    while get_remaining() > 0:
        
        # 1. Time Management & Break Conditions
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        
        # Time budget for polish (min 0.5s, max 10% of total)
        polish_buffer = max(0.5, max_time * 0.1)
        if remaining < polish_buffer:
            break
            
        # Estimate total evaluations possible
        avg_eval_time = elapsed / eval_count if eval_count > 0 else 0
        safe_avg = avg_eval_time if avg_eval_time > 1e-9 else 1e-6
        max_evals_total = int(max_time / safe_avg)
        # Safety clamp
        if max_evals_total < eval_count + 100: max_evals_total = eval_count + 100
        
        # Progress (0.0 to 1.0)
        progress = min(1.0, eval_count / max_evals_total)
        
        # 2. Linear Population Size Reduction (LPSR)
        target_size = int(round((p_min - p_init) * progress + p_init))
        target_size = max(p_min, target_size)
        
        if pop_size > target_size:
            pop_size = target_size
            # Truncate population (already sorted)
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Resize Archive
            if len(archive) > pop_size:
                # Randomly remove elements to match population size
                keep_idx = np.random.choice(len(archive), pop_size, replace=False)
                archive = [archive[i] for i in keep_idx]
                
        # 3. Parameter Adaptation
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_f[r_idx]
        m_cr = mem_cr[r_idx]
        
        # F ~ Cauchy(m_f, 0.1), CR ~ Normal(m_cr, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.05 # Clamp near zero
        
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # 4. Mutation: current-to-pbest/1
        # Linearly reduce p from 0.2 to 0.05
        p_val = 0.2 * (1 - progress) + 0.05
        p_val = max(0.05, p_val)
        top_p = int(max(2, p_val * pop_size))
        
        # Select pbest
        pbest_idx = np.random.randint(0, top_p, pop_size)
        x_pbest = pop[pbest_idx]
        
        # Select r1 (distinct from i)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        mask_self = (r1_idx == np.arange(pop_size))
        r1_idx[mask_self] = (r1_idx[mask_self] + 1) % pop_size
        x_r1 = pop[r1_idx]
        
        # Select r2 (from Union of Pop and Archive)
        if len(archive) > 0:
            u_pop = np.vstack((pop, np.array(archive)))
        else:
            u_pop = pop
        r2_idx = np.random.randint(0, len(u_pop), pop_size)
        x_r2 = u_pop[r2_idx]
        
        # Mutation Vector
        F_ = f[:, None]
        mutant = pop + F_ * (x_pbest - pop) + F_ * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 6. Bound Correction (Midpoint logic)
        lower_mask = trial < min_b
        upper_mask = trial > max_b
        trial[lower_mask] = (pop[lower_mask] + min_b[lower_mask]) / 2
        trial[upper_mask] = (pop[upper_mask] + max_b[upper_mask]) / 2
        
        # 7. Selection & Memory Update
        success_diff = []
        success_f = []
        success_cr = []
        
        for i in range(pop_size):
            if get_remaining() <= 0: return best_fitness
            
            val = evaluate(trial[i])
            eval_count += 1
            
            if val < fitness[i]:
                diff = fitness[i] - val
                success_diff.append(diff)
                success_f.append(f[i])
                success_cr.append(cr[i])
                
                archive.append(pop[i].copy())
                fitness[i] = val
                pop[i] = trial[i]
        
        # Update Memory
        if success_diff:
            w = np.array(success_diff)
            w /= np.sum(w)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            # Weighted Lehmer Mean for F
            mean_f = np.sum(w * s_f**2) / np.sum(w * s_f)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # Re-sort population
        sort_i = np.argsort(fitness)
        pop = pop[sort_i]
        fitness = fitness[sort_i]
        
        # Convergence Check
        if np.std(fitness) < 1e-9:
            break

    # --- Final Polish: MTS-LS1 (Multiple Trajectory Search - Local Search 1) ---
    # Uses remaining time to refine the best solution via Coordinate Descent
    if best_sol is not None:
        current_sol = best_sol.copy()
        current_val = best_fitness
        
        # Initial search range: 40% of domain
        sr = diff_b * 0.4
        
        # Polish Loop
        while get_remaining() > 0:
            improved = False
            # Search dimensions in random order
            dims_order = np.random.permutation(dim)
            
            for d in dims_order:
                if get_remaining() <= 0: return best_fitness
                
                # 1. Try negative direction
                x_new = current_sol.copy()
                x_new[d] = np.clip(x_new[d] - sr[d], min_b[d], max_b[d])
                val = evaluate(x_new)
                
                if val < current_val:
                    current_val = val
                    current_sol = x_new
                    sr[d] *= 1.5 # Expand range on success
                    improved = True
                else:
                    # 2. Try positive direction (half step from original?)
                    # Standard MTS strategy: Check original - SR. If fail, check original + 0.5 * SR.
                    x_new[d] = np.clip(current_sol[d] + 0.5 * sr[d], min_b[d], max_b[d])
                    val = evaluate(x_new)
                    
                    if val < current_val:
                        current_val = val
                        current_sol = x_new
                        sr[d] *= 1.5
                        improved = True
                    else:
                        # 3. Fail: Shrink range
                        sr[d] *= 0.5
            
            # If all search ranges are negligible, we are converged
            if np.max(sr) < 1e-12:
                break
                
    return best_fitness
