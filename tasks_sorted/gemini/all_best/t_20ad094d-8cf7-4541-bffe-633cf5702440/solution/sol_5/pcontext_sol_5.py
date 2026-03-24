#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 8.602952425706874
#2. output value is: 2.6589557565111335
#3. output value is: 6.787810833108026
#4. output value is: 37.71041518360055
#5. output value is: 63.823511875384256
#
#The best-performing algorithm so far is the **Time-Adaptive L-SHADE** (Algorithm 2). To improve upon this, I propose an optimized variant: **L-SHADE with Dynamic Exploitation (iL-SHADE-Dyn)**.
#
#**Key Improvements:**
#1.  **Dynamic $p$-best Strategy**: Instead of a fixed top percentage (e.g., 11%) for mutation guidance, this algorithm linearly reduces $p$ from an exploratory $0.20$ to a highly exploitative $0.05$ as time progresses. This mimics the "focusing" behavior of global-to-local search.
#2.  **Optimized Archive Size**: Increased the archive rate from 1.4 to 2.5 times the population size. A larger archive maintains a richer history of diverse solutions, preventing the population from converging too quickly into a local optimum.
#3.  **Refined Population Resizing**: The initial population size is slightly increased (capped at 250) to ensure adequate sampling of the search space before the Linear Population Size Reduction (LPSR) mechanism forces convergence.
#4.  **Robust Bound Handling**: Retained the **Clipping** method (from Algorithm 2) as it proved superior to the Midpoint Target method (from Algorithm 1) for this specific problem class, ensuring solutions stay on potentially optimal boundaries.
#
#Here is the fully operational and self-contained Python code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # -------------------------------------------------------------------------
    # Initialization & Time Management
    # -------------------------------------------------------------------------
    start_time = datetime.now()
    # Safety buffer: stop slightly before max_time to ensure return
    time_limit = timedelta(seconds=max_time) - timedelta(milliseconds=100)

    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # -------------------------------------------------------------------------
    # Hyperparameters (Optimized L-SHADE)
    # -------------------------------------------------------------------------
    # Population Size
    # Start with enough individuals for exploration, but cap for efficiency.
    # 25*dim allows broad initial sampling, capped at 250 to ensure iteration speed.
    pop_size_init = int(25 * dim)
    pop_size_init = max(30, min(250, pop_size_init))
    pop_size_min = 4

    # Archive Size: 2.5x population size to maintain high diversity history
    arc_rate = 2.5 
    
    # Memory Size for parameter adaptation
    h_mem_size = 5
    
    # Dynamic p-best parameters
    # Linearly decrease from p_max (exploration) to p_min (exploitation)
    p_max = 0.20
    p_min = 0.05

    # -------------------------------------------------------------------------
    # Setup Data Structures
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Memory for adaptive parameters (initialized to 0.5)
    mem_cr = np.full(h_mem_size, 0.5)
    mem_f = np.full(h_mem_size, 0.5)
    k_mem = 0

    # Initialize Population
    pop_size = pop_size_init
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    archive = []
    
    # Global Best
    best_val = float('inf')
    best_sol = None

    # -------------------------------------------------------------------------
    # Initial Evaluation
    # -------------------------------------------------------------------------
    for i in range(pop_size):
        if check_time(): return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
    
    # Sort for p-best selection strategy
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while True:
        if check_time(): return best_val

        # --- 1. Calculate Progress (0.0 to 1.0) ---
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress >= 1.0: return best_val

        # --- 2. Linear Population Size Reduction (LPSR) ---
        # Linearly reduce population from init to min based on time progress
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Resize: Population is already sorted, remove worst (end of array)
            new_pop_size = plan_pop_size
            population = population[:new_pop_size]
            fitness = fitness[:new_pop_size]
            pop_size = new_pop_size
            
            # Resize Archive to match new population size
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                # Random removal to maintain diversity
                rm_idx = np.random.randint(0, len(archive))
                archive.pop(rm_idx)

        # --- 3. Dynamic Strategy Parameters ---
        # Linearly decrease p to focus search around best individuals towards the end
        current_p = p_max - (p_max - p_min) * progress
        p_num = max(2, int(current_p * pop_size))

        # --- 4. Generation Loop ---
        new_pop = np.empty_like(population)
        new_fit = np.empty_like(fitness)
        
        s_cr = []
        s_f = []
        s_imp = []

        for i in range(pop_size):
            if check_time(): return best_val

            # A. Parameter Generation
            r_idx = np.random.randint(0, h_mem_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]

            # Generate CR ~ Normal(mu_cr, 0.1)
            if mu_cr == -1:
                cr = 0.0
            else:
                cr = np.random.normal(mu_cr, 0.1)
                cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            while True:
                f_val = mu_f + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    if f_val > 1: f_val = 1.0
                    break
            
            # B. Mutation: current-to-pbest/1
            x_i = population[i]
            
            # Select x_pbest from top p%
            p_idx = np.random.randint(0, p_num)
            x_pbest = population[p_idx]
            
            # Select x_r1 from Population (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select x_r2 from Union(Population, Archive) (distinct from i and r1)
            n_arch = len(archive)
            total_pool = pop_size + n_arch
            
            r2 = np.random.randint(0, total_pool)
            while True:
                if r2 < pop_size:
                    if r2 == i or r2 == r1:
                        r2 = np.random.randint(0, total_pool)
                        continue
                break
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]

            # Mutation vector
            mutant = x_i + f_val * (x_pbest - x_i) + f_val * (x_r1 - x_r2)
            
            # C. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True # Ensure at least one dimension is changed
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # D. Bound Handling
            # Using Clipping as it performed best in previous iterations
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection
            val_trial = func(trial)
            
            if val_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = val_trial
                
                # Update Global Best
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial.copy()
                
                # Store success data for adaptation
                if val_trial < fitness[i]:
                    diff = fitness[i] - val_trial
                    s_cr.append(cr)
                    s_f.append(f_val)
                    s_imp.append(diff)
                    
                    # Add replaced parent to archive
                    archive.append(x_i.copy())
            else:
                new_pop[i] = x_i
                new_fit[i] = fitness[i]

        # --- 5. Post-Generation Updates ---
        population = new_pop
        fitness = new_fit

        # Archive Maintenance (Remove random elements if full)
        max_arc_size = int(pop_size * arc_rate)
        while len(archive) > max_arc_size:
            idx = np.random.randint(0, len(archive))
            archive.pop(idx)

        # Memory Update (Weighted Lehmer Mean)
        if len(s_cr) > 0:
            s_cr_np = np.array(s_cr)
            s_f_np = np.array(s_f)
            s_imp_np = np.array(s_imp)
            
            # Weights based on fitness improvement
            weights = s_imp_np / np.sum(s_imp_np)
            
            # Update Memory CR (Weighted Mean)
            if np.max(mem_cr) != -1:
                mean_cr = np.sum(weights * s_cr_np)
                mem_cr[k_mem] = mean_cr
            
            # Update Memory F (Weighted Lehmer Mean)
            sum_wf = np.sum(weights * s_f_np)
            if sum_wf > 0:
                mean_f = np.sum(weights * (s_f_np ** 2)) / sum_wf
                mem_f[k_mem] = mean_f
            
            k_mem = (k_mem + 1) % h_mem_size

        # Sort population for next generation (required for p-best)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_val
