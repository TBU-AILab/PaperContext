#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 2.6589557565111335
#2. output value is: 6.787810833108026
#3. output value is: 37.71041518360055
#4. output value is: 63.823511875384256
#
#The best-performing algorithm so far is the **Time-Adaptive L-SHADE** (Algorithm 1). To improve this further, I propose incorporating techniques from **jSO (iL-SHADE)**, a highly successful refinement of the L-SHADE algorithm in competition settings.
#
#**Key Improvements:**
#1.  **Midpoint Target Bound Handling**: The previous algorithm used simple clipping (`np.clip`), which forces invalid solutions to the boundaries. This often causes the population to stagnate on the edges. The new algorithm uses a "Midpoint Target" approach: if a solution violates a bound, the new value is set to the midpoint between the parent and the bound (`(parent + bound) / 2`). This keeps the solution feasible while preserving the evolutionary search direction.
#2.  **Dynamic $p$ Strategy**: Instead of a fixed top-percentage ($p=0.11$) for mutation guidance, we linearly reduce $p$ from an exploratory $0.25$ to an exploitative $0.05$ based on the remaining time. This allows broad searching early on and faster convergence towards the end.
#3.  **Weighted Parameter Memory**: The parameter adaptation uses weighted Lehmer means based on fitness improvements, ensuring that parameters generating larger fitness gains have a stronger influence on the historical memory.
#4.  **Optimized Hyperparameters**: Adjusted population resizing ($N_{init} \approx 20D$) and archive size ($2.0 \times PopSize$) to better suit the limited time constraint.
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
    # Hyperparameters (jSO / iL-SHADE Inspired)
    # -------------------------------------------------------------------------
    # Initial Population: Start with a reasonably large size for exploration
    # Common heuristic: 20 * dim, clamped between 30 and 200 for efficiency
    pop_size_init = int(20 * dim)
    pop_size_init = max(30, min(200, pop_size_init))
    
    # Minimum Population: Threshold to prevent total collapse of diversity
    pop_size_min = 4

    # Memory Size for SHADE parameter adaptation
    h_mem_size = 6
    
    # Archive Size Factor (relative to current population)
    # Larger archive allows referencing more diverse historical vectors
    arc_rate = 2.0

    # -------------------------------------------------------------------------
    # Setup Data Structures
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Memory
    # Start CR high (0.8) to encourage component mixing, F neutral (0.5)
    mem_cr = np.full(h_mem_size, 0.8)
    mem_f = np.full(h_mem_size, 0.5)
    k_mem = 0

    # Initialize Population (Uniform Random)
    pop_size = pop_size_init
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))

    # Global Best Tracking
    best_val = float('inf')
    best_sol = None

    # External Archive
    archive = []

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

    # Sort population by fitness (required for current-to-pbest strategy)
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
        # Linearly reduce population size from init to min based on time progress
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Reduction: Remove worst individuals (end of the sorted array)
            pop_size = plan_pop_size
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            
            # Archive resizing: Maintain arc_rate * current_pop_size
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                # Efficient random removal: swap with last and pop
                rm_idx = np.random.randint(0, len(archive))
                archive[rm_idx] = archive[-1]
                archive.pop()

        # --- 3. Dynamic Strategy Parameters ---
        # Linearly decrease p (for p-best selection) from 0.25 (exploration) to 0.05 (exploitation)
        p_val = 0.25 - (0.20 * progress)
        p_best_num = max(2, int(p_val * pop_size))

        # --- 4. Generation Loop ---
        new_pop = np.empty_like(population)
        new_fit = np.empty_like(fitness)
        
        # Lists to store successful parameter configurations
        s_cr = []
        s_f = []
        s_imp = []

        for i in range(pop_size):
            if check_time(): return best_val

            # A. Parameter Generation based on Memory
            r_idx = np.random.randint(0, h_mem_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]

            # CR ~ Normal(mu_cr, 0.1)
            if mu_cr == -1:
                cr = 0.0
            else:
                cr = np.random.normal(mu_cr, 0.1)
                cr = np.clip(cr, 0.0, 1.0)

            # F ~ Cauchy(mu_f, 0.1)
            while True:
                f_val = mu_f + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    if f_val > 1: f_val = 1.0
                    break
            
            # B. Mutation: current-to-pbest-w/1
            # x_pbest: Randomly selected from top p_best_num individuals
            p_idx = np.random.randint(0, p_best_num)
            x_pbest = population[p_idx]

            # x_r1: Random from population, distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]

            # x_r2: Random from Union(Population, Archive), distinct from i and r1
            n_arch = len(archive)
            total_pool = pop_size + n_arch
            
            while True:
                r2 = np.random.randint(0, total_pool)
                # Check distinctness
                if r2 < pop_size:
                    if r2 != i and r2 != r1:
                        x_r2 = population[r2]
                        break
                else:
                    # Archive members are distinct from current pop by definition
                    x_r2 = archive[r2 - pop_size]
                    break
            
            mutant = population[i] + f_val * (x_pbest - population[i]) + f_val * (x_r1 - x_r2)

            # C. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True # Ensure at least one dimension comes from mutant
            
            trial = np.where(cross_mask, mutant, population[i])

            # D. Bound Handling (Midpoint Target)
            # Instead of clipping (which stacks points on bounds), set to midpoint 
            # between parent and bound. This preserves search direction and diversity.
            lower_violation = trial < min_b
            upper_violation = trial > max_b
            
            if np.any(lower_violation):
                trial[lower_violation] = (population[i][lower_violation] + min_b[lower_violation]) / 2.0
            if np.any(upper_violation):
                trial[upper_violation] = (population[i][upper_violation] + max_b[upper_violation]) / 2.0

            # E. Selection
            trial_val = func(trial)

            if trial_val <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = trial_val
                
                # Update Global Best
                if trial_val < best_val:
                    best_val = trial_val
                    best_sol = trial.copy()

                # Record success if strictly better
                if trial_val < fitness[i]:
                    diff = fitness[i] - trial_val
                    s_cr.append(cr)
                    s_f.append(f_val)
                    s_imp.append(diff)
                    
                    # Add replaced parent to archive
                    archive.append(population[i].copy())
            else:
                new_pop[i] = population[i]
                new_fit[i] = fitness[i]

        # --- 5. Post-Generation Updates ---
        population = new_pop
        fitness = new_fit

        # Trim Archive to maintain size limit
        max_arc_size = int(pop_size * arc_rate)
        while len(archive) > max_arc_size:
            rm_idx = np.random.randint(0, len(archive))
            archive[rm_idx] = archive[-1] # Swap with last
            archive.pop()                 # Remove last (O(1))

        # Update Memory (Weighted Lehmer Mean)
        if len(s_cr) > 0:
            s_cr_np = np.array(s_cr)
            s_f_np = np.array(s_f)
            s_imp_np = np.array(s_imp)
            
            # Weights proportional to fitness improvement
            total_imp = np.sum(s_imp_np)
            if total_imp > 0:
                weights = s_imp_np / total_imp
                
                # Update Memory F (Weighted Lehmer Mean)
                sum_wf = np.sum(weights * s_f_np)
                if sum_wf > 0:
                    mean_f = np.sum(weights * (s_f_np ** 2)) / sum_wf
                    mem_f[k_mem] = mean_f

                # Update Memory CR (Weighted Mean)
                if np.max(mem_cr[k_mem]) != -1:
                    mean_cr = np.sum(weights * s_cr_np)
                    mem_cr[k_mem] = mean_cr
                
                # Move memory pointer
                k_mem = (k_mem + 1) % h_mem_size

        # Sort population for next generation (required for p-best)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
    return best_val
