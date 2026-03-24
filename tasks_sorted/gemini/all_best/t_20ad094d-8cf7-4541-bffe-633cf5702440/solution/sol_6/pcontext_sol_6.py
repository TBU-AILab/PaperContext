#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 8.602952425706874
#2. output value is: 2.6589557565111335
#3. output value is: 6.787810833108026
#4. output value is: 37.71041518360055
#5. output value is: 63.823511875384256
#6. output value is: 1.2731495637392527
#
#The best-performing algorithm so far is the **L-SHADE with Dynamic Exploitation** (Algorithm 6, output 1.27). To further improve performance, I propose **iL-SHADE-OBL (Improved L-SHADE with Opposition-Based Learning Initialization)**.
#
#**Key Improvements:**
#1.  **Opposition-Based Learning (OBL) Initialization**: Instead of starting with a purely random population, this algorithm generates a random population *and* its opposite counterparts ($x' = a + b - x$). It then selects the fittest $N$ individuals from this combined pool ($2N$). This provides a significantly better initial approximation of the global optimum and accelerates convergence, which is critical under time constraints.
#2.  **Optimized Hyperparameters**: 
#    - Increased `pop_size_init` to $30 \times dim$ (capped at 300) to allow OBL to filter a richer set of initial candidates.
#    - Adjusted `p_max` to $0.25$ (up from $0.20$) to encourage slightly broader exploration in the early phase, matching state-of-the-art configurations (jSO).
#    - Tuned Archive Rate to $2.3$ to balance diversity storage with memory management overhead.
#3.  **Robust Time-Bound Execution**: The initialization phase explicitly checks `max_time` during the $2N$ evaluations to ensure that even if the function is expensive, the algorithm returns the best result found so far without violating the time limit.
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
    # Hyperparameters (iL-SHADE-OBL)
    # -------------------------------------------------------------------------
    # Population Size
    # Increased scaling (30*dim) to maximize OBL effectiveness, capped for speed
    pop_size_init = int(30 * dim)
    pop_size_init = max(50, min(300, pop_size_init))
    pop_size_min = 5

    # Archive Size: 2.3x population size
    arc_rate = 2.3
    
    # Memory Size for parameter adaptation
    h_mem_size = 5
    
    # Dynamic p-best parameters
    # Linearly decrease from p_max (exploration) to p_min (exploitation)
    p_max = 0.25
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

    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # External Archive
    archive = []

    # -------------------------------------------------------------------------
    # Initialization (Opposition-Based Learning)
    # -------------------------------------------------------------------------
    # 1. Generate Random Population
    pop_rand = min_b + np.random.rand(pop_size_init, dim) * diff_b
    fit_rand = np.full(pop_size_init, float('inf'))

    # Evaluate Random
    for i in range(pop_size_init):
        if check_time(): return best_val
        val = func(pop_rand[i])
        fit_rand[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop_rand[i].copy()

    # 2. Generate Opposite Population (OBL)
    # x' = lower + upper - x
    pop_opp = min_b + max_b - pop_rand
    # Handle bounds (Clipping ensures valid opposite)
    pop_opp = np.clip(pop_opp, min_b, max_b)
    fit_opp = np.full(pop_size_init, float('inf'))

    # Evaluate Opposite
    for i in range(pop_size_init):
        if check_time(): return best_val
        val = func(pop_opp[i])
        fit_opp[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop_opp[i].copy()

    # 3. Select Best N Individuals
    total_pop = np.vstack((pop_rand, pop_opp))
    total_fit = np.concatenate((fit_rand, fit_opp))
    
    # Sort to keep the elite starting set
    sorted_idx = np.argsort(total_fit)
    pop_size = pop_size_init
    population = total_pop[sorted_idx[:pop_size]]
    fitness = total_fit[sorted_idx[:pop_size]]

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while True:
        if check_time(): return best_val

        # --- 1. Calculate Progress & LPSR ---
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress >= 1.0: return best_val

        # Linear Population Size Reduction (LPSR)
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Resize: Population is already sorted, remove worst (end of array)
            new_size = plan_pop_size
            population = population[:new_size]
            fitness = fitness[:new_size]
            pop_size = new_size
            
            # Archive Resize: Maintain ratio to current pop_size
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                # Efficient removal: Swap with last and pop (O(1))
                rm_idx = np.random.randint(0, len(archive))
                archive[rm_idx] = archive[-1]
                archive.pop()

        # --- 2. Dynamic Strategy Parameters ---
        # p linearly decreases to focus search
        current_p = p_max - (p_max - p_min) * progress
        p_num = max(2, int(current_p * pop_size))

        # --- 3. Generation Loop ---
        new_pop = np.empty_like(population)
        new_fit = np.empty_like(fitness)
        
        s_cr = []
        s_f = []
        s_imp = []

        for i in range(pop_size):
            if check_time(): return best_val

            # A. Parameter Generation from Memory
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
            # x_pbest: Random from top p%
            p_idx = np.random.randint(0, p_num)
            x_pbest = population[p_idx]
            
            # x_r1: Random from population, distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # x_r2: Random from Union(Population, Archive), distinct from i and r1
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

            mutant = population[i] + f_val * (x_pbest - population[i]) + f_val * (x_r1 - x_r2)
            
            # C. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True # Ensure at least one change
            
            trial = np.where(cross_mask, mutant, population[i])
            
            # D. Bound Handling (Clipping)
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
                
                # Store success info
                if val_trial < fitness[i]:
                    diff = fitness[i] - val_trial
                    s_cr.append(cr)
                    s_f.append(f_val)
                    s_imp.append(diff)
                    
                    # Add replaced parent to archive
                    archive.append(population[i].copy())
            else:
                new_pop[i] = population[i]
                new_fit[i] = fitness[i]

        # --- 4. Post-Generation Updates ---
        population = new_pop
        fitness = new_fit

        # Archive Maintenance
        max_arc_size = int(pop_size * arc_rate)
        while len(archive) > max_arc_size:
            idx = np.random.randint(0, len(archive))
            archive[idx] = archive[-1]
            archive.pop()

        # Memory Update (Weighted Lehmer Mean)
        if len(s_cr) > 0:
            s_cr_np = np.array(s_cr)
            s_f_np = np.array(s_f)
            s_imp_np = np.array(s_imp)
            
            # Weights based on fitness improvement
            weights = s_imp_np / np.sum(s_imp_np)
            
            # Update CR (Weighted Mean)
            if np.max(mem_cr) != -1:
                mean_cr = np.sum(weights * s_cr_np)
                mem_cr[k_mem] = mean_cr
            
            # Update F (Weighted Lehmer Mean)
            sum_wf = np.sum(weights * s_f_np)
            if sum_wf > 0:
                mean_f = np.sum(weights * (s_f_np ** 2)) / sum_wf
                mem_f[k_mem] = mean_f
            
            k_mem = (k_mem + 1) % h_mem_size

        # Sort population for next generation (required for p-best selection)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_val
