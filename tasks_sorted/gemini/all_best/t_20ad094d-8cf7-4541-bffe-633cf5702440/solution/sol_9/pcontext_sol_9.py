#The previous best-performing algorithm (Algorithm 6) achieved an output of **1.27** using **L-SHADE-Dyn** (L-SHADE with Dynamic Exploitation). To further improve performance and robustness, particularly against premature convergence, I propose **iL-SHADE-LHS-Restart** (Improved L-SHADE with Latin Hypercube Sampling and Adaptive Restarts).
#
#**Key Improvements:**
#1.  **Latin Hypercube Sampling (LHS) Initialization**: Replaces random initialization with LHS to ensure a stratified and uniform coverage of the initial search space. This maximizes the probability of finding promising basins of attraction early, which is crucial given the limited time.
#2.  **Stagnation Detection & Dispersal**: A check is added to detect if the population has converged (fitness variance $\approx$ 0) while time remains. If detected, a **Dispersal** mechanism is triggered: the global best solution is preserved, but the rest of the population is re-initialized randomly. This allows the algorithm to escape local optima and utilize the remaining computational budget effectively.
#3.  **Refined Dynamic $p$-best**: The exploration-to-exploitation parameter $p$ now scales from **0.25** down to **0.05** (previously 0.20 to 0.05). This slight increase in initial $p$ aligns with state-of-the-art jSO settings, encouraging broader exploration in the early phases.
#4.  **Optimized LHS Resolution**: The initial population size is set to $30 \times dim$ (capped at 200) to provide sufficient resolution for the Latin Hypercube grid.
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
    # Hyperparameters
    # -------------------------------------------------------------------------
    # Population Size
    # LHS requires good initial resolution. 
    # 30*dim provides dense sampling, capped between 30 and 200 for efficiency.
    pop_size_init = int(30 * dim)
    pop_size_init = max(30, min(200, pop_size_init))
    pop_size_min = 4

    # Archive Size: 2.3x population size (balances diversity and memory)
    arc_rate = 2.3
    
    # Memory Size for parameter adaptation (History length)
    h_mem_size = 6
    
    # Dynamic p-best parameters
    # Start broader (0.25) for exploration, narrow to 0.05 for exploitation
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

    best_val = float('inf')
    best_sol = None
    archive = []

    # -------------------------------------------------------------------------
    # Initialization: Latin Hypercube Sampling (LHS)
    # -------------------------------------------------------------------------
    # LHS ensures exactly one sample per interval in each dimension
    pop_size = pop_size_init
    population = np.zeros((pop_size, dim))
    
    for d in range(dim):
        # Create strata
        perm = np.random.permutation(pop_size)
        # Random offset within stratum
        r = np.random.rand(pop_size)
        # Map to [0, 1]
        vals = (perm + r) / pop_size
        # Map to bounds
        population[:, d] = min_b[d] + vals * diff_b[d]

    fitness = np.full(pop_size, float('inf'))

    # Initial Evaluation
    for i in range(pop_size):
        if check_time(): return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()

    # Sort population (required for p-best strategy)
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

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
        # Reduces population size linearly with time to focus computational budget
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Reduction: Remove worst individuals (end of sorted array)
            new_size = plan_pop_size
            population = population[:new_size]
            fitness = fitness[:new_size]
            pop_size = new_size
            
            # Archive Resizing
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                # Efficient swap-and-pop removal
                rm_idx = np.random.randint(0, len(archive))
                archive[rm_idx] = archive[-1]
                archive.pop()

        # --- 2. Stagnation Detection & Dispersal ---
        # If variance is effectively zero and we have significant time left, restart
        if pop_size >= 4:
            fit_range = fitness[-1] - fitness[0]
            # Threshold: 1e-9 check for convergence
            if fit_range < 1e-9 and progress < 0.9:
                # Dispersal: Keep best (index 0), re-init others randomly
                population[1:] = min_b + np.random.rand(pop_size - 1, dim) * diff_b
                
                # Reset evaluations for new individuals
                for k in range(1, pop_size):
                    if check_time(): return best_val
                    val = func(population[k])
                    fitness[k] = val
                    if val < best_val:
                        best_val = val
                        best_sol = population[k].copy()
                
                # Reset Parameter Memory to allow new learning curve
                mem_cr.fill(0.5)
                mem_f.fill(0.5)
                
                # Re-sort population
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx]
                fitness = fitness[sorted_idx]
                
                # Skip to next generation
                continue

        # --- 3. Dynamic Strategy Parameters ---
        # Linearly decrease p from p_max to p_min
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
            
            # B. Mutation: current-to-pbest/1
            # Select x_pbest from top p%
            p_idx = np.random.randint(0, p_num)
            x_pbest = population[p_idx]
            
            # Select x_r1 from Population (!= i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select x_r2 from Union(Population, Archive) (!= i, != r1)
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
            cross_mask[j_rand] = True # Ensure at least one dimension changes
            
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
                
                # Record Success
                if val_trial < fitness[i]:
                    diff = fitness[i] - val_trial
                    s_cr.append(cr)
                    s_f.append(f_val)
                    s_imp.append(diff)
                    archive.append(population[i].copy())
            else:
                new_pop[i] = population[i]
                new_fit[i] = fitness[i]

        # --- 5. Post-Generation Updates ---
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
            
            weights = s_imp_np / np.sum(s_imp_np)
            
            # Update CR
            if np.max(mem_cr) != -1:
                mean_cr = np.sum(weights * s_cr_np)
                mem_cr[k_mem] = mean_cr
            
            # Update F
            sum_wf = np.sum(weights * s_f_np)
            if sum_wf > 0:
                mean_f = np.sum(weights * (s_f_np ** 2)) / sum_wf
                mem_f[k_mem] = mean_f
            
            k_mem = (k_mem + 1) % h_mem_size

        # Sort for next generation (required for p-best selection)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_val
