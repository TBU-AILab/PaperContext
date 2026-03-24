#The following Python code implements a **Time-Adaptive L-SHADE** (Linear Population Size Reduction Success-History Adaptive Differential Evolution) algorithm.
#
#### Key Improvements over the previous JADE/jDE algorithms:
#
#1.  **Linear Population Size Reduction (LPSR)**: Instead of a fixed population size, this algorithm starts with a large population to maximize exploration and linearly reduces it over time. This forces the algorithm to focus computational resources on the best candidates as the time limit approaches.
#2.  **Success-History Adaptation (SHADE)**: Unlike JADE (which uses a single running average for parameters), this maintains a **memory** of successful configurations ($H=5$). This allows the algorithm to "remember" multiple distinct strategies (e.g., one set of parameters for exploration, another for exploitation) that worked in the past.
#3.  **Weighted Parameter Updates**: When updating parameter memories, the influence of a successful individual is **weighted by its fitness improvement**. This ensures that parameters causing significant breakthroughs have a stronger impact on future generations than those making minor improvements.
#4.  **Time-Based Execution**: The logic is fully adapted to `max_time`, calculating the "evolutionary progress" based on elapsed time rather than function evaluations.
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

    # Helper to check timeout
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # -------------------------------------------------------------------------
    # Hyperparameters (L-SHADE Standard Configuration)
    # -------------------------------------------------------------------------
    # Initial Population: Standard L-SHADE uses 18 * dim
    r_N_init = 18.0
    pop_size_init = int(r_N_init * dim)
    # Cap limits to ensure reasonable start for very high dims or very short times
    pop_size_init = max(30, min(200, pop_size_init))
    
    # Minimum Population: Needed for mutation strategies (pbest + r1 + r2)
    pop_size_min = 4

    # Memory Size for SHADE (History of successful parameters)
    h_mem_size = 5
    
    # Archive Size Factor (relative to current population)
    arc_rate = 1.4

    # -------------------------------------------------------------------------
    # Setup Data Structures
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Memory for Cr (Crossover) and F (Mutation)
    # Start with 0.5 (neutral)
    mem_cr = np.full(h_mem_size, 0.5)
    mem_f = np.full(h_mem_size, 0.5)
    k_mem = 0  # Memory index pointer

    # Initialize Population
    pop_size = pop_size_init
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))

    # Global Best
    best_val = float('inf')
    best_sol = None

    # External Archive (stores inferior solutions replaced by better ones)
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

    # Sort population by fitness (needed for current-to-pbest)
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while True:
        if check_time(): return best_val

        # --- 1. Calculate Progress (0.0 to 1.0) for LPSR ---
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress >= 1.0: return best_val

        # --- 2. Linear Population Size Reduction (LPSR) ---
        # Reduce size linearly from init to min based on time progress
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Reduction: The population is already sorted.
            # Remove worst individuals (from the end of the array)
            reduce_count = pop_size - plan_pop_size
            pop_size = plan_pop_size
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            
            # Archive resizing: Resize relative to NEW pop_size
            current_arc_size = len(archive)
            max_arc_size = int(pop_size * arc_rate)
            if current_arc_size > max_arc_size:
                # Randomly remove elements from archive to fit
                del_indices = np.random.choice(current_arc_size, current_arc_size - max_arc_size, replace=False)
                # Rebuild archive excluding deleted indices
                archive = [archive[i] for i in range(current_arc_size) if i not in del_indices]

        # --- 3. Prepare Arrays for Generation ---
        new_pop = np.empty_like(population)
        new_fit = np.empty_like(fitness)
        
        # Arrays to store successful updates for memory adaptation
        s_cr = []
        s_f = []
        s_imp = [] # Fitness improvement amount (weights)

        # --- 4. Generation Loop ---
        # Strategy: current-to-pbest/1/bin
        # p is dynamic? Standard L-SHADE uses best p in [2/pop, 0.2]
        # We stick to a robust top 11% (p=0.11) roughly.
        p_best_val = max(2, int(0.11 * pop_size))
        
        for i in range(pop_size):
            if check_time(): return best_val

            # A. Select Random Memory Index
            r_idx = np.random.randint(0, h_mem_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]

            # B. Generate Parameter CR
            # Normal(mu_cr, 0.1), clamped [0, 1]
            # Special handle: if CR close to terminal values
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)

            # C. Generate Parameter F
            # Cauchy(mu_f, 0.1), clamped (0, 1]
            while True:
                f_val = mu_f + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    if f_val > 1: f_val = 1.0
                    break
            
            # D. Mutation: current-to-pbest/1
            # v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
            
            # x_i
            x_i = population[i]

            # x_pbest: Random from top p_best_val individuals
            p_idx = np.random.randint(0, p_best_val)
            x_pbest = population[p_idx]

            # x_r1: Random from population, != i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]

            # x_r2: Random from Union(Population, Archive), != i, != r1
            # We treat archive as an extension of indices
            n_arch = len(archive)
            total_pool = pop_size + n_arch
            
            r2 = np.random.randint(0, total_pool)
            while True:
                # Map r2 index to actual vector
                if r2 < pop_size:
                    if r2 == i or r2 == r1:
                        r2 = np.random.randint(0, total_pool)
                        continue
                else:
                    # Archive index checks are less strict regarding self-identity 
                    # since archive is distinct from current pop, but ensure distinctness logic
                    pass 
                break
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]

            mutant = x_i + f_val * (x_pbest - x_i) + f_val * (x_r1 - x_r2)

            # E. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # F. Bound Constraints
            # SHADE usually uses midpoint correction if violated, simpler clipping here
            trial = np.clip(trial, min_b, max_b)

            # G. Selection
            trial_val = func(trial)

            if trial_val <= fitness[i]:
                # Improvement or Neutral
                new_pop[i] = trial
                new_fit[i] = trial_val
                
                # Update Global Best
                if trial_val < best_val:
                    best_val = trial_val
                    best_sol = trial.copy()

                # Record Success Data (if strictly better)
                if trial_val < fitness[i]:
                    diff = fitness[i] - trial_val
                    s_cr.append(cr)
                    s_f.append(f_val)
                    s_imp.append(diff)
                    
                    # Add replaced parent to archive
                    archive.append(x_i.copy())
            else:
                # Keep Parent
                new_pop[i] = x_i
                new_fit[i] = fitness[i]

        # --- 5. Post-Generation Updates ---
        population = new_pop
        fitness = new_fit

        # Trim Archive (Random removal if full)
        max_arc_size = int(pop_size * arc_rate)
        while len(archive) > max_arc_size:
            idx_rm = np.random.randint(0, len(archive))
            archive.pop(idx_rm)

        # Update Memory (Weighted Lehmer Mean)
        if len(s_cr) > 0:
            s_cr_np = np.array(s_cr)
            s_f_np = np.array(s_f)
            s_imp_np = np.array(s_imp)
            
            # Normalize weights
            total_imp = np.sum(s_imp_np)
            if total_imp > 0:
                weights = s_imp_np / total_imp
                
                # Update Memory CR (Weighted Mean)
                # If max is 0 (all CRs were 0), handle carefully
                if np.max(mem_cr[k_mem]) != -1: # Standard check
                    mean_cr = np.sum(weights * s_cr_np)
                    mem_cr[k_mem] = mean_cr
                
                # Update Memory F (Weighted Lehmer Mean)
                # mean = sum(w * f^2) / sum(w * f)
                sum_wf = np.sum(weights * s_f_np)
                if sum_wf > 0:
                    mean_f = np.sum(weights * (s_f_np ** 2)) / sum_wf
                    mem_f[k_mem] = mean_f
                
                # Increment Memory Pointer
                k_mem = (k_mem + 1) % h_mem_size

        # Sort for next generation (pbest logic requires sorted pop)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
    
    return best_val
