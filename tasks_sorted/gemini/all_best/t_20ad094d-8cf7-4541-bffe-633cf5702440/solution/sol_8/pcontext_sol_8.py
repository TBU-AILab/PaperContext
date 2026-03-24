#The following algorithm, **iL-SHADE-R (Improved L-SHADE with Latin Hypercube Sampling and Stagnation Restarts)**, builds upon the successful features of the `L-SHADE-Dyn` (Algorithm 3) while addressing its potential stagnation in local optima.
#
#**Key Improvements:**
#1.  **Latin Hypercube Sampling (LHS) Initialization**: Instead of random initialization, LHS is used to generate the initial population. This ensures a more stratified and uniform coverage of the search space, increasing the probability of starting in a promising basin of attraction.
#2.  **Stagnation Detection & Restart**: The algorithm monitors the population's fitness variance. If the population converges (stagnates) before the time limit, it triggers a **Restart Mechanism**:
#    *   **Local Polish**: A quick Gaussian Walk is performed on the best individual to exploit any remaining precision gains.
#    *   **Dispersal**: The rest of the population is re-initialized randomly. This effectively allows the algorithm to escape local optima and use the remaining time to explore new areas (Iterated Local Search behavior).
#3.  **L-SHADE Core with LPSR**: Retains the highly effective Linear Population Size Reduction (LPSR) and Success-History Adaptation (SHADE) from the previous best-performing algorithm.
#4.  **Optimized Hyperparameters**: Maintains the tuning that proved best in Algorithm 3 ($p=0.2 \to 0.05$, $Archive=2.5 \times Pop$).
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
    # Population Size: Start large for LHS coverage, reduce linearly
    pop_size_init = int(25 * dim)
    pop_size_init = max(30, min(200, pop_size_init))
    pop_size_min = 4

    # Archive Size: 2.5x population size
    arc_rate = 2.5
    
    # Memory Size for parameter adaptation
    h_mem_size = 5
    
    # Dynamic p-best parameters (0.2 -> 0.05)
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
        # Permute strata indices
        perm = np.random.permutation(pop_size)
        # Random offset within stratum
        r = np.random.rand(pop_size)
        # Map to bounds
        vals = (perm + r) / pop_size
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

        # Linear Population Size Reduction
        plan_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Reduction: Remove worst individuals
            new_size = plan_pop_size
            population = population[:new_size]
            fitness = fitness[:new_size]
            pop_size = new_size
            
            # Archive Resizing
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                rm_idx = np.random.randint(0, len(archive))
                archive.pop(rm_idx)

        # --- 2. Stagnation Detection & Restart ---
        # If the population variance is near zero, we are stuck.
        # Check range of fitness in current population.
        if pop_size >= 4:
            fit_range = fitness[-1] - fitness[0]
            if fit_range < 1e-8 * (abs(fitness[0]) + 1e-8):
                # A. Local Polish (Gaussian Walk on Best)
                # Try to squeeze the current basin before leaving
                for _ in range(5):
                    if check_time(): return best_val
                    # Small perturbation (0.1% of domain)
                    sigma = 0.001 * diff_b
                    gw_trial = best_sol + np.random.normal(0, 1, dim) * sigma
                    gw_trial = np.clip(gw_trial, min_b, max_b)
                    gw_val = func(gw_trial)
                    if gw_val < best_val:
                        best_val = gw_val
                        best_sol = gw_trial.copy()
                        # Update population elitism
                        fitness[0] = best_val
                        population[0] = best_sol
                
                # B. Dispersal (Restart)
                # Keep the best (index 0), re-initialize the rest
                n_reset = pop_size - 1
                if n_reset > 0:
                    population[1:] = min_b + np.random.rand(n_reset, dim) * diff_b
                    
                    # Evaluate new candidates
                    for k in range(1, pop_size):
                        if check_time(): return best_val
                        val = func(population[k])
                        fitness[k] = val
                        if val < best_val:
                            best_val = val
                            best_sol = population[k].copy()
                    
                    # Reset Parameter Memory (Old parameters might be basin-specific)
                    mem_cr.fill(0.5)
                    mem_f.fill(0.5)
                    
                    # Re-sort
                    sorted_idx = np.argsort(fitness)
                    population = population[sorted_idx]
                    fitness = fitness[sorted_idx]

        # --- 3. Dynamic Strategy Parameters ---
        # Linearly decrease p from p_max to p_min
        current_p = p_max - (p_max - p_min) * progress
        p_num = max(2, int(current_p * pop_size))

        # --- 4. Generation Loop (L-SHADE) ---
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
            p_idx = np.random.randint(0, p_num)
            x_pbest = population[p_idx]
            
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
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
            
            # C. Crossover
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, population[i])
            
            # D. Bound Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection
            val_trial = func(trial)
            
            if val_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = val_trial
                
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial.copy()
                
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
            archive.pop(idx)

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

        # Sort for next generation
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_val
