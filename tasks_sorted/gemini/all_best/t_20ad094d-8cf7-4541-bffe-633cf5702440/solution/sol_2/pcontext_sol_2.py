#The best previous algorithm used **Self-Adaptive Differential Evolution (jDE)** with a **Restart Mechanism** and achieved an output of approximately **37.71**. To improve upon this, I propose an implementation of a simplified **JADE (Adaptive Differential Evolution with Optional External Archive)**.
#
#**Key Improvements:**
#1.  **Current-to-p-best Mutation**: Instead of guiding evolution towards the single best individual (which causes premature convergence), this strategy guides individuals towards the top $p\%$ of the population. This balances exploration and exploitation significantly better.
#2.  **External Archive**: The algorithm maintains an archive of recently replaced inferior solutions. These are used as difference vectors in mutation to maintain population diversity without additional function evaluations.
#3.  **Advanced Parameter Adaptation**: Instead of random resets (jDE), this algorithm learns the optimal Mutation Factor ($F$) and Crossover Rate ($CR$) using historical success data (Lehmer mean for $F$, arithmetic mean for $CR$).
#4.  **Stagnation Restart**: Retains the restart mechanism to escape local optima if the population variance collapses.
#
#Here is the fully operational and self-contained Python code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    # Safety buffer: 50ms to ensure we return strictly before max_time
    time_limit = timedelta(seconds=max_time) - timedelta(milliseconds=50)

    # -------------------------------------------------------------
    # Hyperparameters (JADE/SHADE-inspired)
    # -------------------------------------------------------------
    # Population size: adaptive to dimension.
    # A range of 20 to 100 is generally robust for constrained time.
    pop_size = int(20 * dim)
    pop_size = max(20, min(100, pop_size))
    
    # Archive size matches population size (stores historical vectors)
    archive_size = pop_size
    
    # Adaptation parameters
    p_best_rate = 0.10   # Top 10% used for mutation guidance
    c_adapt = 0.1        # Learning rate for parameter adaptation
    
    # Initial adaptation means (start neutral)
    mu_cr = 0.5
    mu_f = 0.5

    # -------------------------------------------------------------
    # Setup & Pre-processing
    # -------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    # Archive list
    archive = []

    # Helper for timeout check
    def is_timeout():
        return (datetime.now() - start_time) >= time_limit

    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))

    # Initial Evaluation
    for i in range(pop_size):
        if is_timeout(): return global_best_val
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            global_best_sol = population[i].copy()

    # -------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------
    while True:
        if is_timeout(): return global_best_val

        # 1. Sort Population
        # Necessary for 'current-to-pbest' strategy to identify top performers
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # 2. Restart Mechanism (Stagnation Check)
        # If population diversity (fitness spread) is negligible, restart.
        if fitness[-1] - fitness[0] < 1e-8:
            # Generate fresh population
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            # Elitism: Keep the global best at index 0
            population[0] = global_best_sol.copy()
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = global_best_val
            
            # Reset adaptation memory and archive
            archive = []
            mu_cr, mu_f = 0.5, 0.5
            
            # Evaluate new individuals (skipping best)
            for i in range(1, pop_size):
                if is_timeout(): return global_best_val
                val = func(population[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_sol = population[i].copy()
            
            # Skip to next generation (implicitly re-sorts)
            continue

        # 3. Evolution Cycle (JADE: DE/current-to-pbest/1/bin)
        next_population = np.empty_like(population)
        next_fitness = np.empty_like(fitness)
        
        successful_cr = []
        successful_f = []
        
        # Determine number of individuals in top p%
        p_num = max(1, int(p_best_rate * pop_size))
        
        for i in range(pop_size):
            if is_timeout(): return global_best_val
            
            # --- Parameter Generation ---
            # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
            cr_i = np.random.normal(mu_cr, 0.1)
            cr_i = np.clip(cr_i, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1), clipped to (0, 1]
            # Retry if <= 0, clamp at 1
            while True:
                f_i = mu_f + 0.1 * np.random.standard_cauchy()
                if f_i > 0:
                    if f_i > 1: f_i = 1.0
                    break
            
            # --- Mutation Strategy ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            # This guides the search towards good areas (pbest) while maintaining
            # diversity via the difference vector (r1 - r2).
            
            x_i = population[i]
            
            # Select x_pbest: random from top p% sorted individuals
            p_idx = np.random.randint(0, p_num)
            x_pbest = population[p_idx]
            
            # Select x_r1: random from population, distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select x_r2: random from (Population U Archive), distinct from i and r1
            n_arch = len(archive)
            total_pool = pop_size + n_arch
            
            # Try to find a valid r2
            r2 = np.random.randint(0, total_pool)
            # Simple loop to ensure distinctness (mostly for population indices)
            while (r2 < pop_size and (r2 == i or r2 == r1)):
                r2 = np.random.randint(0, total_pool)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
                
            # Compute Mutant Vector
            mutant = x_i + f_i * (x_pbest - x_i) + f_i * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(dim) < cr_i
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True 
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # --- Bound Handling ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            val_trial = func(trial)
            
            # Update Global Best greedily
            if val_trial < global_best_val:
                global_best_val = val_trial
                global_best_sol = trial.copy()
            
            if val_trial <= fitness[i]:
                # Trial is better/equal: replace parent
                next_population[i] = trial
                next_fitness[i] = val_trial
                
                # Save successful parameters
                successful_cr.append(cr_i)
                successful_f.append(f_i)
                
                # Add replaced parent to archive
                archive.append(x_i.copy())
            else:
                # Keep parent
                next_population[i] = x_i
                next_fitness[i] = fitness[i]
        
        # 4. Finalize Generation
        population = next_population
        fitness = next_fitness
        
        # Trim Archive to keep size <= pop_size
        while len(archive) > archive_size:
            rm_idx = np.random.randint(0, len(archive))
            archive.pop(rm_idx)
            
        # 5. Adapt Parameters (Lehmer Mean for F, Arithmetic for CR)
        if len(successful_cr) > 0:
            # Update mu_cr
            mean_cr = np.mean(successful_cr)
            mu_cr = (1 - c_adapt) * mu_cr + c_adapt * mean_cr
            
            # Update mu_f
            f_arr = np.array(successful_f)
            # Lehmer mean = sum(f^2) / sum(f) favors larger/more effective F
            if np.sum(f_arr) > 0:
                mean_lehmer = np.sum(f_arr**2) / np.sum(f_arr)
                mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer

    return global_best_val
