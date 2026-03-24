#To improve upon the previous result, we need an algorithm that handles the exploration-exploitation trade-off better than standard Differential Evolution.
#
#The algorithm below implements **SHADE (Success-History based Adaptive Differential Evolution)** with an **Optional External Archive**.
#
#**Key Improvements over the previous attempt:**
#1.  **Memory-Based Parameter Adaptation:** Instead of random jittering, it uses historical success data to adapt the mutation factor ($F$) and crossover rate ($CR$) specifically for the problem landscape.
#2.  **Current-to-pBest Mutation:** It directs the search toward the top $p\%$ of best solutions (greedy but stochastic), which converges significantly faster than random strategies.
#3.  **External Archive:** It stores previously successful solutions that were discarded. This preserves diversity and allows the mutation operator to "learn" from past trajectories, preventing stagnation in local optima.
#4.  **Soft Restarts:** If the population variance collapses (all individuals are identical), it triggers a restart while keeping the global best, ensuring the algorithm uses the full time allowance effectively.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using SHADE (Success-History based Adaptive DE) 
    with External Archive and Soft Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 18 * dim is a standard heuristic for SHADE, 
    # but we cap it for time efficiency on expensive functions.
    pop_size = min(max(30, 18 * dim), 100)
    
    # SHADE Memory Parameters
    memory_size = 5
    memory_sf = np.full(memory_size, 0.5) # Memory for Scaling Factor F
    memory_scr = np.full(memory_size, 0.5) # Memory for Crossover Rate CR
    memory_pos = 0
    
    # Archive parameters (stores displaced parent vectors to maintain diversity)
    archive = []
    max_archive_size = pop_size 
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global state
    global_best_val = float('inf')
    global_best_pos = None

    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper Functions ---
    def trim_archive():
        while len(archive) > max_archive_size:
            # Remove random elements if archive is too big
            idx_to_remove = np.random.randint(0, len(archive))
            archive.pop(idx_to_remove)

    def cauchy_dist(loc, scale, size):
        return loc + scale * np.random.standard_cauchy(size)

    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_timeout():
            return global_best_val if global_best_val != float('inf') else fitness[0]
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            global_best_pos = population[i].copy()

    # --- Main Loop ---
    while not check_timeout():
        
        # 1. Parameter Generation
        # Generate CR and F for each individual based on memory
        r_idx = np.random.randint(0, memory_size, size=pop_size)
        m_cr = memory_scr[r_idx]
        m_sf = memory_sf[r_idx]
        
        # CR ~ Normal(M_cr, 0.1), clamped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_sf, 0.1), clamped (0, 1]
        # If F <= 0, regenerate. If F > 1, clip to 1.
        f = cauchy_dist(m_sf, 0.1, pop_size)
        while np.any(f <= 0):
            mask = f <= 0
            f[mask] = cauchy_dist(m_sf[mask], 0.1, np.sum(mask))
        f = np.clip(f, 0, 1)
        
        # 2. Mutation: current-to-pbest/1
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        # p is random between 2/pop_size and 0.2 (top 20%)
        p_val = np.random.uniform(2/pop_size, 0.2)
        top_p_cnt = max(2, int(pop_size * p_val))
        top_p_indices = sorted_indices[:top_p_cnt]
        
        # Generate vectors
        # x_pbest
        pbest_indices = np.random.choice(top_p_indices, size=pop_size)
        x_pbest = population[pbest_indices]
        
        # x_r1 (random from population, distinct from current)
        r1_indices = np.random.randint(0, pop_size, size=pop_size)
        # Fix indices where r1 == current_i (simple reshuffle approximation)
        mask_conflict = (r1_indices == np.arange(pop_size))
        r1_indices[mask_conflict] = (r1_indices[mask_conflict] + 1) % pop_size
        x_r1 = population[r1_indices]
        
        # x_r2 (random from Union(Population, Archive))
        # This is crucial for SHADE's performance
        pop_archive = population.copy()
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_archive = np.vstack((population, arr_archive))
        
        r2_indices = np.random.randint(0, len(pop_archive), size=pop_size)
        # Avoid simple conflicts (r2 != r1 and r2 != current is ideal but costly to enforce perfectly vectorized)
        x_r2 = pop_archive[r2_indices]
        
        # Compute Mutant
        # Vectorized mutation: v = x + F(xp - x) + F(xr1 - xr2)
        diff_1 = x_pbest - population
        diff_2 = x_r1 - x_r2
        # Reshape F for broadcasting
        f_col = f[:, np.newaxis]
        mutant = population + f_col * diff_1 + f_col * diff_2
        
        # 3. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals <= cr[:, np.newaxis]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, size=pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # Boundary constraint (Clip)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 4. Selection & Memory Update
        new_fitness = np.zeros(pop_size)
        
        success_sf = []
        success_scr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if check_timeout():
                return global_best_val
            
            t_val = func(trial_pop[i])
            new_fitness[i] = t_val
            
            if t_val < global_best_val:
                global_best_val = t_val
                global_best_pos = trial_pop[i].copy()
            
            if t_val < fitness[i]:
                # Improvement found
                archive.append(population[i].copy())
                success_sf.append(f[i])
                success_scr.append(cr[i])
                diff_fitness.append(fitness[i] - t_val)
                
                # Update population immediately (greedy)
                population[i] = trial_pop[i]
                fitness[i] = t_val
        
        trim_archive()
        
        # 5. Update Memory
        if len(success_sf) > 0:
            success_sf = np.array(success_sf)
            success_scr = np.array(success_scr)
            diff_fitness = np.array(diff_fitness)
            
            # Weighted Lehmer Mean
            total_diff = np.sum(diff_fitness)
            if total_diff > 0:
                weights = diff_fitness / total_diff
                
                # Mean for CR
                if np.max(success_scr) > 0: # Avoid null
                    mean_scr = np.sum(weights * success_scr)
                    memory_scr[memory_pos] = mean_scr
                
                # Lehmer Mean for F
                sum_sq_sf = np.sum(weights * success_sf**2)
                sum_sf = np.sum(weights * success_sf)
                if sum_sf > 0:
                    mean_sf = sum_sq_sf / sum_sf
                    memory_sf[memory_pos] = mean_sf
                
                memory_pos = (memory_pos + 1) % memory_size

        # 6. Convergence Check / Soft Restart
        # If population diversity is extremely low, we are stuck.
        # Soft restart: Keep best, scatter the rest.
        std_dev = np.std(fitness)
        if std_dev < 1e-6 and not check_timeout():
            # Reset population around global best or completely random
            # Keep the best
            best_idx = np.argmin(fitness)
            population[:] = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = population[best_idx] # Keep best in slot 0
            fitness[:] = float('inf')
            # Evaluate only new ones (slot 0 is known)
            fitness[0] = global_best_val
            # Reset memory slightly to allow new exploration
            memory_sf.fill(0.5)
            memory_scr.fill(0.5)
            archive = [] # Clear archive to adapt to new phase

    return global_best_val
