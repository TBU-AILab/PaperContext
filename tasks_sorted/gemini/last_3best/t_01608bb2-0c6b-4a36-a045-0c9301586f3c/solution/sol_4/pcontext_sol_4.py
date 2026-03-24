#The following algorithm implements a **Vectorized Self-Adaptive Differential Evolution (jDE)** with **Linear Population Size Reduction (LPSR)** and a **Soft Restart** mechanism.
#
#**Key Improvements:**
#1.  **Mutation Strategy**: Uses `current-to-pbest/1` instead of `rand/1`. This strategy (popularized by SHADE/JADE) guides the search towards the best individuals found so far, offering significantly faster convergence than random mutation.
#2.  **Vectorized Implementation**: The evolutionary operators (mutation, crossover, boundary handling) are fully vectorized using NumPy. This eliminates Python loop overheads, allowing for more generations within the `max_time`.
#3.  **Linear Population Size Reduction (LPSR)**: The population size decays linearly from a high initial value (for exploration) to a minimum (for exploitation). This aligns the search behavior with the remaining time budget.
#4.  **Restart Mechanism**: If the population converges (variance drops to zero) before time runs out, the algorithm restarts with a fresh population but preserves the global best solution. This allows it to escape local optima and explore other basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized jDE with current-to-pbest/1 mutation, 
    LPSR, and Restart mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population: Scaled with dimension. 
    # 20*dim gives good initial coverage. Capped at 300 for efficiency.
    pop_size_init = min(300, max(30, 20 * dim))
    pop_size_min = 4  # Minimum size to support mutation operators
    
    # jDE Self-Adaptation Parameters
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Helper: Bounce-Back Boundary Handling ---
    def handle_bounds(u):
        # Reflect lower bound violations: u = min + (min - u)
        viol_l = u < min_b
        if np.any(viol_l):
            u[viol_l] = 2 * min_b[viol_l] - u[viol_l]
            # Clip if reflection is still out of bounds
            u[u < min_b] = min_b[np.where(u < min_b)[1]]
            
        # Reflect upper bound violations: u = max - (u - max)
        viol_u = u > max_b
        if np.any(viol_u):
            u[viol_u] = 2 * max_b[viol_u] - u[viol_u]
            # Clip if reflection is still out of bounds
            u[u > max_b] = max_b[np.where(u > max_b)[1]]
        return u

    # --- Main Optimization Loop (Restart) ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Initialization
        pop_size = pop_size_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best from previous restart
        if global_best_sol is not None:
            population[0] = global_best_sol
            
        # Initialize jDE control parameters (F=0.5, CR=0.9 is a robust start)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        fitness = np.full(pop_size, float('inf'))
        
        # Initial Evaluation
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = population[i].copy()
                
        # 2. Evolutionary Cycle
        while True:
            # Check Time Constraint
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            if elapsed_seconds >= max_time:
                return global_best_val
            
            # --- Linear Population Size Reduction (LPSR) ---
            progress = elapsed_seconds / max_time
            target_pop = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            # Sort population by fitness (needed for LPSR and p-best selection)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            F = F[sorted_indices]
            CR = CR[sorted_indices]
            
            # Reduction Step
            if pop_size > target_pop:
                pop_size = target_pop
                population = population[:pop_size]
                fitness = fitness[:pop_size]
                F = F[:pop_size]
                CR = CR[:pop_size]
                
            # --- Convergence Check ---
            # If population has collapsed to a single point, restart
            if fitness[-1] - fitness[0] < 1e-8:
                break
                
            # --- Parameter Adaptation (jDE) ---
            # Generate masks for updates
            rand_F = np.random.rand(pop_size)
            rand_CR = np.random.rand(pop_size)
            
            mask_F = rand_F < tau_F
            mask_CR = rand_CR < tau_CR
            
            # Trial parameters
            trial_F = F.copy()
            trial_CR = CR.copy()
            
            # Update F: 0.1 + 0.9 * rand
            if np.any(mask_F):
                trial_F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
                
            # Update CR: rand
            if np.any(mask_CR):
                trial_CR[mask_CR] = np.random.rand(np.sum(mask_CR))
                
            # --- Mutation: current-to-pbest/1 ---
            # Select p-best (top 10% roughly)
            p_val = 0.1
            top_p_count = max(2, int(pop_size * p_val))
            
            # Index vectors
            idxs = np.arange(pop_size)
            
            # r_pbest: selected from top p% individuals
            idx_pbest = np.random.randint(0, top_p_count, pop_size)
            
            # r1, r2: selected from entire population
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            idx_r2 = np.random.randint(0, pop_size, pop_size)
            
            # Simple collision handling (re-roll once) to ensure distinctness
            # r1 != i
            collision_r1 = idx_r1 == idxs
            if np.any(collision_r1):
                idx_r1[collision_r1] = np.random.randint(0, pop_size, np.sum(collision_r1))
                
            # r2 != i and r2 != r1
            collision_r2 = (idx_r2 == idxs) | (idx_r2 == idx_r1)
            if np.any(collision_r2):
                idx_r2[collision_r2] = np.random.randint(0, pop_size, np.sum(collision_r2))
            
            # Vector calculation
            # Since population is sorted, population[idx_pbest] picks from best
            x = population
            xp = population[idx_pbest]
            xr1 = population[idx_r1]
            xr2 = population[idx_r2]
            
            # v = x + F * (xp - x) + F * (xr1 - xr2)
            F_col = trial_F[:, None]
            v = x + F_col * (xp - x) + F_col * (xr1 - xr2)
            
            # --- Crossover: Binomial ---
            # mask: True where we take from mutant v
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < trial_CR[:, None]
            cross_mask[idxs, j_rand] = True # Ensure at least one dimension is mutant
            
            u = np.where(cross_mask, v, x)
            
            # --- Boundary Handling ---
            u = handle_bounds(u)
            
            # --- Selection ---
            # Evaluate trials
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return global_best_val
                
                f_trial = func(u[i])
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    population[i] = u[i]
                    fitness[i] = f_trial
                    F[i] = trial_F[i]
                    CR[i] = trial_CR[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_sol = u[i].copy()

    return global_best_val
