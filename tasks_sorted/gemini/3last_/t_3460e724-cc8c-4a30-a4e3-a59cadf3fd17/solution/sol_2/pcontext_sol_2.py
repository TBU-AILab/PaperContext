#The following algorithm implements **Restarting Self-Adaptive Differential Evolution (jDE)**. 
#
#This approach improves upon standard Differential Evolution and CMA-ES in the following ways:
#1.  **Self-Adaptation (jDE)**: Instead of fixed mutation (F) and crossover (CR) parameters which may be suboptimal for specific functions, each individual encodes its own F and CR values. These values evolve along with the solution, allowing the algorithm to dynamically tune itself to the landscape.
#2.  **Restart Mechanism**: To address the issue of premature convergence (stuck in local optima, likely the cause of the score 35 vs 203), the algorithm monitors population diversity. If the population converges to a single valley, it triggers a restart, re-initializing the population while preserving the best solution found so far (elitism). This allows global exploration within the time limit.
#3.  **Time-Awareness**: The algorithm strictly checks the time limit before every expensive function evaluation, ensuring it returns the best result immediately upon timeout without overrunning.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Restarting Self-Adaptive Differential Evolution (jDE).
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adaptive to dimension.
    # We clip it to ensure enough generations run within limited time 
    # while maintaining sufficient diversity.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # jDE Control Parameter Adaptation Probabilities
    tau1 = 0.1  # Probability to update F
    tau2 = 0.1  # Probability to update CR
    
    # Convergence criteria for restart
    restart_tol = 1e-6
    min_generations_before_restart = 20
    
    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Main Restart Loop ---
    while True:
        # Check if we have enough time to meaningfully start a new run
        if (time.time() - start_time) > max_time - 0.05:
            return global_best_val

        # Initialize Population
        # Shape: (pop_size, dim)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution into the new population
        start_eval_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_eval_idx = 1  # Skip re-evaluating the best
        
        # Initialize Control Parameters (F and CR) per individual
        # F initialized to 0.5, CR to 0.9
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
        
        # --- Evolution Loop ---
        gen = 0
        while True:
            # Time check at start of generation
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            gen += 1
            
            # 1. Parameter Adaptation (jDE)
            # Create trial parameter arrays
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            # Update F: with prob tau1, new F = 0.1 + 0.9 * rand
            mask_f = np.random.rand(pop_size) < tau1
            F_trial[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
            
            # Update CR: with prob tau2, new CR = rand
            mask_cr = np.random.rand(pop_size) < tau2
            CR_trial[mask_cr] = np.random.rand(np.sum(mask_cr))
            
            # 2. Mutation (DE/rand/1)
            # Generate random indices
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            # Mutation vector: v = x_r1 + F * (x_r2 - x_r3)
            # Broadcasting F_trial to shape (pop_size, 1)
            v = pop[r1] + F_trial[:, None] * (pop[r2] - pop[r3])
            
            # 3. Crossover (Binomial)
            # Generate mask for crossover
            cross_rand = np.random.rand(pop_size, dim)
            # Ensure at least one dimension is inherited from mutant (j_rand)
            j_rand = np.random.randint(0, dim, pop_size)
            j_mask = np.zeros((pop_size, dim), dtype=bool)
            j_mask[np.arange(pop_size), j_rand] = True
            
            mask_cross = (cross_rand < CR_trial[:, None]) | j_mask
            u = np.where(mask_cross, v, pop)
            
            # 4. Bound Constraint Handling (Clip)
            u = np.clip(u, min_b, max_b)
            
            # 5. Selection
            # Track which individuals are replaced to update F and CR
            replaced_mask = np.zeros(pop_size, dtype=bool)
            
            for i in range(pop_size):
                # Strict time check per individual
                if (time.time() - start_time) >= max_time:
                    return global_best_val
                
                f_u = func(u[i])
                
                # Greedy selection
                if f_u <= fitness[i]:
                    fitness[i] = f_u
                    pop[i] = u[i]
                    replaced_mask[i] = True
                    
                    if f_u < global_best_val:
                        global_best_val = f_u
                        global_best_sol = u[i].copy()
            
            # Update F and CR only for successful offspring (Survival of parameters)
            F[replaced_mask] = F_trial[replaced_mask]
            CR[replaced_mask] = CR_trial[replaced_mask]
            
            # 6. Convergence Check for Restart
            # If population fitness range is very small, we are stuck in a local optimum.
            # Check periodically to save computation.
            if gen >= min_generations_before_restart and gen % 10 == 0:
                fit_range = np.max(fitness) - np.min(fitness)
                if fit_range < restart_tol:
                    # Break inner loop to restart population
                    break
    
    return global_best_val
