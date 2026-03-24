#The following algorithm implements a **Self-Adaptive Differential Evolution (jDE) with Restart Strategy**.
#
#Key improvements over the standard Differential Evolution include:
#1.  **Self-Adaptation (jDE)**: Instead of fixed `F` and `CR` parameters, each individual evolves its own control parameters. This allows the algorithm to adapt to the specific landscape of the objective function dynamically.
#2.  **Restart Mechanism**: If the population converges (low variance) or stagnates (no improvement over time), the algorithm saves the best solution found so far and restarts the population. This helps escape local optima, which is crucial for achieving global minimization.
#3.  **Vectorized Operations**: Mutation and crossover steps are vectorized using NumPy to minimize Python loop overhead, reserving the available time budget for the objective function evaluations.
#4.  **Elitism**: When restarting, the global best solution is re-injected into the new population to ensure the search refines the best-known area while exploring new ones.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum output value using Self-Adaptive Differential Evolution (jDE)
    with a Restart Strategy to handle local optima and time constraints.
    """
    # --- Initialization & Timing ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Helper to enforce time limit strictness
    def check_time():
        return (datetime.now() - start_time) < time_limit

    # Pre-process bounds for efficient vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Configuration ---
    # Population size: A dynamic size capped at 50 ensures we can perform enough 
    # generations even if func() is moderately slow, while maintaining diversity.
    pop_size = int(np.clip(15 * dim, 10, 50))
    
    # jDE Self-Adaptation probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Track global best solution
    best_fitness = float('inf')
    best_sol = None
    
    # --- Main Loop (Outer Loop for Restarts) ---
    while check_time():
        # 1. Initialize Population
        # Random initialization within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject the best solution found in previous runs into the new population
        if best_sol is not None:
            pop[0] = best_sol
        
        # Initialize Control Parameters for jDE (F in [0.1, 1.0], CR in [0.0, 1.0])
        F = 0.1 + 0.9 * np.random.rand(pop_size)
        CR = np.random.rand(pop_size)
        
        # Evaluate Initial Population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if not check_time():
                return best_fitness
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_sol = pop[i].copy()
                
        # --- Evolution Loop (Inner Loop) ---
        stagnation_count = 0
        last_pop_best = np.min(fitness)
        
        while check_time():
            # 2. Parameter Adaptation (jDE)
            # Create masks for individuals that will update their parameters
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F_new = F.copy()
            CR_new = CR.copy()
            
            # Generate new parameter values where masks are True
            if np.any(mask_F):
                F_new[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            if np.any(mask_CR):
                CR_new[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # 3. Mutation (DE/rand/1/bin)
            # Vectorized selection of distinct indices: r1 != r2 != r3 != i
            # We use random shifts to efficiently avoid selecting index 'i'
            idxs = np.arange(pop_size)
            r1 = (idxs + np.random.randint(1, pop_size, pop_size)) % pop_size
            r2 = (idxs + np.random.randint(1, pop_size, pop_size)) % pop_size
            r3 = (idxs + np.random.randint(1, pop_size, pop_size)) % pop_size
            
            # Calculate Mutant Vectors: V = X_r1 + F * (X_r2 - X_r3)
            # Reshape F for broadcasting across dimensions
            F_broad = F_new[:, np.newaxis]
            mutant = pop[r1] + F_broad * (pop[r2] - pop[r3])
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover
            # Create trial vectors by mixing target and mutant vectors based on CR
            cross_rand = np.random.rand(pop_size, dim)
            CR_broad = CR_new[:, np.newaxis]
            cross_mask = cross_rand < CR_broad
            
            # Ensure at least one dimension comes from the mutant (j_rand)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # 5. Selection (Evaluation)
            for i in range(pop_size):
                if not check_time():
                    return best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy selection: if trial is better or equal, keep it
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    # Update control parameters only on success (jDE mechanism)
                    F[i] = F_new[i]
                    CR[i] = CR_new[i]
                    
                    # Update global best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_sol = trial_pop[i].copy()
            
            # 6. Convergence & Stagnation Check
            current_pop_best = np.min(fitness)
            pop_std = np.std(fitness)
            
            # Check if population best has improved
            if current_pop_best < last_pop_best:
                last_pop_best = current_pop_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Restart condition:
            # - Population has converged (low variance) OR
            # - No improvement for 'stagnation_count' generations
            if pop_std < 1e-6 or stagnation_count > 20:
                break # Break inner loop to trigger restart
                
    return best_fitness
