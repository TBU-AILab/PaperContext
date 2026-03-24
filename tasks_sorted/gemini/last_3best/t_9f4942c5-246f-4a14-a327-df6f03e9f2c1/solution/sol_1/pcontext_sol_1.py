#The following algorithm implements **Self-Adaptive Differential Evolution (jDE) with Population Restarts**.
#
#### Key Improvements over standard DE:
#1.  **Self-Adaptation (jDE):** Instead of fixed `mutation_factor` and `crossover_prob`, each individual carries its own parameter values. These values evolve along with the solution, allowing the algorithm to automatically adapt to the specific "landscape" of the function (e.g., separability, multimodality) without manual tuning.
#2.  **Restart Mechanism:** The algorithm monitors the population's fitness standard deviation. If the population converges (stagnates at a local optimum) and time permits, it triggers a "soft restart." It keeps the best solution found so far but re-initializes the rest of the population to explore new areas of the search space.
#3.  **Robustness:** This approach is significantly more robust against getting stuck in local optima (common in functions where standard DE might return ~20 instead of 0).
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Algorithm: Self-Adaptive Differential Evolution (jDE) with Restarts
    # This enhances standard DE by adapting control parameters F and CR during evolution
    # and restarting the population if convergence is detected to escape local optima.

    start_time = time.time()
    
    # --- Hyperparameters ---
    pop_size = 30  # Sufficient size for diverse exploration
    tau_F = 0.1    # Probability to update F
    tau_CR = 0.1   # Probability to update CR
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    diff_b = upper_b - lower_b
    
    # --- Initialization ---
    # Population: (pop_size, dim)
    population = lower_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)
    
    # Adaptive Parameters Initialization
    # Each individual has its own F (Mutation) and CR (Crossover)
    # F in [0.1, 1.0], CR in [0.0, 1.0]
    F = 0.1 + np.random.rand(pop_size) * 0.9
    CR = np.random.rand(pop_size)
    
    best_val = np.inf
    best_sol = None

    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()

    # --- Main Optimization Loop ---
    while True:
        # Iterate over population
        for i in range(pop_size):
            # Strict time check
            if (time.time() - start_time) >= max_time:
                return best_val
            
            # 1. Parameter Adaptation (jDE logic)
            # Create trial parameters F_i, CR_i based on current ones
            F_i = F[i]
            CR_i = CR[i]
            
            # With probability tau, generate new parameters
            if np.random.rand() < tau_F:
                F_i = 0.1 + np.random.rand() * 0.9
            if np.random.rand() < tau_CR:
                CR_i = np.random.rand()
            
            # 2. Mutation (DE/rand/1)
            # Select 3 distinct random indices r1, r2, r3 != i
            # Using simple rejection sampling for efficiency
            r1, r2, r3 = i, i, i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = np.random.randint(0, pop_size)
            
            # Form mutant vector
            mutant = population[r1] + F_i * (population[r2] - population[r3])
            
            # 3. Crossover (Binomial)
            cross_points = np.random.rand(dim) < CR_i
            # Ensure at least one dimension is taken from mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 4. Boundary Handling
            trial = np.clip(trial, lower_b, upper_b)
            
            # 5. Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Accept trial solution
                population[i] = trial
                fitness[i] = f_trial
                
                # Update individual's control parameters on success
                F[i] = F_i
                CR[i] = CR_i
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial.copy()
        
        # --- Restart Mechanism ---
        # If population diversity is low (converged), restart to search elsewhere.
        # This prevents getting stuck in local minima.
        if np.std(fitness) < 1e-6:
            # Keep the best solution found so far (elitism)
            population[0] = best_sol
            fitness[0] = best_val
            
            # Re-initialize the rest of the population
            population[1:] = lower_b + np.random.rand(pop_size - 1, dim) * diff_b
            
            # Reset adaptive parameters
            F = 0.1 + np.random.rand(pop_size) * 0.9
            CR = np.random.rand(pop_size)
            
            # Evaluate new individuals
            for k in range(1, pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_val
                
                val = func(population[k])
                fitness[k] = val
                
                if val < best_val:
                    best_val = val
                    best_sol = population[k].copy()

    return best_val
