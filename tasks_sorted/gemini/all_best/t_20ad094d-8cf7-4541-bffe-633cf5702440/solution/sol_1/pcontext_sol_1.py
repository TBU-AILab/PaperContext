#Here is the fully operational and self-contained Python code implementing an enhanced **Self-Adaptive Differential Evolution (jDE)** algorithm with a **Restart mechanism**. 
#
#This approach improves upon standard Differential Evolution by:
#1.  **Self-Adaptation**: Automatically tuning the mutation factor ($F$) and crossover rate ($CR$) for each individual during the search, removing the need for manual hyperparameter guessing.
#2.  **Current-to-Best Strategy**: Utilizing a mutation strategy that guides individuals towards the best solution found so far, significantly speeding up convergence.
#3.  **Restart Mechanism**: Detecting when the population has converged (stagnated) and restarting the search with fresh candidates (while preserving the best solution) to escape local optima and maximize the usage of the available time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    # Use a small buffer to ensure we return strictly before the max_time limit
    limit = timedelta(seconds=max_time) - timedelta(milliseconds=50)

    # ---------------------------------------------------------
    # Algorithm Hyperparameters
    # ---------------------------------------------------------
    # Population Size: Adaptive based on dimension, clamped for safety
    # 10*dim is a standard heuristic for DE.
    pop_size = int(10 * dim)
    pop_size = max(20, min(100, pop_size))
    
    # jDE Adaptation Probabilities (Prob to reset F and CR)
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global Best Tracker
    global_best_val = float('inf')
    global_best_sol = None

    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------
    def check_timeout():
        return (datetime.now() - start_time) >= limit

    def init_population(size):
        return min_b + np.random.rand(size, dim) * diff_b

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    population = init_population(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # Control Parameters (F and CR) for each individual
    # Initialized randomly to promote diversity
    F_arr = np.random.uniform(0.1, 1.0, pop_size)
    CR_arr = np.random.uniform(0.0, 1.0, pop_size)

    # Initial Evaluation
    for i in range(pop_size):
        if check_timeout():
            return global_best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_sol = population[i].copy()

    # ---------------------------------------------------------
    # Main Optimization Loop
    # ---------------------------------------------------------
    while True:
        # 1. Timeout Check
        if check_timeout():
            return global_best_val

        # 2. Restart Mechanism
        # If the population fitness spread is very small, we are likely stuck in a local optimum.
        # Restart the population to explore new areas, keeping only the global best (Elitism).
        fitness_spread = np.max(fitness) - np.min(fitness)
        if fitness_spread < 1e-6:
            # Generate new population
            population = init_population(pop_size)
            # Elitism: Inject the best found solution into index 0
            population[0] = global_best_sol
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = global_best_val
            
            # Reset control parameters
            F_arr = np.random.uniform(0.1, 1.0, pop_size)
            CR_arr = np.random.uniform(0.0, 1.0, pop_size)
            
            # Evaluate new individuals (skipping index 0)
            for i in range(1, pop_size):
                if check_timeout():
                    return global_best_val
                val = func(population[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_sol = population[i].copy()
            
            # Skip to next generation
            continue

        # 3. Evolution Cycle (jDE with current-to-best/1 strategy)
        # Identify current best in population for the mutation strategy
        current_best_idx = np.argmin(fitness)
        current_best_vec = population[current_best_idx]

        for i in range(pop_size):
            if check_timeout():
                return global_best_val

            # --- Parameter Adaptation (jDE) ---
            # With probability tau, reset F or CR to new random values
            F_i = F_arr[i]
            CR_i = CR_arr[i]
            
            if np.random.rand() < tau_F:
                F_i = 0.1 + np.random.rand() * 0.9  # F in [0.1, 1.0]
            if np.random.rand() < tau_CR:
                CR_i = np.random.rand()             # CR in [0.0, 1.0]

            # --- Mutation (DE/current-to-best/1) ---
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This strategy converges faster than DE/rand/1
            
            # Select random distinct indices r1, r2 != i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
            
            x_i = population[i]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Calculate mutation vector
            mutant = x_i + F_i * (current_best_vec - x_i) + F_i * (x_r1 - x_r2)

            # --- Crossover (Binomial) ---
            cross_points = np.random.rand(dim) < CR_i
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
            
            trial = np.where(cross_points, mutant, x_i)

            # --- Boundary Handling ---
            trial = np.clip(trial, min_b, max_b)

            # --- Selection ---
            f_trial = func(trial)

            if f_trial <= fitness[i]:
                # Trial is better or equal; replace individual
                population[i] = trial
                fitness[i] = f_trial
                
                # Update control parameters only on success (jDE logic)
                F_arr[i] = F_i
                CR_arr[i] = CR_i
                
                # Check Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial.copy()
                    
                    # Greedy update of current best vector for subsequent mutations
                    current_best_vec = global_best_sol
                    current_best_idx = i

    return global_best_val
