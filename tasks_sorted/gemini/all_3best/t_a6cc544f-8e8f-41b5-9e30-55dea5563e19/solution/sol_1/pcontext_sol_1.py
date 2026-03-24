#The following algorithm implements a **Vectorized Differential Evolution with Restarts and Dithering**.
#
#**Key Improvements over the previous attempt:**
#1.  **Vectorization**: Instead of generating candidates one by one in nested loops, this implementation uses NumPy matrix operations to generate the entire population's trial vectors at once. This significantly reduces interpreter overhead and speeds up the optimization process, allowing for more generations within the `max_time`.
#2.  **Restart Strategy**: Standard DE can converge prematurely to local optima. This algorithm monitors the population's diversity (standard deviation of fitness). If the population stagnates/converges, it triggers a **Restart**: the global best solution is preserved (elitism), while the rest of the population is re-initialized randomly. This allows the algorithm to escape local minima and explore new basins of attraction.
#3.  **Aggressive Strategy (DE/best/1/bin) with Dithering**: It uses the "best/1/bin" mutation strategy, which converges faster than "rand/1". To prevent this aggressive strategy from getting stuck too easily, "Dithering" is applied: the mutation factor $F$ is randomized per individual, adding necessary noise to the search path.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Vectorized Differential Evolution with Restarts.
    
    Strategy: DE/best/1/bin with per-vector dithered F.
    Restarts: Triggered when population variance drops below threshold.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------
    # Configuration
    # -------------------------------
    # Population size: Balance between exploration and computational cost.
    # We ensure enough coverage for higher dims but cap it to allow iterations.
    pop_size = max(20, min(dim * 15, 80))
    
    # Differential Evolution Hyperparameters
    CR = 0.8           # Crossover probability
    # F is dynamic (dithered) in the loop [0.5, 1.0]
    
    # Restart threshold (standard deviation of fitness)
    tol = 1e-6

    # -------------------------------
    # Initialization
    # -------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    global_best_val = float('inf')
    global_best_vec = None

    # Helper for time checking
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # -------------------------------
    # Initial Evaluation
    # -------------------------------
    for i in range(pop_size):
        if check_timeout():
            return global_best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = pop[i].copy()

    # -------------------------------
    # Main Optimization Loop
    # -------------------------------
    while not check_timeout():
        
        # --- Restart Mechanism ---
        # If population has converged (low variance), restart to find new basins.
        # We keep the best solution found so far (Elitism).
        if np.std(fitness) < tol:
            # Re-initialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Inject global best
            pop[0] = global_best_vec
            
            # Re-evaluate (skip index 0 as it's known)
            fitness[0] = global_best_val
            fitness[1:] = float('inf')
            
            for i in range(1, pop_size):
                if check_timeout(): return global_best_val
                val = func(pop[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = pop[i].copy()
            
            if check_timeout(): return global_best_val

        # --- Vectorized Mutation & Crossover ---
        
        # 1. Identify "best" in current population
        curr_best_idx = np.argmin(fitness)
        x_best = pop[curr_best_idx]
        
        # 2. Select random indices r1, r2 for difference vector
        # Note: allowing r1==r2 or r1==i is a slight deviation from canonical DE 
        # but allows for much faster vectorized generation in Python. 
        # The impact is negligible for decent pop_size.
        idxs_r1 = np.random.randint(0, pop_size, pop_size)
        idxs_r2 = np.random.randint(0, pop_size, pop_size)
        
        x_r1 = pop[idxs_r1]
        x_r2 = pop[idxs_r2]
        
        # 3. Dithering: Generate random F in [0.5, 1.0] for each individual
        # This prevents stagnation with a fixed step size.
        F_scale = 0.5 + 0.5 * np.random.rand(pop_size, 1)
        
        # 4. Compute Mutant Vectors (DE/best/1)
        # V = X_best + F * (X_r1 - X_r2)
        mutant = x_best + F_scale * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_matrix = np.random.rand(pop_size, dim)
        cross_points = rand_matrix < CR
        
        # Ensure at least one parameter comes from mutant (prevent clones)
        j_rand = np.random.randint(0, dim, size=pop_size)
        cross_points[np.arange(pop_size), j_rand] = True
        
        # Create Trial Population
        trials = np.where(cross_points, mutant, pop)
        
        # 6. Bound Constraints (Clip)
        trials = np.clip(trials, min_b, max_b)
        
        # --- Selection (Evaluation Loop) ---
        # We must loop for evaluation as func takes a single array.
        for i in range(pop_size):
            if check_timeout():
                return global_best_val
            
            # Evaluate trial
            f_trial = func(trials[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trials[i]
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trials[i].copy()

    return global_best_val
