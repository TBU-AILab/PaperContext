#The following algorithm implements a **Self-Adaptive Differential Evolution (jDE)** combined with a **Gaussian Polishing** phase. 
#
#**Improvements over the previous version:**
#1.  **Self-Adaptation (jDE):** Instead of using random or fixed mutation ($F$) and crossover ($CR$) parameters, this algorithm encodes $F$ and $CR$ into each individual. Successful parameters "survive" to the next generation, allowing the algorithm to "learn" the specific landscape of the function.
#2.  **Gaussian Polishing:** DE is a global searcher and can struggle with the final decimal places of precision. When the population converges (before a restart), this algorithm runs a concentrated local search (Gaussian perturbation) around the global best to refine the solution to high precision.
#3.  **Robust Restart:** It detects stagnation (low variance) and restarts the population while preserving the elite solution, ensuring continuous exploration within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with 'current-to-best' mutation and a Gaussian local search polish 
    upon convergence.
    """
    # --- 1. Initialization & Helpers ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def is_timeout():
        return (datetime.now() - start_time) >= time_limit

    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Hyperparameters
    # Population size: Sufficient for search, small enough for speed
    pop_size = max(20, 10 * dim) 
    
    # jDE Self-Adaptation probabilities
    tau_F = 0.1
    tau_CR = 0.1

    # Global best tracking
    global_best_val = float('inf')
    global_best_vec = None

    # --- 2. Main Optimization Loop (Restarts) ---
    while not is_timeout():
        
        # A. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Initialize F and CR for each individual (jDE strategy)
        # F ~ N(0.5, 0.3) clipped, CR ~ N(0.9, 0.1) clipped
        pop_F = np.clip(np.random.normal(0.5, 0.3, pop_size), 0.1, 1.0)
        pop_CR = np.clip(np.random.normal(0.9, 0.1, pop_size), 0.0, 1.0)
        
        fitness = np.full(pop_size, float('inf'))

        # Inject global best if it exists (Elitism across restarts)
        start_idx = 0
        if global_best_vec is not None:
            pop[0] = global_best_vec
            fitness[0] = global_best_val
            pop_F[0] = 0.5 # Reset params for the survivor to allow fresh adaptation
            pop_CR[0] = 0.9
            start_idx = 1

        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if is_timeout(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()

        # B. Evolution Epochs
        converged = False
        while not converged and not is_timeout():
            
            # Prepare arrays for vectorization
            # We will generate trial vectors for the whole population at once if possible,
            # but simple iterating is safer for time-checking inside the loop.
            
            best_idx = np.argmin(fitness)
            x_best = pop[best_idx]

            # Track improvements to judge convergence
            initial_best_val = fitness[best_idx]

            for i in range(pop_size):
                if is_timeout(): return global_best_val

                # --- jDE Parameter Update ---
                # With probability tau, regenerate F and CR
                if np.random.rand() < tau_F:
                    f_i = 0.1 + np.random.rand() * 0.9 # Uniform [0.1, 1.0]
                else:
                    f_i = pop_F[i]

                if np.random.rand() < tau_CR:
                    cr_i = np.random.rand() # Uniform [0.0, 1.0]
                else:
                    cr_i = pop_CR[i]

                # --- Mutation: DE/current-to-best/1 ---
                # V = Xi + F*(Xbest - Xi) + F*(Xr1 - Xr2)
                # Select r1, r2 distinct from i
                indices = [idx for idx in range(pop_size) if idx != i]
                r1, r2 = np.random.choice(indices, 2, replace=False)
                
                x_i = pop[i]
                x_r1 = pop[r1]
                x_r2 = pop[r2]

                mutant = x_i + f_i * (x_best - x_i) + f_i * (x_r1 - x_r2)

                # --- Crossover: Binomial ---
                rand_vals = np.random.rand(dim)
                j_rand = np.random.randint(dim)
                mask = (rand_vals <= cr_i)
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, x_i)
                
                # --- Bound Constraint (Clipping) ---
                trial = np.clip(trial, min_b, max_b)

                # --- Selection ---
                trial_val = func(trial)

                if trial_val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    # Update successful parameters
                    pop_F[i] = f_i
                    pop_CR[i] = cr_i

                    if trial_val < global_best_val:
                        global_best_val = trial_val
                        global_best_vec = trial.copy()

            # --- Convergence Check & Polishing ---
            # If population variance is very low, we are stuck in a basin.
            # Before restarting, we polish the best result.
            std_dev = np.std(fitness)
            
            # Dynamic threshold or hard limit
            if std_dev < 1e-6 * (abs(global_best_val) + 1.0):
                converged = True
                
                # --- Gaussian Polish (Local Search) ---
                # Refines the best solution found before discarding the population
                # Uses shrinking variance to hone in on the minimum.
                current_polish_vec = global_best_vec.copy()
                current_polish_val = global_best_val
                
                # Determine how much time is left for polishing
                # We allocate a small budget of iterations
                polish_iters = 100 
                sigma = 0.01 * np.max(diff_b) # Start with 1% of domain size

                for k in range(polish_iters):
                    if is_timeout(): return global_best_val
                    
                    # Generate a neighbor
                    perturbation = np.random.normal(0, sigma, dim)
                    neighbor = np.clip(current_polish_vec + perturbation, min_b, max_b)
                    
                    n_val = func(neighbor)
                    
                    if n_val < current_polish_val:
                        current_polish_val = n_val
                        current_polish_vec = neighbor
                        if n_val < global_best_val:
                            global_best_val = n_val
                            global_best_vec = neighbor.copy()
                    else:
                        # Decay sigma if no improvement (adaptive step size)
                        sigma *= 0.95

    return global_best_val
