#Here is an improved algorithm. It implements **Self-Adaptive Differential Evolution (jDE) with a Restart Mechanism**. 
#
#**Why this is better:**
#1.  **Self-Adaptation (jDE):** Instead of using fixed parameters for Mutation ($F$) and Crossover ($CR$), this algorithm evolves them along with the solution. This allows it to automatically tune itself to the specific landscape of the function.
#2.  **Restart Mechanism:** If the population converges (stagnates) before the time limit, the algorithm triggers a restart. It preserves the best-so-far solution but re-initializes the rest of the population to explore new areas of the search space.
#3.  **Robustness:** It maintains strict time management and error handling for function evaluations.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function using Self-Adaptive Differential Evolution (jDE).
    Features:
    - Adaptive F and CR parameters (jDE logic).
    - Population Restart mechanism to escape local optima.
    - Strict time budget adherence.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # 1. Parameter Setup
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Population size: Balance between exploration (large) and speed (small).
    # Clamped between 20 and 100 to ensure responsiveness within limited time.
    pop_size = int(max(20, min(100, 15 * dim)))
    
    # jDE Hyperparameters
    tau_F = 0.1   # Probability to adjust F
    tau_CR = 0.1  # Probability to adjust CR

    # 2. Initialization
    population = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    # Adaptive parameters for each individual
    # F initialized to 0.5, CR to 0.9 (heuristics)
    F_arr = np.full(pop_size, 0.5)
    CR_arr = np.full(pop_size, 0.9)

    best_fitness = float('inf')
    best_individual = None
    
    pop_indices = list(range(pop_size))

    # Helper: Check strictly if time is up
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # Helper: Safe function evaluation
    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')

    # Helper: Initialize or Reset Population
    def init_population(preserve_best=False):
        nonlocal best_fitness, best_individual
        
        start_idx = 0
        if preserve_best and best_individual is not None:
            # Place global best at index 0
            population[0] = best_individual
            fitness[0] = best_fitness
            start_idx = 1
        
        for i in range(start_idx, pop_size):
            if is_time_up(): return False
            
            # Random position
            ind = min_b + np.random.rand(dim) * diff_b
            population[i] = ind
            
            val = safe_evaluate(ind)
            fitness[i] = val
            
            # Randomize adaptive params for new individuals
            F_arr[i] = 0.1 + 0.9 * random.random()
            CR_arr[i] = random.random()

            if val < best_fitness:
                best_fitness = val
                best_individual = ind.copy()
        
        return True

    # Initial Population Generation
    if not init_population():
        return best_fitness

    # 3. Main Optimization Loop
    while not is_time_up():
        
        # --- Restart Logic ---
        # If population diversity (std dev of fitness) is too low, we are stagnant.
        # We restart to search elsewhere, keeping the best solution found so far.
        if np.std(fitness) < 1e-6:
            if not init_population(preserve_best=True):
                return best_fitness
            continue

        # --- Evolution Cycle ---
        for i in range(pop_size):
            if is_time_up(): return best_fitness

            # A. jDE Parameter Adaptation
            # ---------------------------
            # Propose new F
            if random.random() < tau_F:
                F_new = 0.1 + 0.9 * random.random()
            else:
                F_new = F_arr[i]
            
            # Propose new CR
            if random.random() < tau_CR:
                CR_new = random.random()
            else:
                CR_new = CR_arr[i]

            # B. Mutation (DE/rand/1)
            # -----------------------
            # Pick 3 random distinct individuals other than i
            candidates = pop_indices[:i] + pop_indices[i+1:]
            a, b, c = random.sample(candidates, 3)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            mutant = x_a + F_new * (x_b - x_c)

            # C. Crossover (Binomial)
            # -----------------------
            cross_points = np.random.rand(dim) < CR_new
            # Ensure at least one dimension is changed
            forced_idx = np.random.randint(dim)
            cross_points[forced_idx] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # D. Bound Handling
            # -----------------
            trial = np.clip(trial, min_b, max_b)

            # E. Selection
            # ------------
            f_trial = safe_evaluate(trial)

            if f_trial <= fitness[i]:
                # Trial is better or equal: accept it
                population[i] = trial
                fitness[i] = f_trial
                
                # Update adaptive parameters to the ones that worked
                F_arr[i] = F_new
                CR_arr[i] = CR_new
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_individual = trial.copy()

    return best_fitness
