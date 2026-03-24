#Here is a fully operational implementation using the **Self-Adaptive Differential Evolution (jDE)** algorithm with a **Restart Strategy**.
#
#### Algorithm Explanation
#1.  **jDE (Self-Adaptive DE)**: Unlike standard DE which uses fixed control parameters ($F$ and $CR$), jDE encodes these parameters into each individual. They evolve alongside the solution, allowing the algorithm to automatically learn whether exploration (high $F$) or exploitation (low $F$) is needed for the specific problem landscape.
#2.  **Restart Strategy**: If the population converges (variance becomes negligible) before the time limit expires, the algorithm restarts the population while keeping the global best solution. This prevents the solver from getting stuck in local optima and wasting the remaining time budget.
#3.  **Vectorization**: The mutation and crossover steps are vectorized using NumPy to maximize the number of generations processed within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) with Restarts.
    """
    start_time = datetime.now()
    # Use 98% of max_time to ensure safe return before external timeout
    time_limit = timedelta(seconds=max_time * 0.98)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population size: adaptive to dimension but bounded for performance
    # Range [20, 100] is generally robust for constrained-time black-box optimization
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    best_fitness = float('inf')
    best_solution = None
    
    # --- Restart Loop ---
    # Restarts help if the algorithm converges to a local optimum early.
    while True:
        # Check time before starting a new run/restart
        if datetime.now() - start_time >= time_limit:
            return best_fitness

        # Initialize Population: Uniform random within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject best solution found so far into new population (if any)
        # This ensures we never lose the global best during a restart.
        if best_solution is not None:
            pop[0] = best_solution
            
        # jDE Adaptive Parameters Initialization
        # F (Mutation Factor) initialized to 0.5
        # CR (Crossover Probability) initialized to 0.9
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Current population fitness
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            
            try:
                val = func(pop[i])
            except:
                val = float('inf')
                
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = pop[i].copy()
                
        # --- Generations Loop ---
        while True:
            # Check time strictly
            if datetime.now() - start_time >= time_limit:
                return best_fitness
                
            # 1. jDE Parameter Adaptation
            # Probabilities for update: tau1 = tau2 = 0.1
            mask_F = np.random.rand(pop_size) < 0.1
            mask_CR = np.random.rand(pop_size) < 0.1
            
            trial_F = F.copy()
            trial_CR = CR.copy()
            
            # F_new = 0.1 + 0.9 * rand() (Explore range [0.1, 1.0])
            trial_F[mask_F] = 0.1 + 0.9 * np.random.rand(mask_F.sum())
            # CR_new = rand() (Explore range [0.0, 1.0])
            trial_CR[mask_CR] = np.random.rand(mask_CR.sum())
            
            # 2. Mutation: DE/rand/1
            # We need indices r1 != r2 != r3 != i
            idxs = np.arange(pop_size)
            
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            # Resolve collisions by shifting indices (vectorized approximation)
            r1 = np.where(r1 == idxs, (r1 + 1) % pop_size, r1)
            r2 = np.where((r2 == idxs) | (r2 == r1), (r2 + 2) % pop_size, r2)
            r3 = np.where((r3 == idxs) | (r3 == r1) | (r3 == r2), (r3 + 3) % pop_size, r3)
            
            # Mutant Vector Calculation: V = X_r1 + F * (X_r2 - X_r3)
            # Reshape F for broadcasting: (pop_size, 1)
            F_col = trial_F[:, np.newaxis]
            mutant = pop[r1] + F_col * (pop[r2] - pop[r3])
            
            # Bound Constraint Handling (Clipping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 3. Crossover: Binomial
            # Create mask where rand < CR
            cross_mask = np.random.rand(pop_size, dim) < trial_CR[:, np.newaxis]
            
            # Ensure at least one dimension is taken from mutant to avoid duplication
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            
            # Combine masks
            cross_mask = cross_mask | j_rand_mask
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # 4. Selection & Evaluation
            # We loop here to check time budget during potentially expensive evaluations
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_fitness
                
                try:
                    f_trial = func(trial_pop[i])
                except:
                    f_trial = float('inf')
                
                # Greedy selection: if trial is better or equal, it survives
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Key jDE step: Inherit the successful parameters
                    F[i] = trial_F[i]
                    CR[i] = trial_CR[i]
                    
                    # Update global best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()

            # 5. Convergence Check
            # If population fitness variance is essentially zero, we have converged.
            # Break inner loop to trigger a Restart in the outer loop.
            if np.std(fitness) < 1e-8:
                break

    return best_fitness
