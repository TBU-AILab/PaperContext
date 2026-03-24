#Here is an optimized algorithm.
#
#### Analysis of Previous Failure
#The previous result (~28.5) strongly suggests the algorithm was optimizing a multimodal function (like Rastrigin) but getting stuck in local optima for a few dimensions. While L-SHADE is powerful, the Python overhead for maintaining history archives and complex adaptation strategies can reduce the number of evaluations within a fixed time limit. Furthermore, standard DE mutations often lack the "fine-tuning" capability to shake a single dimension out of a local basin without disrupting others.
#
#### Proposed Solution: Hybrid Adaptive DE with Coordinate Polishing
#This solution implements **Vectorized jDE (Self-Adaptive Differential Evolution)** combined with an interleaved **Greedy Coordinate Descent (GCD)**.
#
#**Key Features:**
#1.  **jDE Self-Adaptation:** Instead of complex history archives, every individual carries its own $F$ (Mutation Factor) and $CR$ (Crossover Rate). These evolve with the individual, automatically tuning between exploration ($High F$) and exploitation ($Low F$).
#2.  **Vectorized Operations:** The core evolutionary cycle handles the entire population in a single NumPy block, minimizing Python loop overhead.
#3.  **Coordinate Polishing:** Periodically, the algorithm takes the best solution found so far and attempts to "wiggle" each dimension individually. This is extremely effective for functions where variables are loosely coupled (separability), quickly fixing dimensions stuck in local optima (the "28.5" problem).
#4.  **Stagnation-Based Restart:** If the population fails to improve the global best for a set number of generations, a "Soft Restart" is triggered (keeping the best, randomizing the rest) to escape deep basins.
#
#### The Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a Hybrid Vectorized jDE (Self-Adaptive Differential Evolution)
    with Periodic Coordinate Descent polishing and Stagnation Restarts.
    """
    
    # --- Configuration ---
    # Population size: Lower than standard L-SHADE to allow more generations in Python
    pop_size = min(100, max(20, 10 * dim)) 
    
    # jDE Constants
    tau_F = 0.1
    tau_CR = 0.1
    Fl = 0.1
    Fu = 0.9
    
    # Polishing Constants
    polish_interval = 50  # Run coordinate search every N generations
    polish_step_base = 0.05 # Initial step size as % of domain
    
    # Restart Constants
    stagnation_limit = 100 # Generations without improvement before restart
    
    # --- Initialization ---
    start_time = time.time()
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Pre-allocate memory
    pop = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    
    # Initialize jDE control parameters per individual
    # F = 0.5 initially, CR = 0.9 initially
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Global Best tracking
    best_idx = 0
    global_best_val = float('inf')
    global_best_vec = None
    
    # --- Helper: Safe Function Evaluation ---
    def safe_eval(vector):
        # Enforce bounds hard before eval
        clipped = np.clip(vector, min_b, max_b)
        return func(clipped)

    # --- Initial Population ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    for i in range(pop_size):
        if time.time() - start_time > max_time:
            # Emergency exit if initialization takes too long
            if global_best_vec is None: return 0.0 # Should not happen given constraints
            return global_best_val
            
        val = safe_eval(pop[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            global_best_vec = pop[i].copy()
            best_idx = i

    # --- Main Loop ---
    gens_since_improvement = 0
    total_gens = 0
    
    while True:
        # Time Check (Strict)
        if time.time() - start_time >= max_time:
            return global_best_val
            
        # 1. Parameter Adaptation (jDE logic)
        # Create masks for updates
        mask_F = np.random.rand(pop_size) < tau_F
        mask_CR = np.random.rand(pop_size) < tau_CR
        
        # Update F: 0.1 + rand * 0.9
        if np.any(mask_F):
            F[mask_F] = Fl + np.random.rand(np.sum(mask_F)) * Fu
            
        # Update CR: rand [0, 1]
        if np.any(mask_CR):
            CR[mask_CR] = np.random.rand(np.sum(mask_CR))
            
        # 2. Mutation (DE/rand/1/bin) - Vectorized
        # We need r1 != r2 != r3 != i
        # Fast approximate method: permute indices, fix collisions later if critical
        # (In DE/rand/1, slight collision bias is acceptable for speed in Python)
        
        idxs = np.arange(pop_size)
        r1 = np.random.permutation(idxs)
        r2 = np.random.permutation(idxs)
        r3 = np.random.permutation(idxs)
        
        # Ensure r1 != r2 != r3 by simply rotating until distinct (good enough approximation)
        # For simplicity in Python, we trust random permutation for Pop > 20
        
        # V = X_r1 + F * (X_r2 - X_r3)
        # Reshape F for broadcasting: (pop_size, 1)
        F_col = F[:, np.newaxis]
        mutant = pop[r1] + F_col * (pop[r2] - pop[r3])
        
        # Boundary Handling: Random re-initialization usually better than clamping for DE 
        # to prevent stagnation at bounds, but clamping is safer for convex.
        # Let's use Bounce-back/Reflection or Clamping. Using Clamping for speed.
        mutant = np.clip(mutant, min_b, max_b)
        
        # 3. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        # CR_col: (pop_size, 1)
        CR_col = CR[:, np.newaxis]
        cross_mask = rand_vals < CR_col
        
        # Ensure at least one parameter comes from mutant (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        # Use advanced indexing to set one True per row
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 4. Selection (Evaluation)
        improved_this_gen = False
        
        for i in range(pop_size):
            # Check time frequently within loop for expensive functions
            if (i % 5 == 0) and (time.time() - start_time >= max_time):
                return global_best_val
            
            f_trial = safe_eval(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Improvement or equal
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                # Update global best?
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trial_pop[i].copy()
                    best_idx = i
                    improved_this_gen = True
                    gens_since_improvement = 0
            else:
                # If trial failed, revert parameters F and CR (optional jDE step)
                # Here we keep them to allow exploration
                pass

        if not improved_this_gen:
            gens_since_improvement += 1
        
        total_gens += 1

        # 5. Greedy Coordinate Polishing
        # If we found a good basin but are stuck on decimals (common in Rastrigin),
        # wiggle the best vector directly.
        if total_gens % polish_interval == 0:
            
            # Determine step size: reduce over time
            elapsed = time.time() - start_time
            progress = elapsed / max_time
            step_scale = polish_step_base * (1.0 - progress) * diff_b
            
            # Try to improve the best vector dimension by dimension
            # We work on a copy to not corrupt the population immediately
            temp_vec = global_best_vec.copy()
            temp_val = global_best_val
            polish_success = False
            
            # Limit polish to random subset of dims if dim is huge to save time
            dims_to_polish = np.arange(dim)
            if dim > 20:
                dims_to_polish = np.random.choice(dim, 20, replace=False)

            for d in dims_to_polish:
                if time.time() - start_time >= max_time: return global_best_val
                
                # Try Positive direction
                current_x = temp_vec[d]
                step = step_scale[d]
                
                # Try + step
                temp_vec[d] = np.clip(current_x + step, min_b[d], max_b[d])
                val_plus = func(temp_vec)
                
                if val_plus < temp_val:
                    temp_val = val_plus
                    global_best_val = val_plus
                    global_best_vec = temp_vec.copy()
                    pop[best_idx] = temp_vec # Update population
                    fitness[best_idx] = temp_val
                    polish_success = True
                    continue # Moved, go to next dim
                
                # Try - step
                temp_vec[d] = np.clip(current_x - step, min_b[d], max_b[d])
                val_minus = func(temp_vec)
                
                if val_minus < temp_val:
                    temp_val = val_minus
                    global_best_val = val_minus
                    global_best_vec = temp_vec.copy()
                    pop[best_idx] = temp_vec
                    fitness[best_idx] = temp_val
                    polish_success = True
                else:
                    # Revert
                    temp_vec[d] = current_x
            
            if polish_success:
                gens_since_improvement = 0

        # 6. Restart Strategy
        # If stagnant for too long, Soft Restart
        # Keep best, randomizing others.
        if gens_since_improvement > stagnation_limit:
            # Check remaining time
            if (time.time() - start_time) < (max_time * 0.9):
                # Reset population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                # Inject best
                pop[0] = global_best_vec
                fitness[:] = 0 # Will be re-evaluated
                # Re-evaluate all except best (optimization)
                fitness[0] = global_best_val
                
                for i in range(1, pop_size):
                    if time.time() - start_time >= max_time: return global_best_val
                    val = safe_eval(pop[i])
                    fitness[i] = val
                    if val < global_best_val:
                        global_best_val = val
                        global_best_vec = pop[i].copy()
                        best_idx = i
                
                gens_since_improvement = 0
                # Reset parameters
                F[:] = 0.5
                CR[:] = 0.9
    
    return global_best_val
