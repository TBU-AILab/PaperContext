#The improved algorithm is a self-contained implementation of **JADE (Adaptive Differential Evolution with Optional External Archive)** combined with a **Multi-Stage Restart Mechanism**.
#
#### Improvements Overview
#1.  **JADE Strategy**: Builds upon the successful jDE approach (Attempt 3) by adding an **External Archive**. The `current-to-pbest/1` mutation strategy benefits significantly from an archive, as it preserves population diversity by using historically replaced "inferior" solutions in the difference vector calculation ($x_{r2}$ from $Population \cup Archive$).
#2.  **Adaptive Parameters**: Implements the parameter adaptation mechanism where `F` (Scaling Factor) and `CR` (Crossover Rate) evolve with individual success, allowing the algorithm to auto-tune to the fitness landscape.
#3.  **Two-Stage Restart**:
#    *   **Convergence Restart**: If the population variance drops below a threshold (`1e-9`), the algorithm keeps only the single best solution and re-initializes the rest. This is crucial for escaping sharp local optima where the whole population has collapsed.
#    *   **Stagnation Restart**: If the global best solution does not improve for a set number of generations (`patience`), a "Soft Restart" is triggered. It keeps the top 30% elite and re-initializes the bottom 70% to inject fresh exploration without losing the current search trajectory.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function using Adaptive Differential Evolution (JADE-like)
    with External Archive and Multi-Stage Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Population size: Robust heuristic based on dimension, clamped for safety
    pop_size = int(15 * dim)
    if pop_size < 30: pop_size = 30
    if pop_size > 100: pop_size = 100
    
    # Archive size (stores historical vectors to preserve diversity)
    archive_size = pop_size
    
    # Adaptation Parameters (jDE style)
    tau_F = 0.1   # Learning rate for F
    tau_CR = 0.1  # Learning rate for CR
    
    # Mutation Strategy Parameter
    p_best_rate = 0.11  # Top 11% used for p-best selection
    
    # Restart Triggers
    patience = 40  # Generations without improvement before soft restart
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize Adaptive Parameters per individual
    F = np.full(pop_size, 0.5) 
    CR = np.full(pop_size, 0.9)
    
    # External Archive
    archive = []
    
    # Global Best Tracking
    global_best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return global_best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            
    # -------------------------------------------------------------------------
    # Optimization Loop
    # -------------------------------------------------------------------------
    gen_no_improv = 0
    
    while datetime.now() - start_time < time_limit:
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        
        # --- Restart Logic ---
        std_fit = np.std(fitness)
        
        # 1. Convergence Restart (Population collapsed)
        if std_fit < 1e-9:
            # Hard Restart: Keep 1 best, re-init all others, clear archive
            keep_idx = sorted_indices[0]
            
            # Re-init the rest
            for k in range(1, pop_size):
                idx = sorted_indices[k]
                if datetime.now() - start_time >= time_limit: return global_best_val
                
                population[idx] = min_b + np.random.rand(dim) * diff_b
                val = func(population[idx])
                fitness[idx] = val
                
                # Reset parameters
                F[idx] = 0.5
                CR[idx] = 0.9
                
                if val < global_best_val:
                    global_best_val = val
            
            # Clear archive to allow fresh exploration
            archive = []
            gen_no_improv = 0
            # Re-sort to maintain index order for p-best selection below
            sorted_indices = np.argsort(fitness)
            
        # 2. Stagnation Restart (No improvement for too long)
        elif gen_no_improv >= patience:
            # Soft Restart: Keep top 30%, re-init bottom 70%
            cut = int(pop_size * 0.3)
            
            for k in range(cut, pop_size):
                idx = sorted_indices[k]
                if datetime.now() - start_time >= time_limit: return global_best_val
                
                population[idx] = min_b + np.random.rand(dim) * diff_b
                val = func(population[idx])
                fitness[idx] = val
                
                # Reset parameters
                F[idx] = 0.5
                CR[idx] = 0.9
                
                if val < global_best_val:
                    global_best_val = val
            
            gen_no_improv = 0
            sorted_indices = np.argsort(fitness)

        # --- Evolution Step ---
        prev_best = global_best_val
        
        # Determine p-best pool
        num_p_best = max(2, int(pop_size * p_best_rate))
        p_best_indices = sorted_indices[:num_p_best]
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # 1. Parameter Adaptation (jDE)
            # F Adaptation: small prob to reset to random [0.1, 1.0], else keep
            if np.random.rand() < tau_F:
                Fi = 0.1 + 0.9 * np.random.rand()
            else:
                Fi = F[i]
                
            # CR Adaptation: small prob to reset to random [0.0, 1.0], else keep
            if np.random.rand() < tau_CR:
                CRi = np.random.rand()
            else:
                CRi = CR[i]
            
            # 2. Mutation: DE/current-to-pbest/1/bin with Archive
            # V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
            
            # Select p-best
            r_best = np.random.choice(p_best_indices)
            x_pbest = population[r_best]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select r2 (distinct from i and r1, from Union(Pop, Archive))
            union_len = pop_size + len(archive)
            while True:
                r2 = np.random.randint(0, union_len)
                if r2 < pop_size:
                    if r2 != i and r2 != r1:
                        x_r2 = population[r2]
                        break
                else:
                    # Index maps into archive
                    x_r2 = archive[r2 - pop_size]
                    break
            
            x_i = population[i]
            
            # Compute Mutant Vector
            mutant = x_i + Fi * (x_pbest - x_i) + Fi * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            mask = np.random.rand(dim) < CRi
            # Ensure at least one parameter is changed
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, x_i)
            
            # 4. Bound Handling (Clip)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Evaluation & Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Update Archive: Add parent before replacing it
                if len(archive) < archive_size:
                    archive.append(x_i.copy())
                else:
                    # Random replacement in archive
                    archive[np.random.randint(0, archive_size)] = x_i.copy()
                
                # Replace individual
                population[i] = trial
                fitness[i] = f_trial
                
                # Update Parameters
                F[i] = Fi
                CR[i] = CRi
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
        
        # Update Stagnation Counter
        if global_best_val < prev_best:
            gen_no_improv = 0
        else:
            gen_no_improv += 1
            
    return global_best_val
