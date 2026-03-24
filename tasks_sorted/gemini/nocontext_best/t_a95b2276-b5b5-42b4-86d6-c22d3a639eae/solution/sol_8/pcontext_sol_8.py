#The following code implements the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm. 
#
##### Improvements over the previous version:
#1.  **Memory-Based Adaptation**: Instead of simple moving averages, it uses a historical memory ($M_{CR}, M_F$) to store successful parameter configurations. This allows the algorithm to recall and utilize strategies that worked well in different phases of the search.
#2.  **Weighted Updates**: Parameter updates are weighted by the magnitude of fitness improvement. Parameters that yield larger drops in the objective function have a stronger influence on future generations.
#3.  **External Archive**: It maintains an archive of inferior solutions recently replaced. These are used in the mutation operator to maintain diversity and prevent premature convergence, a key feature of modern DE variants like SHADE and L-SHADE.
#4.  **Vectorized Trial Generation**: The generation of mutant vectors and crossover operations are fully vectorized using NumPy, reducing overhead and maximizing the time available for evaluating the objective function `func`.
#5.  **Stagnation/Restart Logic**: Detects convergence (via population variance) and triggers a "soft restart" (keeping the best solution but randomizing the rest and resetting memory) to escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with an external archive and a restart mechanism.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: 20 * dim is a robust default for black-box optimization
    pop_size = int(max(30, 20 * dim))
    
    # SHADE Memory Parameters
    memory_size = 5
    M_CR = np.full(memory_size, 0.5)
    M_F = np.full(memory_size, 0.5)
    k_mem = 0  # Memory index pointer
    
    # External Archive (stores recently replaced solutions to maintain diversity)
    archive = []
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize population randomly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    global_best_val = float('inf')
    global_best_vec = None
    
    # Initial Evaluation Loop
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val if global_best_val != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Check time budget
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        # Sort population by fitness (required for p-best selection strategy)
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # --- Restart Mechanism ---
        # If population variance is extremely low, the algorithm has converged.
        # If max_time allows, we restart the search to find better optima.
        if np.std(fitness) < 1e-9:
            # Soft Restart: Keep global best, randomize the rest
            best_ind = pop[0].copy()
            best_fit = fitness[0]
            
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_ind # Elitism
            
            # Re-evaluate new population (skipping the elite preserved one)
            fitness = np.zeros(pop_size)
            fitness[0] = best_fit
            
            for i in range(1, pop_size):
                if (time.time() - start_time) >= max_time: return global_best_val
                val = func(pop[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = pop[i].copy()
            
            # Reset SHADE memory and archive to avoid bias from previous local optima
            M_CR.fill(0.5)
            M_F.fill(0.5)
            archive = []
            continue # Skip to next generation immediately

        # --- Parameter Generation ---
        # For each individual, pick a random index from the memory
        r_idx = np.random.randint(0, memory_size, pop_size)
        
        # Generate CR (Crossover Rate) ~ Normal(M_CR, 0.1)
        cr_g = np.random.normal(M_CR[r_idx], 0.1)
        cr_g = np.clip(cr_g, 0.0, 1.0)
        
        # Generate F (Scaling Factor) ~ Cauchy(M_F, 0.1)
        # Cauchy distribution helps generate occasional large steps (exploration)
        f_g = M_F[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        f_g = np.clip(f_g, 0.1, 1.0) # Clip to (0.1, 1.0] for stability
        
        # --- Mutation: current-to-pbest/1 ---
        # Equation: V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # 1. Select p-best (top 10% of sorted population)
        p_count = max(1, int(pop_size * 0.1))
        pbest_indices = np.random.randint(0, p_count, pop_size) 
        
        # 2. Select r1 (random from population, distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Fix collisions where r1 == i
        mask_col = (r1_indices == np.arange(pop_size))
        while np.any(mask_col):
            r1_indices[mask_col] = np.random.randint(0, pop_size, np.sum(mask_col))
            mask_col = (r1_indices == np.arange(pop_size))
            
        # 3. Select r2 (random from Population Union Archive, distinct from r1 and i)
        # Construct the pool for r2
        if len(archive) > 0:
            archive_np = np.array(archive)
            pop_all = np.vstack((pop, archive_np))
        else:
            pop_all = pop
            
        r2_indices = np.random.randint(0, len(pop_all), pop_size)
        # (For efficiency in high-level Python, we skip strict r2!=r1 check; collision probability is low)
        
        # --- Vectorized Trial Generation ---
        x_i = pop
        x_pbest = pop[pbest_indices]
        x_r1 = pop[r1_indices]
        x_r2 = pop_all[r2_indices]
        
        # Compute Mutant Vectors
        F_v = f_g[:, np.newaxis] # Reshape for broadcasting
        mutant = x_i + F_v * (x_pbest - x_i) + F_v * (x_r1 - x_r2)
        
        # Binomial Crossover
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr_g[:, np.newaxis]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, x_i)
        
        # Bound Constraints (Clipping)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # --- Selection & Memory Update ---
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        success_cr = []
        success_f = []
        success_diff = []
        
        # Evaluate Trial Vectors
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            t_vec = trial_pop[i]
            t_val = func(t_vec)
            
            if t_val < fitness[i]:
                # Successful Update
                diff = fitness[i] - t_val
                new_pop[i] = t_vec
                new_fitness[i] = t_val
                
                # Record successful parameters
                success_cr.append(cr_g[i])
                success_f.append(f_g[i])
                success_diff.append(diff)
                
                # Add the replaced solution to the archive
                archive.append(pop[i].copy())
                
                # Update Global Best
                if t_val < global_best_val:
                    global_best_val = t_val
                    global_best_vec = t_vec.copy()
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size (Limit to pop_size)
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # Update SHADE Memory (Weighted by fitness improvement)
        if len(success_cr) > 0:
            s_cr = np.array(success_cr)
            s_f = np.array(success_f)
            s_diff = np.array(success_diff)
            
            # Calculate weights based on improvement magnitude
            total_diff = np.sum(s_diff)
            weights = s_diff / total_diff if total_diff > 0 else np.ones(len(s_diff)) / len(s_diff)
            
            # Weighted Lehmer Mean for F
            mean_f_num = np.sum(weights * (s_f**2))
            mean_f_den = np.sum(weights * s_f)
            new_f = mean_f_num / (mean_f_den + 1e-10)
            
            # Weighted Arithmetic Mean for CR
            new_cr = np.sum(weights * s_cr)
            
            # Update memory at current pointer
            M_CR[k_mem] = new_cr
            M_F[k_mem] = new_f
            k_mem = (k_mem + 1) % memory_size
            
    return global_best_val
