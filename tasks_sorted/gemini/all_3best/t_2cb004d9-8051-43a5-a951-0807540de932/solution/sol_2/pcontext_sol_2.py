#The following Python code implements a **SHADE (Success-History based Adaptive Differential Evolution) with Restarts** algorithm.
#
##### Improvements over the previous Algorithm:
#1.  **Adaptive Parameter History**: Unlike simple jDE which adapts parameters per individual based on immediate success, this algorithm maintains a **historical memory** ($M_{CR}, M_{F}$) of successful parameters. New parameters are sampled from distributions centered on this memory, allowing the algorithm to learn the global properties of the landscape (e.g., separability, modality) over time.
#2.  **Current-to-pbest Mutation**: Instead of `rand/1`, this uses `current-to-pbest/1`. This strategy drives the search towards the top $p\%$ of the best solutions found so far, significantly speeding up convergence on minimization problems while using difference vectors for exploration.
#3.  **External Archive**: An archive of recently replaced inferior solutions is maintained. This preserves population diversity and prevents the difference vectors in the mutation step from vanishing too quickly, avoiding premature convergence.
#4.  **Restart Mechanism**: If the population converges (stagnates), the algorithm restarts with a fresh population (resetting the search) while preserving the global best. This ensures the solver effectively utilizes the entire `max_time` to find better minima in multimodal landscapes.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution) 
    with Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size
    # SHADE typically performs well with a population size around 18 * dim.
    # We clamp it to ensure iterations are fast enough within the time limit.
    pop_size = int(max(20, 18 * dim))
    if pop_size > 80: 
        pop_size = 80
    
    # Global best tracker
    best_fitness = float('inf')
    
    # --- SHADE Parameters ---
    H = 5  # Size of historical memory
    
    # --- Restart Loop ---
    # Algorithm restarts if the population converges/stagnates
    while True:
        # Check time before starting a new run
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        # Initialize Memory for Adaptive Parameters
        # M_cr (Crossover) and M_f (Mutation factor) initialized to 0.5
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
        
        # Archive to store inferior solutions (maintains diversity)
        archive = []
        
        # --- Evolution Loop ---
        while True:
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            # Sort population by fitness (needed for p-best selection)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Check for Convergence (Stagnation)
            # If population variance is negligible, break to trigger restart
            if np.std(fitness) < 1e-6 or (fitness[-1] - fitness[0]) < 1e-6:
                break
            
            # 1. Parameter Generation based on History
            # Select random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            
            # Generate CR: Normal distribution around memory
            cr = np.random.normal(mem_cr[r_idx], 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy distribution around memory
            # Numpy standard_cauchy is dist(0,1). Scale to dist(loc, 0.1)
            f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
            # F constraints: if > 1 clamp to 1. If <= 0, clamp to small value (re-sampling is slow)
            f[f > 1] = 1.0
            f[f <= 0] = 0.1
            
            # 2. Mutation: current-to-pbest/1
            # vector = current + F*(pbest - current) + F*(r1 - r2)
            
            # Select p-best (top 15% of population)
            p_best_size = max(2, int(pop_size * 0.15))
            p_best_idxs = np.random.randint(0, p_best_size, pop_size)
            x_pbest = population[p_best_idxs]
            
            # Select r1 (random from population, distinct from i implicitly via randomization)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            # Ensure r1 != i (simple shift strategy)
            mask_fail = (r1_idxs == np.arange(pop_size))
            r1_idxs[mask_fail] = (r1_idxs[mask_fail] + 1) % pop_size
            x_r1 = population[r1_idxs]
            
            # Select r2 (random from Population U Archive)
            if len(archive) > 0:
                # Create Union
                arr_archive = np.array(archive)
                union_pop = np.vstack((population, arr_archive))
            else:
                union_pop = population
            
            r2_idxs = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2_idxs]
            
            # Calculate Mutant Vectors (Vectorized)
            diff_pbest = x_pbest - population
            diff_r1r2 = x_r1 - x_r2
            mutant = population + f[:, None] * diff_pbest + f[:, None] * diff_r1r2
            
            # 3. Crossover (Binomial)
            # Generate mask
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, population)
            # Boundary Constraint Handling
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection and Updates
            success_f = []
            success_cr = []
            
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_fitness
                
                # Evaluate Trial
                val = func(trial[i])
                
                if val <= fitness[i]:
                    # Trial is better or equal
                    new_population[i] = trial[i]
                    new_fitness[i] = val
                    
                    # Store successful parameters
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                    # Add original parent to archive
                    archive.append(population[i].copy())
                    
                    # Update Global Best
                    if val < best_fitness:
                        best_fitness = val
            
            # Apply updates
            population = new_population
            fitness = new_fitness
            
            # 5. Archive Maintenance
            # If archive exceeds population size, remove random elements
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
            
            # 6. Update Historical Memory
            if success_f:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                
                # Calculate Lehmer Mean for F
                mean_f = np.sum(s_f**2) / np.sum(s_f)
                
                # Calculate Arithmetic Mean for CR
                mean_cr = np.mean(s_cr)
                
                # Update Memory (weighted average with previous)
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % H

    return best_fitness
