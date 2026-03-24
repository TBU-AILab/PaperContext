#To improve the performance of the optimization algorithm, I propose implementing the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm. 
#
#The previous algorithm (jDE) used a static mutation strategy ('current-to-best/1') which can lead to premature convergence, and a restart mechanism that discards learned information. 
#
#**SHADE** improves upon this by:
#1.  **Maintaining an External Archive**: It stores inferior solutions recently replaced in the population. These are used in the mutation step to maintain diversity and prevent stagnation.
#2.  **Adaptive Parameter History**: Instead of resetting parameters or using simple adaptation, it builds a memory (History) of successful `F` (scaling factor) and `CR` (crossover rate) values based on fitness improvements, guiding the generation of new parameters.
#3.  **Current-to-pbest Mutation**: This strategy balances greediness (moving towards the top $p\%$ best individuals) with exploration (difference vectors involving the archive).
#
#This approach is significantly more robust for various objective functions (multimodal, non-separable) and generally achieves much lower minima than standard DE or jDE.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using SHADE (Success-History based Adaptive Differential Evolution).
    """
    # Initialize timer
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Helper for timeout check
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Preprocessing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population size: Compromise between exploration capability and time constraints.
    # For SHADE, 18*dim is standard, but we clamp it to [30, 100] to ensure 
    # reasonable iteration count within limited time.
    pop_size = int(np.clip(10 * dim, 30, 100))
    
    # SHADE Parameters
    H = 6  # Size of the historical memory
    M_CR = np.full(H, 0.5) # Memory for Crossover Rate
    M_F = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0              # Memory index pointer
    
    archive = []           # External archive to maintain diversity
    
    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = None

    # Evaluate Initial Population
    for i in range(pop_size):
        if check_timeout():
            # If timeout during init, return best found or fallback
            return best_val if best_val != float('inf') else func(population[0])
            
        try:
            val = func(population[i])
        except Exception:
            val = float('inf')
            
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()
            
    # --- Main Loop ---
    while not check_timeout():
        
        # 1. Sort population by fitness (needed for p-best selection)
        sorted_indices = np.argsort(fitness)
        pop_sorted = population[sorted_indices]
        
        # 2. Parameter Adaptation
        # Select random memory indices for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # Generate CR ~ Normal(m_cr, 0.1), clipped to [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(m_f, 0.1)
        # F must be > 0. If > 1, clip to 1.
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f = np.minimum(f, 1.0)
        
        # Retry logic for F <= 0 (vectorized)
        bad_f = f <= 0
        while np.any(bad_f):
            count = np.sum(bad_f)
            f[bad_f] = m_f[bad_f] + 0.1 * np.random.standard_cauchy(count)
            f = np.minimum(f, 1.0)
            bad_f = f <= 0
            
        # 3. Mutation Strategy: current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Select p-best individuals (top p%, where p is random in [2/N, 0.2])
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        top_p = int(p_val * pop_size)
        if top_p < 2: top_p = 2
        
        # X_pbest indices
        pbest_indices = np.random.randint(0, top_p, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # X_r1 indices (random from population)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # X_r2 indices (from Union of Population and Archive)
        n_arc = len(archive)
        if n_arc > 0:
            total_size = pop_size + n_arc
            r2_indices = np.random.randint(0, total_size, pop_size)
            
            # Construct x_r2 array efficiently
            x_r2 = np.empty((pop_size, dim))
            
            # Indices < pop_size come from current population
            mask_pop = r2_indices < pop_size
            x_r2[mask_pop] = population[r2_indices[mask_pop]]
            
            # Indices >= pop_size come from archive
            mask_arc = ~mask_pop
            if np.any(mask_arc):
                # Map global index to archive index
                arc_idx = r2_indices[mask_arc] - pop_size
                # Use list comprehension for accessing archive list
                x_r2[mask_arc] = np.array([archive[k] for k in arc_idx])
        else:
            # If archive empty, sample r2 from population
            r2_indices = np.random.randint(0, pop_size, pop_size)
            x_r2 = population[r2_indices]
            
        # Compute Mutant Vectors
        mutant = population + f[:, None] * (x_pbest - population) + f[:, None] * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        # Create mask: True if rand < CR or index == j_rand
        j_rand = np.random.randint(0, dim, pop_size)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, None]
        # Enforce at least one dimension changed
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, population)
        
        # 5. Bound Constraints (Clip)
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Selection & Archive/Memory Update
        succ_f = []
        succ_cr = []
        diff_f = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if check_timeout():
                return best_val
            
            try:
                t_val = func(trial[i])
            except Exception:
                t_val = float('inf')
                
            # Greedy Selection
            if t_val <= fitness[i]:
                # If strictly better, record success info
                if t_val < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_f.append(fitness[i] - t_val)
                    # Add replaced individual to archive
                    archive.append(population[i].copy())
                
                new_pop[i] = trial[i]
                new_fit[i] = t_val
                
                # Update Global Best
                if t_val < best_val:
                    best_val = t_val
                    best_vec = trial[i].copy()
                    
        # Update Population
        population = new_pop
        fitness = new_fit
        
        # Maintain Archive Size (Limit to pop_size)
        if len(archive) > pop_size:
            random.shuffle(archive)
            archive = archive[:pop_size]
            
        # Update Historical Memory (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            diff = np.array(diff_f)
            
            # Weights based on fitness improvement
            total_diff = np.sum(diff)
            if total_diff == 0:
                weights = np.ones(len(diff)) / len(diff)
            else:
                weights = diff / total_diff
            
            # Update M_CR (Weighted Mean)
            m_cr_new = np.sum(weights * succ_cr)
            M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
            
            # Update M_F (Weighted Lehmer Mean: sum(w*f^2)/sum(w*f))
            num = np.sum(weights * (succ_f ** 2))
            den = np.sum(weights * succ_f)
            m_f_new = num / den if den > 0 else 0.5
            M_F[k_mem] = np.clip(m_f_new, 0, 1)
            
            # Advance memory index
            k_mem = (k_mem + 1) % H

    return best_val
