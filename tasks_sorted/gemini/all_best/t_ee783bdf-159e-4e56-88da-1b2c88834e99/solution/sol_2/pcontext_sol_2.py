#The output value of the best generated algorithm is: 19.9004503424417.
#To improve upon this, I propose implementing the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm. 
#
#The previous best algorithm used a simpler adaptive DE with a restart mechanism. While effective, it likely suffered from inefficient parameter adaptation or lack of diversity maintenance, leading to premature convergence (stagnation) at a sub-optimal value (around 19.9). 
#
#**SHADE** improves this by:
#1.  **Historical Memory**: It learns the optimal mutation factor (`F`) and crossover rate (`CR`) values over time based on successful updates, rather than just randomizing them.
#2.  **External Archive**: It maintains a set of recently discarded superior solutions to preserve diversity. This prevents the population from clustering too tightly too quickly, allowing for better exploration of the search space.
#3.  **Current-to-pbest Mutation**: A robust strategy that guides individuals towards the top $p\%$ of the population, balancing greedy convergence with exploration.
#
#I have also retained a **Restart Mechanism** as a failsafe: if the population variance drops below a threshold (stagnation), the algorithm keeps the best solution and re-initializes the rest to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using SHADE (Success-History based Adaptive Differential Evolution).
    Includes an external archive for diversity and a restart mechanism for stagnation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Standard heuristic is roughly 18*dim, clamped for time/performance
    pop_size = int(np.clip(20 * dim, 40, 100))
    
    # SHADE Memory parameters
    H = 5 # Size of historical memory
    mem_cr = np.full(H, 0.5) # Memory for Crossover Rate
    mem_f = np.full(H, 0.5)  # Memory for Mutation Factor
    k_mem = 0 # Memory index pointer
    
    # Archive to maintain diversity (stores inferior solutions replaced by better ones)
    archive = []
    
    # Parse bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # Uniform random distribution within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_fit = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Strict time check
        if (datetime.now() - start_time) >= time_limit:
            return best_fit
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_idx = i
            
    # Safety check if time ran out during initialization
    if best_idx == -1: return float('inf')

    # --- Main Optimization Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Stagnation Check (Restart Mechanism)
        # If population diversity is lost (low standard deviation), restart.
        # Keep global best, re-initialize the rest.
        if np.std(fitness) < 1e-6:
            for i in range(pop_size):
                if i == best_idx: continue
                
                # Re-initialize
                pop[i] = min_b + np.random.rand(dim) * diff_b
                
                # Evaluate
                if (datetime.now() - start_time) >= time_limit: return best_fit
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fit:
                    best_fit = val
                    best_idx = i
            
            # Clear archive on restart to adapt to new basin
            archive = []
            
        # 2. Parameter Generation (Adaptive)
        # For each individual, select a memory index r randomly
        r_idx = np.random.randint(0, H, pop_size)
        
        # Generate CR: Normal distribution based on memory
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F: Cauchy distribution based on memory
        # Cauchy helps generate occasional large steps to escape locals
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F values
        # If F > 1, clip to 1. If F <= 0, regenerate.
        neg_f_idx = np.where(f <= 0)[0]
        while len(neg_f_idx) > 0:
            f[neg_f_idx] = mem_f[r_idx][neg_f_idx] + 0.1 * np.random.standard_cauchy(len(neg_f_idx))
            neg_f_idx = np.where(f <= 0)[0]
        f = np.clip(f, 0, 1)
        
        # 3. Mutation: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Sort population to find top p-best individuals
        sorted_idx = np.argsort(fitness)
        
        # p is random in [2/N, 0.2] (Standard SHADE strategy)
        # We simplify to a robust fixed top 15% or at least 2 individuals
        num_pbest = max(2, int(0.15 * pop_size))
        
        # Select pbest indices for each individual
        pbest_choices = np.random.randint(0, num_pbest, pop_size)
        pbest_indices = sorted_idx[pbest_choices]
        x_pbest = pop[pbest_indices]
        
        # Select r1: random from population, distinct from current i
        # We use a shift to ensure r1 != i
        shift_r1 = np.random.randint(1, pop_size, pop_size)
        r1_indices = (np.arange(pop_size) + shift_r1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2: random from Union(Population, Archive), distinct from i and r1
        # We allow small collision probability for vectorization speed
        if len(archive) > 0:
            arr_archive = np.array(archive)
            union_pop = np.vstack((pop, arr_archive))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutant Vectors
        f_col = f.reshape(-1, 1)
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover: Binomial
        # Create mask based on CR
        cross_mask = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        
        # Ensure at least one parameter is taken from mutant (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Selection and Memory Update
        succ_cr = []
        succ_f = []
        diff_fit = []
        
        # Evaluate trials
        # We process individually to check time constraints and handle archive
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit
                
            f_trial = func(trial[i])
            
            # Greedy selection (<= allows movement on plateaus)
            if f_trial <= fitness[i]:
                # Improvement found
                if f_trial < fitness[i]:
                    # Add original solution to archive before replacing
                    archive.append(pop[i].copy())
                    
                    # Record successful parameters
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diff_fit.append(fitness[i] - f_trial)
                
                # Replace
                new_pop[i] = trial[i]
                new_fitness[i] = f_trial
                
                # Update Global Best
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_idx = i
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size (Limit to pop_size)
        while len(archive) > pop_size:
            idx_rm = np.random.randint(0, len(archive))
            archive.pop(idx_rm)
            
        # Update Memory (Weighted Lehmer Mean)
        if len(diff_fit) > 0:
            diff_np = np.array(diff_fit)
            scr_np = np.array(succ_cr)
            sf_np = np.array(succ_f)
            
            # Calculate weights based on improvement magnitude
            total_diff = np.sum(diff_np)
            if total_diff > 0:
                weights = diff_np / total_diff
                
                # Weighted Mean for CR
                mean_cr = np.sum(weights * scr_np)
                
                # Weighted Lehmer Mean for F
                sum_wf = np.sum(weights * sf_np)
                if sum_wf > 0:
                    mean_f = np.sum(weights * (sf_np ** 2)) / sum_wf
                else:
                    mean_f = 0.5
                
                # Update memory at pointer k
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                # Advance pointer
                k_mem = (k_mem + 1) % H

    return best_fit
