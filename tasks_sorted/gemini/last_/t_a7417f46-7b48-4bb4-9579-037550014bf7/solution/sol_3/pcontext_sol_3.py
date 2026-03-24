#Here is an improved optimization algorithm.
#
#It implements **L-SHADE** (Linear Success-History based Adaptive Differential Evolution with Linear Population Size Reduction), augmented with a final **Pattern Search** polishing phase.
#
#### Key Improvements over previous attempts:
#1.  **Current-to-pbest Mutation**: Unlike `DE/rand/1` (blind exploration), this strategy directs the search towards the top $p\%$ of best individuals found so far, significantly speeding up convergence on unimodal and multimodal functions.
#2.  **External Archive**: It maintains a history of recently replaced inferior solutions. These are used in the mutation step to increase diversity and prevent the population from collapsing too early.
#3.  **Historical Parameter Adaptation**: Instead of randomizing mutation factors ($F$) and crossover rates ($CR$) or using simple adaptive logic, it uses a memory of *successful* parameters from previous generations to guide the generation of new parameters (using Cauchy and Normal distributions).
#4.  **Linear Population Size Reduction (LPSR)**: The population size dynamically shrinks as time progresses. This allows for massive exploration at the start and highly efficient exploitation (fewer evaluations per generation) towards the end.
#5.  **Pattern Search Polishing**: A fast, derivative-free local search is applied at the very end to fine-tune the best solution up to machine precision limits.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive Differential Evolution
    with Linear Population Size Reduction) and a final Pattern Search polish.
    """
    
    # --- 1. Initialization and Configuration ---
    start_time = datetime.now()
    # Reserve small buffer for final polish (e.g., 5% of time or 2 seconds)
    polish_buffer = min(max_time * 0.05, 2.0)
    limit_time = timedelta(seconds=max_time - polish_buffer)
    
    # Helper for checking evolutionary time limit
    def check_evo_timeout():
        return (datetime.now() - start_time) >= limit_time

    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # L-SHADE Parameters
    # Initial population size: 18 * dim is a robust heuristic for SHADE
    N_init = int(round(18 * dim))
    N_min = 4  # Minimum population size
    pop_size = N_init
    
    # Memory for adaptive parameters (History length H=5)
    H = 5
    mem_M_CR = np.full(H, 0.5)
    mem_M_F = np.full(H, 0.5)
    k_mem = 0  # Memory index counter
    
    # Archive for diversity (External population)
    archive = [] 
    
    # Initialize Population
    # Uniform random initialization
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        if check_evo_timeout(): break
        fitness[i] = func(pop[i])
        
    # Sort population by fitness
    sorted_indices = np.argsort(fitness)
    pop = pop[sorted_indices]
    fitness = fitness[sorted_indices]
    
    best_idx = 0
    best_val = fitness[0]
    best_vec = pop[0].copy()
    
    # --- 2. Main Evolutionary Loop ---
    # We estimate max_generations dynamically based on time elapsed
    # but L-SHADE usually relies on NFE (Max Function Evaluations).
    # We will use time ratio for Linear Population Reduction.
    
    while not check_evo_timeout():
        
        # A. Parameter Adaptation
        # Generate CR and F for each individual based on memory
        rand_indices = np.random.randint(0, H, pop_size)
        r_M_CR = mem_M_CR[rand_indices]
        r_M_F = mem_M_F[rand_indices]
        
        # CR: Normal distribution (mean=r_M_CR, std=0.1), clamped [0, 1]
        CR = np.random.normal(r_M_CR, 0.1, pop_size)
        CR = np.clip(CR, 0.0, 1.0)
        # Special case: if CR is extremely close to 0, fix it to ensure some crossover
        CR[CR < 0.0] = 0.0
        
        # F: Cauchy distribution (loc=r_M_F, scale=0.1)
        # We generate until F > 0. If F > 1, clamp to 1.
        F = np.random.standard_cauchy(pop_size) * 0.1 + r_M_F
        # Repair F values
        attempts = 0
        while np.any(F <= 0) and attempts < 10:
            bad_mask = F <= 0
            F[bad_mask] = np.random.standard_cauchy(np.sum(bad_mask)) * 0.1 + r_M_F[bad_mask]
            attempts += 1
        F = np.clip(F, 0.0, 1.0) # Clip high values to 1.0
        F[F <= 0] = 0.5 # Fallback if regeneration fails
        
        # B. Mutation: 'current-to-pbest/1'
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        
        # p-best selection: Randomly choose from top p% individuals
        # p reduces linearly from 0.2 to 0.05? Standard SHADE uses fixed p usually, 
        # or p_min=2/pop_size. Let's use p=0.11 (standard).
        p_val = 0.11
        top_p_count = max(2, int(round(p_val * pop_size)))
        
        # Vectors arrays
        pbest_indices = np.random.randint(0, top_p_count, pop_size)
        x_pbest = pop[pbest_indices]
        
        # r1 indices (different from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Ensure r1 != i (simple loop correction for robustness)
        for i in range(pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, pop_size)
        x_r1 = pop[r1_indices]
        
        # r2 indices: chosen from Union(Population, Archive)
        # Union size
        pop_arc_size = pop_size + len(archive)
        r2_indices = np.random.randint(0, pop_arc_size, pop_size)
        
        # Construct Union Matrix
        if len(archive) > 0:
            pop_all = np.vstack((pop, np.array(archive)))
        else:
            pop_all = pop
            
        # Ensure r2 != r1 and r2 != i
        for i in range(pop_size):
            while r2_indices[i] == i or r2_indices[i] == r1_indices[i]:
                r2_indices[i] = np.random.randint(0, pop_arc_size)
                
        x_r2 = pop_all[r2_indices]
        
        # Generate Mutant Vectors
        # shape broadcasting: F is (pop_size,), needs (pop_size, 1)
        F_col = F[:, None]
        v = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # C. Crossover: Binomial
        # Mask: True if we take from mutant, False if from parent
        j_rand = np.random.randint(0, dim, pop_size)
        rand_matrix = np.random.rand(pop_size, dim)
        cross_mask = rand_matrix < CR[:, None]
        # Force at least one dim from mutant
        cross_mask[np.arange(pop_size), j_rand] = True
        
        u = np.where(cross_mask, v, pop)
        
        # Boundary Constraint: Bounce back or Clip? Clip is safer for general funcs.
        u = np.clip(u, min_b, max_b)
        
        # D. Selection
        new_pop = np.copy(pop)
        new_fitness = np.copy(fitness)
        
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        
        # Iterate to evaluate (can't vectorise func easily)
        for i in range(pop_size):
            if check_evo_timeout(): break
            
            f_u = func(u[i])
            
            if f_u <= fitness[i]:
                new_pop[i] = u[i]
                new_fitness[i] = f_u
                success_mask[i] = True
                diff_fitness[i] = fitness[i] - f_u
                
                # Add inferior parent to archive
                archive.append(pop[i].copy())
                
                # Update global best
                if f_u < best_val:
                    best_val = f_u
                    best_vec = u[i].copy()
        
        if check_evo_timeout(): break
        
        pop = new_pop
        fitness = new_fitness
        
        # Sort population (important for p-best selection next iter)
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        # Re-sync success masks to sorted order (needed for memory update)
        success_mask = success_mask[sorted_indices]
        diff_fitness = diff_fitness[sorted_indices]
        F = F[sorted_indices]
        CR = CR[sorted_indices]

        # E. Update Memory
        num_success = np.sum(success_mask)
        if num_success > 0:
            # Filter successful parameters
            S_F = F[success_mask]
            S_CR = CR[success_mask]
            # Weights based on fitness improvement
            denom = np.sum(diff_fitness[success_mask])
            if denom > 0:
                weights = diff_fitness[success_mask] / denom
            else:
                weights = np.ones(num_success) / num_success
            
            # Weighted Lehmer Mean for F
            mean_F_num = np.sum(weights * (S_F ** 2))
            mean_F_den = np.sum(weights * S_F)
            if mean_F_den > 0:
                mem_M_F[k_mem] = mean_F_num / mean_F_den
            
            # Weighted Arithmetic Mean for CR
            # SHADE definition: if max(S_CR) == 0 or similar, handle it.
            # Usually simple weighted mean.
            mem_M_CR[k_mem] = np.sum(weights * S_CR)
            
            # Increment memory index
            k_mem = (k_mem + 1) % H
            
        # F. Maintain Archive Size
        # Archive size capacity is usually equal to pop_size
        while len(archive) > pop_size:
            # Remove random elements to maintain size
            idx_to_remove = np.random.randint(0, len(archive))
            archive.pop(idx_to_remove)
            
        # G. Linear Population Size Reduction (LPSR)
        # Calculate progress ratio based on TIME
        elapsed = (datetime.now() - start_time).total_seconds()
        total_available = max_time - polish_buffer
        progress = min(1.0, elapsed / total_available)
        
        # Calculate target population size
        target_size = int(round(N_init + (N_min - N_init) * progress))
        target_size = max(N_min, target_size)
        
        if pop_size > target_size:
            # Reduction: Population is already sorted, remove worst (last)
            reduce_count = pop_size - target_size
            pop = pop[:-reduce_count]
            fitness = fitness[:-reduce_count]
            pop_size = target_size
            
            # Resize archive capacity as well? 
            # SHADE literature usually keeps Archive size = Current Pop Size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))

    # --- 3. Final Polish: Pattern Search (Coordinate Descent variant) ---
    # We use the remaining time to refine the best solution found.
    # This acts like a simplified Hooke-Jeeves or Coordinate Descent.
    
    current_best_x = best_vec.copy()
    current_best_f = best_val
    
    # Step sizes for polishing
    steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    for step in steps:
        if (datetime.now() - start_time).total_seconds() >= max_time: break
        
        improved = True
        while improved:
            improved = False
            # Randomize order of dimensions to avoid bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if (datetime.now() - start_time).total_seconds() >= max_time: break
                
                # Check bounds range for this dimension
                d_range = diff_b[d]
                s_val = step * d_range
                
                # Try positive
                temp_x = current_best_x.copy()
                temp_x[d] += s_val
                if temp_x[d] > max_b[d]: temp_x[d] = max_b[d]
                
                val = func(temp_x)
                if val < current_best_f:
                    current_best_f = val
                    current_best_x = temp_x
                    improved = True
                    continue # Move to next dim
                
                # Try negative
                temp_x = current_best_x.copy()
                temp_x[d] -= s_val
                if temp_x[d] < min_b[d]: temp_x[d] = min_b[d]
                
                val = func(temp_x)
                if val < current_best_f:
                    current_best_f = val
                    current_best_x = temp_x
                    improved = True

    return current_best_f
