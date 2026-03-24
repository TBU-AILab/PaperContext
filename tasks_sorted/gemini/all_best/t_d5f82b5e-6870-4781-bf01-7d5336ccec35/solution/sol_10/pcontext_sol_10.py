#The improved algorithm is a **Self-Adaptive Differential Evolution with Success-History Adaptation (SHADE) and Multi-Stage Restarts**.
#
#### Improvements Overview
#1.  **SHADE Adaptation**: Replaces the simpler `jDE` parameter adaptation (from the previous best Attempt 1) with **SHADE** (Success-History based Adaptive DE). It uses a historical memory ($M_{CR}, M_F$) to guide the generation of $CR$ and $F$ based on parameters that successfully improved fitness in the past. This allows the algorithm to learn the landscape's requirements (e.g., separability vs. rotation) more effectively.
#2.  **Weighted Lehmer Mean**: The memory update uses a weighted mean based on fitness improvement magnitude. Parameters that resulted in larger fitness drops contribute more to the memory, steering the population faster towards high-quality solutions.
#3.  **Expanded External Archive**: The archive size is increased to `2.0 * pop_size` (up from 1.0) to maintain higher diversity in the `current-to-pbest` mutation strategy, reducing the risk of premature convergence.
#4.  **Robust Restart Mechanism**: Retains the successful multi-stage restart (Convergence and Stagnation triggers) from Attempt 1 but fine-tunes the population sizing and restart behavior to work synergistically with the SHADE history mechanism.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function using SHADE (Success-History Adaptive Differential Evolution)
    combined with an External Archive and a Multi-Stage Restart mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # -------------------------------------------------------------------------
    # Helper: Time Check
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Population size: SHADE benefits from a decent size to build history
    pop_size = int(20 * dim)
    pop_size = max(30, min(100, pop_size))
    
    # Archive size: Larger archive preserves diversity for longer
    archive_size = int(2.0 * pop_size)
    
    # SHADE Memory Parameters
    H = 6  # Memory size
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate
    mem_M_F = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0                  # Memory index pointer
    
    # Optimization Strategy Parameters
    p_best_rate = 0.11  # Top 11% used for p-best selection
    
    # Restart Triggers
    patience = 40       # Generations without global improvement -> Soft Restart
    conv_threshold = 1e-9 # Standard deviation threshold -> Hard Restart

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initial Population
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # External Archive
    archive = []
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_vec = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_time(): return global_best_val
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            global_best_vec = population[i].copy()

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    gen_no_improv = 0
    prev_global_best = global_best_val

    while not check_time():
        
        # Sort population for p-best selection and analysis
        sorted_indices = np.argsort(fitness)
        
        # --- Restart Logic ---
        fit_std = np.std(fitness)
        
        # 1. Convergence Restart (Hard Reset)
        # If population has collapsed to a point, keep best and re-init rest
        if fit_std < conv_threshold:
            # Keep the single best
            keep_idx = sorted_indices[0]
            best_indiv = population[keep_idx].copy()
            best_fit = fitness[keep_idx]
            
            # Re-initialize the rest
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_indiv # Put elite at index 0
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fit
            
            # Reset Memory and Archive to allow new learning
            mem_M_CR.fill(0.5)
            mem_M_F.fill(0.5)
            archive = []
            gen_no_improv = 0
            
            # Evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if check_time(): return global_best_val
                val = func(population[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = population[i].copy()
            
            # Re-sort after restart
            sorted_indices = np.argsort(fitness)

        # 2. Stagnation Restart (Soft Reset)
        # If no improvement for 'patience' gens, keeping searching but inject diversity
        elif gen_no_improv >= patience:
            # Keep top 30% elites, randomize bottom 70%
            elite_count = int(pop_size * 0.3)
            
            # The bottom 70% indices in the sorted list
            reset_indices = sorted_indices[elite_count:]
            
            for idx in reset_indices:
                if check_time(): return global_best_val
                population[idx] = min_b + np.random.rand(dim) * diff_b
                val = func(population[idx])
                fitness[idx] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = population[idx].copy()
            
            # Jolt the memory slightly to encourage new parameter exploration
            mem_M_F = np.clip(mem_M_F + np.random.uniform(-0.1, 0.1, H), 0.1, 1.0)
            
            gen_no_improv = 0
            sorted_indices = np.argsort(fitness)

        # --- SHADE Parameter Generation ---
        # Generate random indices from memory
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idx]
        m_f = mem_M_F[r_idx]
        
        # Generate CR: Normal(M_CR, 0.1), clamped [0, 1]
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0, 1)
        # Ensure CR is not 0 (optional, but helps exploration)
        CR = np.maximum(CR, 0.05) 
        
        # Generate F: Cauchy(M_F, 0.1)
        # Cauchy generation: loc + scale * standard_cauchy
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F boundaries
        # If F > 1, clamp to 1. If F <= 0, regenerate until > 0
        # Vectorized correction for F <= 0 is complex, simple loop repair:
        for k in range(pop_size):
            while F[k] <= 0:
                F[k] = m_f[k] + 0.1 * np.random.standard_cauchy()
            if F[k] > 1:
                F[k] = 1.0

        # --- Evolution Step ---
        success_F = []
        success_CR = []
        diff_fitness = []
        
        # Determine p-best pool size
        num_p_best = max(2, int(pop_size * p_best_rate))
        
        # Prepare for archiving
        new_archive_members = []
        
        for i in range(pop_size):
            if check_time(): return global_best_val
            
            # Strategy: current-to-pbest/1/bin
            # v = x_i + F(x_pbest - x_i) + F(x_r1 - x_r2)
            
            # Select p-best from top sorted indices
            p_best_idx = sorted_indices[np.random.randint(0, num_p_best)]
            x_pbest = population[p_best_idx]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select r2 (distinct from i and r1, from Union(Pop, Archive))
            union_size = pop_size + len(archive)
            r2 = np.random.randint(0, union_size)
            while True:
                is_valid = True
                if r2 < pop_size:
                    if r2 == i or r2 == r1:
                        is_valid = False
                # If r2 >= pop_size, it points to archive, which is distinct from i and r1 (by definition of implementation)
                
                if is_valid:
                    break
                r2 = np.random.randint(0, union_size)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
                
            x_i = population[i]
            
            # Mutation
            mutant = x_i + F[i] * (x_pbest - x_i) + F[i] * (x_r1 - x_r2)
            
            # Crossover
            mask = np.random.rand(dim) < CR[i]
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, x_i)
            trial = np.clip(trial, min_b, max_b)
            
            # Evaluation
            f_trial = func(trial)
            
            # Selection
            if f_trial <= fitness[i]:
                # Improvement or Neutral:
                # 1. Add parent to archive candidates
                new_archive_members.append(x_i.copy())
                
                # 2. Record success for SHADE memory
                if f_trial < fitness[i]:
                    success_CR.append(CR[i])
                    success_F.append(F[i])
                    diff_fitness.append(fitness[i] - f_trial)
                
                # 3. Update Population
                population[i] = trial
                fitness[i] = f_trial
                
                # 4. Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trial.copy()

        # --- End of Generation Processing ---
        
        # 1. Update Archive
        for member in new_archive_members:
            if len(archive) < archive_size:
                archive.append(member)
            else:
                # Random replacement
                rep_idx = np.random.randint(0, archive_size)
                archive[rep_idx] = member
                
        # 2. Update SHADE Memory (Weighted Lehmer Mean)
        if len(success_F) > 0:
            s_F = np.array(success_F)
            s_CR = np.array(success_CR)
            df = np.array(diff_fitness)
            
            # Normalize weights
            total_diff = np.sum(df)
            if total_diff > 0:
                weights = df / total_diff
            else:
                weights = np.ones(len(df)) / len(df)
                
            # Update Mean F (Weighted Lehmer Mean)
            # mean = sum(w * f^2) / sum(w * f)
            mean_F = np.sum(weights * (s_F ** 2)) / (np.sum(weights * s_F) + 1e-10)
            mem_M_F[k_mem] = np.clip(mean_F, 0.1, 1.0)
            
            # Update Mean CR (Weighted Arithmetic Mean)
            mean_CR = np.sum(weights * s_CR)
            mem_M_CR[k_mem] = np.clip(mean_CR, 0.0, 1.0)
            
            # Advance pointer
            k_mem = (k_mem + 1) % H
            
        # 3. Update Stagnation Counter
        if global_best_val < prev_global_best:
            gen_no_improv = 0
            prev_global_best = global_best_val
        else:
            gen_no_improv += 1
            
    return global_best_val
