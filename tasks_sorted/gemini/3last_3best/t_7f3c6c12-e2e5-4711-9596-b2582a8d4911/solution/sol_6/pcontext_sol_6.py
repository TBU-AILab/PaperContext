The output values and codes for the best generated algorithms are as follows:
1. output value is: 9.79997386118447

 1. algorithm code is:
#The following Python algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with a Restart Mechanism**.
#
#### Key Improvements
#1.  **SHADE Strategy**: Implements the state-of-the-art SHADE algorithm which uses a historical memory to adapt the control parameters ($F$ and $CR$) based on the success of previous generations. This allows the algorithm to learn the landscape properties much faster than static or simple adaptive DE.
#2.  **`current-to-pbest/1` Mutation**: Utilizes a mutation strategy that guides individuals toward the top $p\\%$ of the best solutions found so far (exploitation) while using an external archive of inferior solutions to maintain direction diversity (exploration).
#3.  **External Archive**: Stores recently replaced solutions to provide diverse difference vectors for mutation, preventing premature convergence.
#4.  **Restart Mechanism**: Monitors population diversity (fitness standard deviation). If the population converges to a local optimum before the time limit, it saves the best solution (elite), restarts the rest of the population, and resets the historical memory to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using SHADE (Success-History based Adaptive DE) 
    with a Restart Mechanism within 'max_time'.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Set a strict deadline with a small buffer for return operations
    deadline = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Configuration ---
    # Population size: Adaptive to dimension, clamped to ensure throughput
    # SHADE typically performs well with D-dependent population size
    pop_size = int(np.clip(20 * dim, 50, 150))
    
    # SHADE Memory Parameters
    # History size H
    H = 5
    mem_cr = np.full(H, 0.5) # Memory for Crossover Rate
    mem_f = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0                # Memory index pointer
    
    # Archive Configuration
    # Stores diverse solutions for mutation strategy
    archive_size = pop_size
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= deadline:
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while datetime.now() < deadline:
        
        # 1. Restart Mechanism (Convergence Detection)
        # If population diversity is lost (low std dev), restart to explore new areas
        if np.std(fitness) < 1e-6:
            # Preserve the Elite (Global Best)
            elite = best_sol.copy()
            elite_fit = best_fitness
            
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Inject Elite at index 0
            pop[0] = elite
            fitness[:] = float('inf')
            fitness[0] = elite_fit
            
            # Reset SHADE Memory and Archive
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            n_archive = 0
            
            # Evaluate new population (skipping elite)
            for i in range(1, pop_size):
                if datetime.now() >= deadline: return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            # Continue to next generation immediately
            continue

        # 2. Generate Adaptive Parameters (Vectorized)
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F Constraints
        # If F > 1, clamp to 1. If F <= 0, clamp to small value (force mutation)
        f = np.where(f > 1.0, 1.0, f)
        f = np.where(f <= 0.0, 0.1, f)
        
        # 3. Mutation Strategy: current-to-pbest/1
        # Sort population to identify p-best
        sort_idx = np.argsort(fitness)
        pop_sorted = pop[sort_idx]
        
        # Select p-best individuals (randomly from top p%)
        # p is typically ~0.1 (top 10%)
        top_p_count = max(2, int(pop_size * 0.1))
        pbest_indices = np.random.randint(0, top_p_count, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Select r1 (random from pop, distinct from current i)
        # We ensure approximate distinctness via index manipulation
        r1_indices = np.random.randint(0, pop_size, pop_size)
        conflict_mask = (r1_indices == np.arange(pop_size))
        r1_indices[conflict_mask] = (r1_indices[conflict_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (random from Union(Pop, Archive))
        if n_archive > 0:
            # Create a view of the union
            pop_union = np.vstack((pop, archive[:n_archive]))
        else:
            pop_union = pop
            
        r2_indices = np.random.randint(0, len(pop_union), pop_size)
        x_r2 = pop_union[r2_indices]
        
        # Compute Mutation Vectors
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        f_col = f[:, np.newaxis]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = rand_vals < cr[:, np.newaxis]
        # Ensure at least one dimension is taken from mutant
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 6. Selection and Adaptation Update
        successful_f = []
        successful_cr = []
        improvement_diffs = []
        
        for i in range(pop_size):
            if datetime.now() >= deadline:
                return best_fitness
            
            # Evaluate Candidate
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial < fitness[i]:
                diff = fitness[i] - f_trial
                
                # Add old solution to archive
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Randomly replace an archive member
                    replace_idx = np.random.randint(0, archive_size)
                    archive[replace_idx] = pop[i].copy()
                
                # Accept new solution
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Record success for adaptation
                successful_f.append(f[i])
                successful_cr.append(cr[i])
                improvement_diffs.append(diff)
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    
        # 7. Update SHADE History Memory
        if len(successful_f) > 0:
            s_f = np.array(successful_f)
            s_cr = np.array(successful_cr)
            imp = np.array(improvement_diffs)
            
            # Calculate weights based on fitness improvement amount
            total_imp = np.sum(imp)
            weights = imp / total_imp if total_imp > 0 else np.ones_like(imp) / len(imp)
            
            # Update Mean CR (Weighted Arithmetic Mean)
            mean_cr = np.sum(weights * s_cr)
            
            # Update Mean F (Weighted Lehmer Mean)
            # mean_lehmer = sum(w * f^2) / sum(w * f)
            num = np.sum(weights * (s_f ** 2))
            den = np.sum(weights * s_f)
            mean_f = num / den if den > 0 else 0.5
            
            # Update Memory at pointer k
            mem_cr[k_mem] = mean_cr
            mem_f[k_mem] = mean_f
            k_mem = (k_mem + 1) % H

    return best_fitness


2. output value is: 24.871751996997318

 2. algorithm code is:
#The following Python algorithm implements **L-SHADE (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)** enhanced with a **Session-Based Restart Mechanism** and **Local Search Polishing**.
#
##### Key Improvements
#1.  **L-SHADE with Time-Based Scheduling**: Utilizes the powerful Linear Population Size Reduction (LPSR) strategy. Instead of a fixed population, the algorithm starts with a large diverse population to explore and linearly reduces it to concentrate on exploitation. The reduction schedule is dynamically calculated based on the *remaining time* in the current session.
#2.  **Session-Based Restarts**: Optimization is divided into "sessions". If the population converges (low diversity) or reaches the minimum size before the time limit, the session ends. A "Local Search" phase is triggered to refine the best solution, and then a new session starts. This prevents stagnation and maximizes the utility of the available time.
#3.  **Local Search Polishing**: Before restarting, a coordinate-descent-based local search is applied to the global best solution. This helps to exploit the current basin of attraction to its fullest potential, often squeezing out significant fitness improvements that DE mutation might miss.
#4.  **SHADE Adaptation**: Uses historical memory ($M_{CR}, M_F$) to adapt crossover and mutation parameters, ensuring the algorithm tunes itself to the specific problem landscape.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using L-SHADE with Session-Based Restarts and Local Search.
    """
    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time - 0.05) # Buffer for safety
    
    # --- Configuration ---
    # L-SHADE Population Sizing
    # N_init: Large enough for exploration. N_min: Small for final convergence.
    initial_pop_size = int(np.clip(18 * dim, 50, 200))
    min_pop_size = 4
    
    # SHADE Parameters
    H = 5                   # History Memory Size
    arc_rate = 2.0          # Archive size relative to population
    p_best_rate = 0.11      # Top 11% for p-best selection
    
    # Helper: Bound Constraints
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_sol = None
    
    # --- Optimization Loop (Sessions) ---
    # Each iteration is a "Session" that runs L-SHADE from N_init down to N_min
    while True:
        current_time = datetime.now()
        if current_time >= end_time:
            break
            
        # Calculate remaining budget for this session
        remaining_time = (end_time - current_time).total_seconds()
        # If very little time remains, stop to avoid partial/useless runs
        if remaining_time < 0.1:
            break
            
        session_start_time = current_time
        
        # --- Session Initialization ---
        pop_size = initial_pop_size
        
        # Reset SHADE Memory
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Reset Archive
        archive = np.zeros((int(initial_pop_size * arc_rate), dim))
        n_arc = 0
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Inject Elite (Global Best) to preserve knowledge
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if datetime.now() >= end_time: return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
                best_sol = pop[i].copy()
                
        # Sort Population (Requirement for L-SHADE p-best logic)
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # --- Inner Session Loop (Generations) ---
        while True:
            now = datetime.now()
            if now >= end_time:
                return best_fitness
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress based on TIME relative to this session's budget
            elapsed = (now - session_start_time).total_seconds()
            progress = elapsed / remaining_time # 0.0 (start) -> 1.0 (end)
            if progress > 1.0: progress = 1.0
            
            # Determine target population size
            target_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            # Apply Reduction
            if target_size < pop_size:
                pop_size = target_size
                # Remove worst individuals (at end of sorted array)
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive
                max_arc_size = int(pop_size * arc_rate)
                if n_arc > max_arc_size:
                    # Randomly discard excess archive members
                    keep_idxs = np.random.choice(n_arc, max_arc_size, replace=False)
                    archive[:max_arc_size] = archive[keep_idxs]
                    n_arc = max_arc_size
            
            # 2. Convergence Check
            # If population variance is low OR we reached minimum size, restart.
            std_fit = np.std(fitness)
            if pop_size <= min_pop_size or std_fit < 1e-6:
                
                # --- Local Search (Polishing) ---
                # Attempt to refine the best solution using Coordinate Descent
                # Only if we have a slight buffer time
                if (end_time - datetime.now()).total_seconds() > 0.15:
                    ls_sol = best_sol.copy()
                    ls_fit = best_fitness
                    # Step size heuristic: based on population spread
                    radius = np.max(np.std(pop, axis=0))
                    if radius < 1e-8: radius = 1e-5
                    
                    improved = True
                    # Limit passes to avoid hogging time
                    for _ in range(2): 
                        if not improved: break
                        improved = False
                        
                        for d in range(dim):
                            if datetime.now() >= end_time: return best_fitness
                            
                            original = ls_sol[d]
                            
                            # Try positive step
                            ls_sol[d] = np.clip(original + radius, min_b[d], max_b[d])
                            val = func(ls_sol)
                            if val < ls_fit:
                                ls_fit = val
                                best_fitness = val
                                best_sol = ls_sol.copy()
                                improved = True
                                continue # Keep change
                            
                            # Try negative step
                            ls_sol[d] = np.clip(original - radius, min_b[d], max_b[d])
                            val = func(ls_sol)
                            if val < ls_fit:
                                ls_fit = val
                                best_fitness = val
                                best_sol = ls_sol.copy()
                                improved = True
                                continue
                            
                            # Revert
                            ls_sol[d] = original
                        
                        # Shrink radius for finer search
                        radius *= 0.5

                # Break Inner Loop -> Triggers Restart in Outer Loop
                break

            # 3. Parameter Adaptation (SHADE)
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f = np.where(f > 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.1, f)
            
            # 4. Mutation: current-to-pbest/1
            # Select p-best
            p_count = max(2, int(pop_size * p_best_rate))
            pbest_indices = np.random.randint(0, p_count, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (distinct from i)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            conflict = (r1_indices == np.arange(pop_size))
            r1_indices[conflict] = (r1_indices[conflict] + 1) % pop_size
            x_r1 = pop[r1_indices]
            
            # Select r2 (from Union of Pop and Archive)
            if n_arc > 0:
                union_pop = np.vstack((pop, archive[:n_arc]))
            else:
                union_pop = pop
            r2_indices = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2_indices]
            
            # Calculate Mutation Vectors
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = rand_vals < cr[:, np.newaxis]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # 6. Evaluation and Selection
            success_f = []
            success_cr = []
            diffs = []
            
            max_arc_size = int(pop_size * arc_rate)
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                f_trial = func(trial[i])
                
                if f_trial < fitness[i]:
                    diff = fitness[i] - f_trial
                    
                    # Update Archive
                    if n_arc < max_arc_size:
                        archive[n_arc] = pop[i].copy()
                        n_arc += 1
                    else:
                        if max_arc_size > 0:
                            rep = np.random.randint(0, max_arc_size)
                            archive[rep] = pop[i].copy()
                    
                    # Update Population
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Store Success Data
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diffs.append(diff)
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_sol = trial[i].copy()
            
            # 7. Update SHADE Memory
            if len(success_f) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                s_diff = np.array(diffs)
                
                # Weights based on fitness improvement
                weights = s_diff / np.sum(s_diff)
                
                # Update Mean CR (Weighted Mean)
                mean_cr = np.sum(weights * s_cr)
                
                # Update Mean F (Weighted Lehmer Mean)
                num = np.sum(weights * (s_f ** 2))
                den = np.sum(weights * s_f)
                mean_f = num / den if den > 0 else 0.5
                
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = mean_f
                k_mem = (k_mem + 1) % H
            
            # Sort for next generation (required for p-best and LPSR)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]

    return best_fitness


3. output value is: 24.924021171182545

 3. algorithm code is:
#The following is a fully operational and self-contained Python algorithm. It implements a **Self-Adaptive Differential Evolution (jDE)** strategy with a **Restart Mechanism**.
#
#### Key Improvements:
#1.  **Self-Adaptation (jDE)**: Instead of fixed mutation ($F$) and crossover ($CR$) parameters, each individual adapts its own parameters during the search. This allows the algorithm to dynamically tune itself to the function landscape.
#2.  **Restart Mechanism**: If the population converges (standard deviation of fitness becomes negligible) before the time limit expires, the algorithm restarts the population while preserving the best solution found so far. This prevents getting stuck in local optima and utilizes the full available time for exploration.
#3.  **Strict Time Management**: Time is checked frequently within the inner loops to ensuring the result is returned within `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of 'func' using Self-Adaptive Differential Evolution (jDE)
    with a Restart Mechanism within 'max_time'.
    """
    start_time = datetime.now()
    # Subtract a small buffer to ensure we return before hard timeout
    end_time = start_time + timedelta(seconds=max_time - 0.05)

    # --- Hyperparameters ---
    # Population size: Scale with dimension, clamped to reasonable limits.
    # A size of 10*dim is standard, bounded to ensure generation throughput.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # jDE Control Parameters (Initialized)
    # F: Mutation factor, CR: Crossover probability
    # These are arrays as they adapt per individual
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_idx = -1
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # If time runs out during initialization, return best found so far
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Restart Mechanism
        # If population diversity is lost (converged), restart to search other basins.
        if np.std(fitness) < 1e-6 and pop_size > 5:
            # Save the Elite (Global Best)
            elite_pos = pop[best_idx].copy()
            elite_fit = best_fitness
            
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Inject Elite at index 0
            pop[0] = elite_pos
            
            # Reset Adaptive Parameters
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Reset Fitness array and re-evaluate (skipping elite)
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = elite_fit
            best_idx = 0
            
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_idx = i
            
            # Start new generation immediately after restart
            continue

        # 2. jDE Parameter Adaptation Step
        # Generate new candidate F and CR values for this generation
        mask_f = np.random.rand(pop_size) < 0.1
        mask_cr = np.random.rand(pop_size) < 0.1
        
        F_new = F.copy()
        CR_new = CR.copy()
        
        # F evolves in [0.1, 1.0]
        F_new[mask_f] = 0.1 + np.random.rand(np.sum(mask_f)) * 0.9
        # CR evolves in [0.0, 1.0]
        CR_new[mask_cr] = np.random.rand(np.sum(mask_cr))
        
        # 3. Evolution Loop
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_fitness
            
            # Mutation (DE/rand/1)
            # Pick 3 distinct random individuals excluding i
            # Using rejection sampling for indices (efficient for pop_size >= 20)
            idxs = np.random.randint(0, pop_size, 3)
            while i in idxs or idxs[0]==idxs[1] or idxs[0]==idxs[2] or idxs[1]==idxs[2]:
                idxs = np.random.randint(0, pop_size, 3)
            
            a, b, c = idxs
            
            mutant = pop[a] + F_new[i] * (pop[b] - pop[c])
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(dim) < CR_new[i]
            # Ensure at least one parameter changes
            if not np.any(cross_mask):
                cross_mask[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_mask, mutant, pop[i])
            
            # Bound Constraint Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Trial is better or equal: accept it
                fitness[i] = f_trial
                pop[i] = trial
                # Update individual's control parameters
                F[i] = F_new[i]
                CR[i] = CR_new[i]
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_idx = i
            # If trial is worse, discard trial and keep old F/CR
            
    return best_fitness


Improve.

The following Python algorithm implements **SHADE with Iterated Local Search (SHADE-ILS)**.
#
#### Key Improvements
#1.  **Robust SHADE Implementation**: Inherits the strong parameter adaptation and mutation strategies of SHADE (Success-History Adaptive Differential Evolution).
#2.  **Dual-Trigger Restart**: Restarts the population not only when diversity (standard deviation) is low but also when improvement stagnates for a set number of generations. This handles both premature convergence and stagnation on plateaus.
#3.  **Coordinate Descent Polishing**: Before every restart, a lightweight "Coordinate Descent" local search is applied to the global best solution. This cheap operation often refines the solution to high precision, utilizing the remaining time more effectively than random mutations.
#4.  **Soft Restart**: On restart, the algorithm keeps the best solution (Elite) and resets the memory, allowing the adaptive mechanism to relearn the landscape from a fresh distribution.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using SHADE enhanced with Stagnation Detection 
    and Coordinate Descent Polishing.
    """
    start_time = datetime.now()
    deadline = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Configuration ---
    # Population Size: Adaptive to dimension
    pop_size = int(np.clip(20 * dim, 40, 160))
    
    # SHADE Parameters
    H = 6
    arc_rate = 2.0
    max_arc_size = int(pop_size * arc_rate)
    p_best_rate = 0.11
    
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Restart / Stagnation Parameters
    imp_patience = 30 + dim  # Patience for no improvement
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= deadline: return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # Archive
    archive = np.zeros((max_arc_size, dim))
    n_arc = 0
    
    # Tracking for Stagnation
    last_best_fit = best_fitness
    no_imp_count = 0
    
    # --- Main Loop ---
    while datetime.now() < deadline:
        
        # 1. Parameter Generation
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]
        
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Retry for F <= 0
        neg_mask = f <= 0
        retry_count = 0
        while np.any(neg_mask) and retry_count < 5:
            n_neg = np.sum(neg_mask)
            # Re-generate
            f[neg_mask] = m_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_neg) - 0.5))
            neg_mask = f <= 0
            retry_count += 1
        
        # If still negative after retries, clamp to small positive
        f[f <= 0] = 0.05
        f = np.clip(f, 0, 1)
        
        # 2. Mutation: current-to-pbest/1
        sort_idx = np.argsort(fitness)
        pop_sorted = pop[sort_idx]
        
        n_pbest = max(2, int(pop_size * p_best_rate))
        pbest_idxs = sort_idx[np.random.randint(0, n_pbest, pop_size)]
        x_pbest = pop[pbest_idxs]
        
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Fix r1 collision
        same = (r1_idxs == np.arange(pop_size))
        r1_idxs[same] = (r1_idxs[same] + 1) % pop_size
        x_r1 = pop[r1_idxs]
        
        if n_arc > 0:
            union_pop = np.vstack((pop, archive[:n_arc]))
        else:
            union_pop = pop
        r2_idxs = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idxs]
        
        f_col = f[:, np.newaxis]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 3. Crossover
        j_rand = np.random.randint(0, dim, pop_size)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, np.newaxis]
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 4. Bound Constraint
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Selection
        success_f = []
        success_cr = []
        diffs = []
        
        for i in range(pop_size):
            if datetime.now() >= deadline: return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial < fitness[i]:
                diff = fitness[i] - f_trial
                
                # Update Archive
                if n_arc < max_arc_size:
                    archive[n_arc] = pop[i].copy()
                    n_arc += 1
                else:
                    rep = np.random.randint(0, max_arc_size)
                    archive[rep] = pop[i].copy()
                
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(diff)
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
        
        # 6. Memory Update
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_diff = np.array(diffs)
            
            weights = s_diff / np.sum(s_diff)
            
            m_cr[k_mem] = np.sum(weights * s_cr)
            
            num = np.sum(weights * (s_f ** 2))
            den = np.sum(weights * s_f)
            m_f[k_mem] = num / den if den > 0 else 0.5
            
            k_mem = (k_mem + 1) % H
            
        # 7. Restart Check
        if best_fitness < last_best_fit:
            last_best_fit = best_fitness
            no_imp_count = 0
        else:
            no_imp_count += 1
            
        std_fit = np.std(fitness)
        
        if std_fit < 1e-6 or no_imp_count > imp_patience:
            # --- Polishing (Coordinate Descent) ---
            # Lightweight: 1 pass
            step_sizes = np.maximum(np.std(pop, axis=0), (max_b - min_b)*1e-5)
            improved = False
            for d in range(dim):
                if datetime.now() >= deadline: return best_fitness
                orig = best_sol[d]
                
                # Positive Step
                best_sol[d] = np.clip(orig + step_sizes[d], min_b[d], max_b[d])
                v = func(best_sol)
                if v < best_fitness:
                    best_fitness = v
                    improved = True
                    continue # Keep change
                
                # Negative Step
                best_sol[d] = np.clip(orig - step_sizes[d], min_b[d], max_b[d])
                v = func(best_sol)
                if v < best_fitness:
                    best_fitness = v
                    improved = True
                    continue
                
                # Revert
                best_sol[d] = orig
            
            if improved:
                last_best_fit = best_fitness
                no_imp_count = 0
                
            # --- Restart ---
            # Reset Population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol.copy() # Keep elite
            
            fitness[:] = float('inf')
            fitness[0] = best_fitness
            
            # Reset Memory
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            archive.fill(0)
            n_arc = 0
            
            # Reset Counters
            no_imp_count = 0
            
            # Evaluate new pop
            for i in range(1, pop_size):
                if datetime.now() >= deadline: return best_fitness
                v = func(pop[i])
                fitness[i] = v
                if v < best_fitness:
                    best_fitness = v
                    best_sol = pop[i].copy()
                    last_best_fit = v

    return best_fitness