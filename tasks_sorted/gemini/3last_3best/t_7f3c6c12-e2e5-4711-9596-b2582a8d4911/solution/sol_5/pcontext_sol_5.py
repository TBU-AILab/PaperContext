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
