#An improved, fully operational, and self-contained algorithm implementing **SHADE-E-R (Success-History Adaptive Differential Evolution with Elitist Restarts)**.
#
#### Improvements & Logic
#1.  **Elitist Restarts**: Unlike standard restarts that discard all progress, this algorithm carries over the global best solution to the new population (`pop[0] = best_x`). This ensures the algorithm never "forgets" the best peak found so far while exploring new areas.
#2.  **Midpoint Bound Handling**: Instead of simple clipping (which causes population bunching at bounds), variables exceeding bounds are reset to the midpoint between the parent value and the bound. This preserves diversity and search direction near the edges of the search space.
#3.  **L-SHADE Mechanics**: Implements the core self-adaptive parameters ($F$ and $CR$) using historical memory and an external archive to maintain diversity, allowing the use of the greedy `current-to-pbest/1` mutation strategy.
#4.  **Robust Stagnation Detection**: Triggers a restart if the population fitness variance collapses (convergence) or if the best solution does not improve for a set number of generations (stagnation), maximizing the utility of the available time.
#5.  **Time Management**: A strict time budget calculation ensures the algorithm returns the best result immediately when the time limit approaches, checking constraints even within inner loops.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using SHADE-E-R: Success-History based Adaptive DE 
    with Elitist Restarts and Midpoint Bound Handling.
    """
    start_time = time.time()
    # Reserve a small buffer (5% or 0.5s) to ensure safe return
    time_budget = max_time - min(max_time * 0.05, 0.5)
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # --- Parameter Setup ---
    # Population size: Adaptive based on dimension, clamped for efficiency
    # Smaller populations (relative to max evals) work better with restarts
    pop_size = int(np.clip(15 * dim, 40, 100))
    
    # SHADE Parameters
    H = 5                   # History memory size
    arc_rate = 2.0          # Archive size relative to population
    archive_size = int(pop_size * arc_rate)
    p_best_rate = 0.11      # Top % for p-best selection
    
    # Global Best Tracker
    best_x = None
    best_f = float('inf')
    
    # --- Main Restart Loop ---
    while True:
        # Check time before starting a new restart
        if time.time() - start_time > time_budget:
            return best_f
            
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(lb[d], ub[d], pop_size + 1)
            u_rand = np.random.rand(pop_size)
            samples = edges[:-1] + (edges[1:] - edges[:-1]) * u_rand
            np.random.shuffle(samples)
            pop[:, d] = samples
            
        # Elitism: Inject global best into new population to preserve learned peaks
        if best_x is not None:
            pop[0] = best_x.copy()
            
        # Evaluate Initial Population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if time.time() - start_time > time_budget:
                return best_f
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_f:
                best_f = val
                best_x = pop[i].copy()
                
        # Initialize Memory (M_CR, M_F)
        mem_M_CR = np.full(H, 0.5)
        mem_M_F = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = np.zeros((archive_size, dim))
        n_arch = 0
        
        # Stagnation tracking
        gens_no_improv = 0
        last_gen_best = np.min(fitness)
        
        # --- Evolutionary Cycle ---
        while True:
            if time.time() - start_time > time_budget:
                return best_f
            
            # Sort population for p-best selection (Minimization)
            # pop[0] becomes the best individual in current gen
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # 2. Parameter Adaptation
            r_idx = np.random.randint(0, H, pop_size)
            r_M_CR = mem_M_CR[r_idx]
            r_M_F = mem_M_F[r_idx]
            
            # CR: Normal(M_CR, 0.1), clipped [0, 1]
            CR = np.random.normal(r_M_CR, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # F: Cauchy(M_F, 0.1)
            u_f = np.random.rand(pop_size)
            F = r_M_F + 0.1 * np.tan(np.pi * (u_f - 0.5))
            
            # F Constraint Handling: Retry if <= 0, clip if > 1
            neg_mask = F <= 0
            while np.any(neg_mask):
                n_neg = np.sum(neg_mask)
                F[neg_mask] = r_M_F[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_neg) - 0.5))
                neg_mask = F <= 0
            F = np.minimum(F, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            p_limit = max(2, int(pop_size * p_best_rate))
            p_indices = np.random.randint(0, p_limit, pop_size)
            x_pbest = pop[p_indices]
            
            # r1: random from population, r1 != i
            # Since pop is sorted, indices are just 0..pop_size-1
            r1 = np.random.randint(0, pop_size, pop_size)
            conflict = r1 == np.arange(pop_size)
            while np.any(conflict):
                r1[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = r1 == np.arange(pop_size)
            x_r1 = pop[r1]
            
            # r2: random from Union(Population, Archive), r2 != i, r2 != r1
            n_total = pop_size + n_arch
            r2 = np.random.randint(0, n_total, pop_size)
            
            # Constraints for r2
            r2_in_pop = r2 < pop_size
            curr_indices = np.arange(pop_size)
            # Conflict if r2 is in pop AND (r2 == i OR r2 == r1)
            conflict = r2_in_pop & ((r2 == curr_indices) | (r2 == r1))
            
            while np.any(conflict):
                r2[conflict] = np.random.randint(0, n_total, np.sum(conflict))
                r2_in_pop = r2 < pop_size
                conflict = r2_in_pop & ((r2 == curr_indices) | (r2 == r1))
                
            # Build x_r2
            x_r2 = np.zeros((pop_size, dim))
            mask_r2_pop = r2 < pop_size
            x_r2[mask_r2_pop] = pop[r2[mask_r2_pop]]
            
            if n_arch > 0:
                mask_r2_arch = ~mask_r2_pop
                # Archive indices are shifted by pop_size
                arch_idx = r2[mask_r2_arch] - pop_size
                x_r2[mask_r2_arch] = archive[arch_idx]
                
            # Compute Mutant Vectors
            F_col = F[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR[:, None]
            # Enforce at least one dimension taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 5. Bound Handling: Midpoint Target
            # If x outside bounds, set to (x_parent + bound) / 2
            mask_lower = trial < lb
            if np.any(mask_lower):
                rows, cols = np.where(mask_lower)
                trial[mask_lower] = (pop[mask_lower] + lb[cols]) / 2.0
                
            mask_upper = trial > ub
            if np.any(mask_upper):
                rows, cols = np.where(mask_upper)
                trial[mask_upper] = (pop[mask_upper] + ub[cols]) / 2.0
                
            # 6. Selection & Evaluation
            trial_fitness = np.zeros(pop_size)
            
            # Periodic time check within evaluation loop
            for i in range(pop_size):
                if (i % 5 == 0) and (time.time() - start_time > time_budget):
                     return best_f
                     
                f_t = func(trial[i])
                trial_fitness[i] = f_t
                
                if f_t < best_f:
                    best_f = f_t
                    best_x = trial[i].copy()
            
            # Determine success
            replace_mask = trial_fitness <= fitness
            better_mask = trial_fitness < fitness
            n_better = np.sum(better_mask)
            
            # 7. Update Archive
            if np.any(replace_mask):
                parents_to_arch = pop[replace_mask]
                num_to_arch = len(parents_to_arch)
                
                if n_arch + num_to_arch <= archive_size:
                    archive[n_arch : n_arch + num_to_arch] = parents_to_arch
                    n_arch += num_to_arch
                else:
                    # Fill remaining space first
                    space = archive_size - n_arch
                    if space > 0:
                        archive[n_arch:] = parents_to_arch[:space]
                        n_arch = archive_size
                        remaining = parents_to_arch[space:]
                    else:
                        remaining = parents_to_arch
                    
                    # Randomly replace existing members
                    if len(remaining) > 0:
                        rnd_idx = np.random.randint(0, archive_size, len(remaining))
                        archive[rnd_idx] = remaining

            # 8. Update Memory (Weighted Lehmer Mean)
            if n_better > 0:
                diff_v = fitness[better_mask] - trial_fitness[better_mask]
                total_diff = np.sum(diff_v)
                
                if total_diff > 0:
                    w = diff_v / total_diff
                    s_CR = CR[better_mask]
                    s_F = F[better_mask]
                    
                    mean_CR = np.sum(w * s_CR)
                    mean_F = np.sum(w * s_F**2) / (np.sum(w * s_F) + 1e-12)
                    
                    mem_M_CR[k_mem] = mean_CR
                    mem_M_F[k_mem] = mean_F
                    k_mem = (k_mem + 1) % H
            
            # 9. Update Population
            pop[replace_mask] = trial[replace_mask]
            fitness[replace_mask] = trial_fitness[replace_mask]
            
            # 10. Restart Trigger Logic
            curr_gen_best = np.min(fitness)
            if np.abs(curr_gen_best - last_gen_best) < 1e-8:
                gens_no_improv += 1
            else:
                gens_no_improv = 0
                last_gen_best = curr_gen_best
                
            # Restart if stagnated (25 gens no improv) or converged (low std)
            if gens_no_improv > 25 or np.std(fitness) < 1e-9:
                break
                
    return best_f
