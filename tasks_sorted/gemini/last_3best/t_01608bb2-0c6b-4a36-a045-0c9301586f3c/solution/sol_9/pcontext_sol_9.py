#An algorithm that implements an **Improved Adaptive Differential Evolution (jDE)** with **Linear Population Size Reduction (LPSR)**, an **External Archive**, and **Cauchy-distributed Parameter Mutation**.
#
#### Key Improvements
#1.  **Diversity Archive**: Unlike standard DE, this implementation maintains an external archive of decent solutions that were recently replaced. The mutation strategy (`current-to-pbest`) draws the second difference vector ($x_{r2}$) from the union of the population and this archive. This significantly mitigates the risk of premature convergence by preserving direction vectors from previous search phases.
#2.  **Adaptive p-Best**: The "greediness" of the mutation strategy is dynamic. The value of $p$ (controlling the top percentage of best individuals to target) linearly decreases from 0.2 (exploration) to 0.05 (exploitation) over time.
#3.  **Cauchy Parameter Mutation**: Instead of Uniform distribution, the scale factor $F$ is updated using a **Cauchy distribution**. The heavy tails of the Cauchy distribution allow for occasional large values of $F$, promoting long jumps to escape local optima, while the central tendency focuses on standard search steps.
#4.  **Linear Population Size Reduction (LPSR)**: The population size starts large to scan the bounds and linearly reduces to a minimal set, forcing the algorithm to transition naturally from global exploration to local refinement.
#5.  **Vectorized Implementation**: All primary evolutionary operations (mutation, crossover, selection, boundary handling) are vectorized using NumPy for maximum throughput.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using an Improved Adaptive DE (jDE variant) with:
    1. Linear Population Size Reduction (LPSR)
    2. External Archive for Diversity (SHADE strategy)
    3. Dynamic 'current-to-pbest' mutation
    4. Cauchy-distributed parameter updates
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population sizing: Start large (exploration), end small (exploitation)
    # Capped at 400 to prevent overhead on simple functions
    pop_size_init = int(min(400, max(50, 20 * dim)))
    pop_size_min = 5
    
    # jDE Control Parameters (Self-Adaptation probabilities)
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # Helper: Check for timeout
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Main Optimization Loop (Restart Mechanism) ---
    # If population converges early, we restart with a new population
    # but keep the global best (Elitism) and reset the archive.
    while not check_timeout():
        
        # 1. Initialize Population
        pop_size = pop_size_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize Parameters (F=0.5, CR=0.5)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.5)
        
        # External Archive: Stores strictly dominated parents to diversify search
        archive = []
        
        # Elitism: Inject global best from previous restarts
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol.copy()
            fitness[0] = best_val
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_timeout(): return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
        
        # --- Evolutionary Cycle ---
        while True:
            # Time Progress
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = elapsed / max_time
            if progress >= 1.0: return best_val
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate target size based on time progress
            target_pop = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                # Reduce population: Keep best individuals
                sorted_idxs = np.argsort(fitness)
                keep_idxs = sorted_idxs[:target_pop]
                
                population = population[keep_idxs]
                fitness = fitness[keep_idxs]
                F = F[keep_idxs]
                CR = CR[keep_idxs]
                pop_size = target_pop
                
                # Resize archive to match current pop_size (prevent archive dominance)
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]
            
            # 2. Convergence Check (Trigger Restart)
            # If variety is lost (std dev is tiny), restart to find new basins
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break
                
            # 3. Parameter Adaptation (jDE Logic with Cauchy Distribution)
            # F comes from Cauchy (0.5, 0.1) to allow larger jumps
            # CR comes from Normal (0.5, 0.1)
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F_new = F.copy()
            CR_new = CR.copy()
            
            if np.any(mask_F):
                n_gen = np.sum(mask_F)
                # Cauchy generation: loc=0.5, scale=0.1
                # Standard Cauchy produces heavy tails
                cauchy_vals = 0.5 + 0.1 * np.random.standard_cauchy(n_gen)
                F_new[mask_F] = np.clip(cauchy_vals, 0.05, 1.0)
                
            if np.any(mask_CR):
                n_gen = np.sum(mask_CR)
                # Normal generation: loc=0.5, scale=0.1
                norm_vals = 0.5 + 0.1 * np.random.randn(n_gen)
                CR_new[mask_CR] = np.clip(norm_vals, 0.0, 1.0)
                
            # 4. Mutation: DE/current-to-pbest/1 with Archive
            # p decreases linearly from 0.2 (exploration) to 0.05 (exploitation)
            p_val = 0.2 - 0.15 * progress
            p_val = max(0.05, p_val)
            n_pbest = max(2, int(pop_size * p_val))
            
            # Identify top p% individuals
            sorted_idxs = np.argsort(fitness)
            pbest_pool = sorted_idxs[:n_pbest]
            
            # Select pbest for each individual
            pbest_idxs = pbest_pool[np.random.randint(0, n_pbest, pop_size)]
            x_pbest = population[pbest_idxs]
            
            # Select r1 (distinct from i)
            idxs_arange = np.arange(pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            # Collision handling: Shift r1 if it equals i
            mask_r1 = r1 == idxs_arange
            r1[mask_r1] = (r1[mask_r1] + 1) % pop_size
            x_r1 = population[r1]
            
            # Select r2 from Union(Population, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.concatenate((population, archive_np), axis=0)
            else:
                union_pop = population
            
            n_union = len(union_pop)
            r2 = np.random.randint(0, n_union, pop_size)
            
            # Fix collisions for r2 (r2 != i and r2 != r1)
            # Loop max 3 times to resolve; usually 0 or 1 is needed
            for _ in range(3):
                # Indices in r2 >= pop_size refer to archive, so no collision with i or r1 possible
                in_pop_mask = r2 < pop_size
                conflict = np.zeros(pop_size, dtype=bool)
                conflict[in_pop_mask] = (r2[in_pop_mask] == idxs_arange[in_pop_mask]) | \
                                        (r2[in_pop_mask] == r1[in_pop_mask])
                if not np.any(conflict):
                    break
                r2[conflict] = np.random.randint(0, n_union, np.sum(conflict))
                
            x_r2 = union_pop[r2]
            
            # Compute Mutant Vector
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            F_col = F_new[:, None]
            mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_j = np.random.rand(pop_size, dim)
            cross_mask = rand_j < CR_new[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # 6. Bound Handling (Reflection / Bounce-Back)
            # Reflect lower bounds
            viol_l = trial_pop < min_b
            if np.any(viol_l):
                trial_pop[viol_l] = 2 * min_b[np.where(viol_l)[1]] - trial_pop[viol_l]
                viol_l_2 = trial_pop < min_b
                trial_pop[viol_l_2] = min_b[np.where(viol_l_2)[1]]
            
            # Reflect upper bounds
            viol_u = trial_pop > max_b
            if np.any(viol_u):
                trial_pop[viol_u] = 2 * max_b[np.where(viol_u)[1]] - trial_pop[viol_u]
                viol_u_2 = trial_pop > max_b
                trial_pop[viol_u_2] = max_b[np.where(viol_u_2)[1]]
                
            # 7. Selection and Update
            fitness_trial = np.zeros(pop_size)
            
            # Evaluate trials
            for i in range(pop_size):
                if check_timeout(): return best_val
                fitness_trial[i] = func(trial_pop[i])
            
            # Identify improvements
            # Note: We use strictly less for archive insertion to avoid duplicates
            better_mask = fitness_trial < fitness 
            improved_mask = fitness_trial <= fitness
            
            # Update Global Best
            min_trial = np.min(fitness_trial)
            if min_trial < best_val:
                best_val = min_trial
                best_idx = np.argmin(fitness_trial)
                best_sol = trial_pop[best_idx].copy()
            
            # Update Archive
            # Add parents that were strictly surpassed by their children
            replaced_parents = population[better_mask]
            for p_vec in replaced_parents:
                archive.append(p_vec.copy())
            
            # Keep archive size <= pop_size
            while len(archive) > pop_size:
                # Random removal
                idx_rm = random.randint(0, len(archive)-1)
                archive.pop(idx_rm)
                
            # Update Population and Parameters
            population[improved_mask] = trial_pop[improved_mask]
            fitness[improved_mask] = fitness_trial[improved_mask]
            F[improved_mask] = F_new[improved_mask]
            CR[improved_mask] = CR_new[improved_mask]

    return best_val
