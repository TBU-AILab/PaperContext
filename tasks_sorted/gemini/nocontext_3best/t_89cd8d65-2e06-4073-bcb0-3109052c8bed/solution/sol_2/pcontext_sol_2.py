#Here is a fully operational implementation using the **SHADE (Success-History based Adaptive Differential Evolution) with Restarts** algorithm.
#
#SHADE is an advanced variant of Differential Evolution that is widely considered state-of-the-art for continuous optimization. It outperforms standard DE and jDE by automatically adapting control parameters ($F$ and $CR$) using a historical memory of successful updates, and by using an external archive to maintain diversity.
#
#### Algorithm Explanation
#1.  **Historical Memory**: The algorithm maintains a memory of $F$ (scaling factor) and $CR$ (crossover rate) values that historically led to improvements. New parameters are sampled from distributions based on this memory, biasing the search towards successful strategies.
#2.  **Current-to-pbest Mutation**: Instead of mutating towards a random individual, agents mutate towards one of the top $p\%$ best individuals ($p$-best). This accelerates convergence.
#3.  **External Archive**: Inferior solutions replaced during selection are moved to an archive. The difference vector ($x_{r1} - x_{r2}$) can draw $x_{r2}$ from this archive, significantly increasing the diversity of potential search directions.
#4.  **Restart Strategy**: If the population converges (variance drops) or the search stagnates, the algorithm restarts with a fresh population (keeping the global best found so far) to explore new areas of the search space within the remaining time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive DE) with Restarts.
    """
    start_time = datetime.now()
    # Use 98% of the time budget to ensure we return before the external timeout
    time_limit = timedelta(seconds=max_time * 0.98)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Population Size
    # Adaptive based on dimension. Range [30, 150] is robust for constrained time.
    # Larger populations explore better but are slower per generation.
    pop_size = int(np.clip(18 * dim, 30, 150))
    
    # External Archive Size (maintains diversity)
    archive_size = pop_size
    
    # Memory for Adaptive Parameters (F and CR)
    H = 10  # Memory size
    mem_f = np.full(H, 0.5)  # Initialize Scaling Factor memory
    mem_cr = np.full(H, 0.5) # Initialize Crossover Rate memory
    k_mem = 0 # Pointer to the current memory slot to update

    best_fitness = float('inf')
    best_solution = None

    # --- Main Loop (Restart Strategy) ---
    while True:
        # Check time budget
        if datetime.now() - start_time >= time_limit:
            return best_fitness

        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))

        # Elitism: Inject global best solution if we have one (from previous restart)
        start_eval_idx = 0
        if best_solution is not None:
            pop[0] = best_solution.copy()
            fitness[0] = best_fitness
            start_eval_idx = 1 # Skip re-evaluating the injected best

        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            try:
                val = func(pop[i])
            except:
                val = float('inf')
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = pop[i].copy()

        # Initialize Archive for this run
        archive = []

        # --- Generation Loop ---
        while True:
            # Time check
            if datetime.now() - start_time >= time_limit:
                return best_fitness

            # 2. Parameter Generation (Adaptive)
            # Assign a random memory index to each individual
            r_idxs = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idxs]
            m_cr = mem_cr[r_idxs]

            # Generate CR: Normal distribution around memory, clipped to [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)

            # Generate F: Cauchy distribution around memory
            # F must be > 0. If > 1, clip to 1.
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Repair invalid F values
            # If F <= 0, retry (simple heuristic: up to 5 times then fallback)
            for _ in range(5):
                bad_f = f <= 0
                if not np.any(bad_f):
                    break
                f[bad_f] = m_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            f[f <= 0] = 0.5 # Fallback
            f = np.minimum(f, 1.0)

            # 3. Mutation: current-to-pbest/1
            # Sort population to find top p-best individuals
            sorted_indices = np.argsort(fitness)
            # p_best rate controls greediness (0.11 is a standard robust value)
            num_p_best = max(2, int(pop_size * 0.11))
            top_indices = sorted_indices[:num_p_best]
            
            # Select pbest for each individual randomly from top %
            pbest_idxs = np.random.choice(top_indices, pop_size)
            x_pbest = pop[pbest_idxs]

            # Select r1: Random distinct from current i
            idxs = np.arange(pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            # Shift collision with self
            r1 = np.where(r1 == idxs, (r1 + 1) % pop_size, r1)
            x_r1 = pop[r1]

            # Select r2: Random distinct from i and r1, from Union(Pop, Archive)
            # Build union pool
            if len(archive) > 0:
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
            pool_size = len(pool)
            
            r2 = np.random.randint(0, pool_size, pop_size)
            # Handle collisions: r2 != i and r2 != r1
            # Note: i and r1 are always < pop_size. r2 can be >= pop_size (archive).
            collision = (r2 == idxs) | (r2 == r1)
            if np.any(collision):
                r2[collision] = (r2[collision] + 1) % pool_size
            x_r2 = pool[r2]

            # Compute Mutant Vector: v = x + F*(pbest - x) + F*(r1 - r2)
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)

            # Bound Constraints (Clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 4. Crossover (Binomial)
            # Force at least one dimension from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_mask = np.zeros((pop_size, dim), dtype=bool)
            j_mask[np.arange(pop_size), j_rand] = True
            
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            trial_mask = cross_mask | j_mask
            trial_pop = np.where(trial_mask, mutant, pop)

            # 5. Selection and Memory Update Prep
            succ_f = []
            succ_cr = []
            diff_fitness = []

            for i in range(pop_size):
                # Check time strictly before expensive eval
                if datetime.now() - start_time >= time_limit:
                    return best_fitness

                try:
                    f_trial = func(trial_pop[i])
                except:
                    f_trial = float('inf')

                # Greedy Selection
                if f_trial <= fitness[i]:
                    # If strictly better, store info for adaptation
                    if f_trial < fitness[i]:
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - f_trial)

                        # Add parent to archive before replacement
                        if len(archive) < archive_size:
                            archive.append(pop[i].copy())
                        else:
                            # Randomly replace an archive member
                            ridx = np.random.randint(0, archive_size)
                            archive[ridx] = pop[i].copy()

                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]

                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()

            # 6. Memory Update (Weighted Lehmer Mean)
            if len(succ_f) > 0:
                succ_f = np.array(succ_f)
                succ_cr = np.array(succ_cr)
                diffs = np.array(diff_fitness)
                
                # Weights based on improvement magnitude
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    weights = diffs / total_diff
                    
                    # Weighted Lehmer Mean for F
                    mean_f = np.sum(weights * (succ_f ** 2)) / np.sum(weights * succ_f)
                    
                    # Weighted Mean for CR
                    mean_cr = np.sum(weights * succ_cr)
                    
                    # Update Memory (Smooth update to be robust)
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                    
                    k_mem = (k_mem + 1) % H

            # 7. Convergence Check
            # If population fitness variance is negligible, restart
            if np.std(fitness) < 1e-8:
                break

    return best_fitness
