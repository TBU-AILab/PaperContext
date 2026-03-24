#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 5.7619078205043195
#2. output value is: 14.848924371508517
#3. output value is: 30.274162611025417
#
#The best generated algorithm code was the JADE-like variant with an external archive. The following algorithm improves upon it by implementing **L-SHADE (Linear Success-History Adaptive Differential Evolution)**.
#
#**Key Improvements:**
#1.  **Linear Population Reduction (LPR)**: The population size starts large to maximize exploration and linearly decreases over time (based on `max_time`) to focus on exploitation. This manages the computational budget far more effectively than a fixed population size.
#2.  **Success-History Adaptation**: Instead of a single learning rate, the algorithm maintains a history memory ($H$) of successful control parameters ($F$ and $CR$). This allows the algorithm to recall and adapt to multimodal landscapes where different phases of search require different behaviors.
#3.  **Weighted Lehmer Mean**: The updates to the history memory are weighted by the fitness improvement amount, giving higher priority to parameter values that generate significant progress.
#4.  **Integrated Restart with LPR**: A stagnation check is included. If the population converges prematurely, it triggers a restart. Crucially, the restart respects the LPR schedule, re-initializing only the number of individuals allowed by the current time progress.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using L-SHADE (Linear Success-History Adaptive 
    Differential Evolution) with time-based population reduction.
    """
    start_time = datetime.now()
    # Safety margin to ensure clean return
    time_limit = timedelta(seconds=max_time * 0.99)

    # ---------------- Parameters ----------------
    # Population Sizing (Linear Reduction)
    # Start large for exploration, end small for convergence.
    # Heuristic: init ~ 18*dim, but capped for time safety.
    min_pop_size = 4
    init_pop_size = int(max(min_pop_size, min(18 * dim, 150)))
    
    # SHADE Memory Size
    H = 6
    
    # Greediness for p-best mutation
    p_best_rate = 0.11
    
    # Archive size factor (relative to current population size)
    arc_rate = 1.4

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # Initialize Memory for Adaptive Parameters (F and CR)
    # Storing mean values. Init at 0.5.
    m_cr = np.full(H, 0.5)
    m_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer

    # Initialize Population (Latin Hypercube Sampling for coverage)
    pop_size = init_pop_size
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        population[:, d] = (np.random.permutation(pop_size) + np.random.rand(pop_size)) / pop_size
    population = min_b + population * diff_b

    # Initial Evaluation
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_sol = np.zeros(dim)

    # Evaluate initial population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # External Archive (stores replaced superior solutions)
    archive = np.empty((0, dim))

    # ---------------- Main Optimization Loop ----------------
    while True:
        # 1. Time Check & Progress Calculation
        current_time = datetime.now()
        elapsed = current_time - start_time
        if elapsed >= time_limit:
            return best_fitness
            
        # Progress ratio (0.0 to 1.0)
        progress = elapsed.total_seconds() / max_time
        
        # 2. Linear Population Reduction (LPR)
        # Calculate target population size based on remaining time
        # Formula: N_t = round( (N_min - N_init) * progress + N_init )
        target_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        target_size = max(min_pop_size, target_size)
        
        if target_size < pop_size:
            # Reduce Population: Keep the best 'target_size' individuals
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices[:target_size]]
            fitness = fitness[sorted_indices[:target_size]]
            
            # Resize Archive to match new population limit
            current_archive_cap = int(target_size * arc_rate)
            if len(archive) > current_archive_cap:
                # Randomly remove excess from archive
                keep_idxs = np.random.choice(len(archive), current_archive_cap, replace=False)
                archive = archive[keep_idxs]
                
            pop_size = target_size

        # 3. Stagnation Check & Soft Restart
        # If population variance is near zero, we are stuck.
        # Restart by randomizing the population (keeping elite), respecting current LPR size.
        fit_spread = np.max(fitness) - np.min(fitness)
        if fit_spread < 1e-9:
            # Generate new random population
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_sol.copy() # Preserve global best
            
            # Reset Memory/Archive
            m_cr = np.full(H, 0.5)
            m_f = np.full(H, 0.5)
            archive = np.empty((0, dim))
            
            # Re-evaluate
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit: return best_fitness
                fitness[i] = func(population[i])
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_sol = population[i].copy()
            continue

        # 4. Generate Parameters (SHADE strategy)
        # Pick a random memory slot for each individual
        r_indices = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr[r_indices], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        # Location + Scale * tan(pi * (rand - 0.5))
        f = m_f[r_indices] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # F constraint handling: if F <= 0, regenerate. if F > 1, clip to 1.
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = m_f[r_indices[bad_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)

        # 5. Mutation: DE/current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Sort to find p-best
        sorted_indices = np.argsort(fitness)
        num_p = max(2, int(pop_size * p_best_rate))
        top_indices = sorted_indices[:num_p]
        
        # Select X_pbest
        pbest_idxs = np.random.choice(top_indices, pop_size)
        x_pbest = population[pbest_idxs]
        
        # Select X_r1 (from population)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_idxs]
        
        # Select X_r2 (from Population U Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
            
        r2_idxs = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idxs]
        
        # Compute Mutant
        f_col = f[:, None]
        mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        mutant = np.clip(mutant, min_b, max_b)

        # 6. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        # Guarantee at least one dimension changes
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)

        # 7. Selection & Success Recording
        improvement_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        new_archive_candidates = []

        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit: return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Successful trial
                diff = fitness[i] - f_trial
                
                # Store parent in candidates for archive
                new_archive_candidates.append(population[i].copy())
                
                # Update population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                improvement_mask[i] = True
                diff_fitness[i] = diff
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
        
        # 8. Update Archive
        if new_archive_candidates:
            candidates = np.array(new_archive_candidates)
            if len(archive) == 0:
                archive = candidates
            else:
                archive = np.vstack((archive, candidates))
                
            # Cap archive size
            current_cap = int(pop_size * arc_rate)
            if len(archive) > current_cap:
                idxs = np.random.choice(len(archive), current_cap, replace=False)
                archive = archive[idxs]

        # 9. Update History Memory (SHADE)
        # Weighted Lehmer Mean based on fitness improvement
        if np.any(improvement_mask):
            succ_f = f[improvement_mask]
            succ_cr = cr[improvement_mask]
            succ_diff = diff_fitness[improvement_mask]
            
            total_diff = np.sum(succ_diff)
            
            # Calculate weights
            if total_diff > 0:
                weights = succ_diff / total_diff
            else:
                weights = np.ones(len(succ_diff)) / len(succ_diff)
            
            # Update M_CR (Weighted Mean)
            m_cr[k_mem] = np.sum(weights * succ_cr)
            
            # Update M_F (Weighted Lehmer Mean)
            sum_wf = np.sum(weights * succ_f)
            if sum_wf > 1e-12:
                m_f[k_mem] = np.sum(weights * (succ_f ** 2)) / sum_wf
            else:
                m_f[k_mem] = 0.5
            
            # Move memory index
            k_mem = (k_mem + 1) % H

    return best_fitness
