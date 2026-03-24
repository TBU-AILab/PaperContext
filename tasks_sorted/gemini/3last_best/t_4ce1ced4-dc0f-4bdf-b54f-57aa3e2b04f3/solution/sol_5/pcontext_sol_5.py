#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 51.45197113599825
#2. output value is: 5.7619078205043195
#3. output value is: 14.848924371508517
#
#The best generated algorithm was the **JADE with External Archive** (Attempt 2). L-SHADE (Attempt 1) performed poorly, likely due to aggressive population reduction or implementation overheads in a short time window.
#
#The following algorithm implements a robust **L-SHADE (Linear Success-History Adaptive Differential Evolution) with Reflection Boundary Handling**. This implementation improves upon the previous best by:
#1.  **L-SHADE Architecture**: Incorporates the superior **Success-History Adaptation** (memory of successful parameters) and **Linear Population Reduction (LPR)**. LPR is tuned more conservatively to prevent premature diversity loss.
#2.  **Reflection Boundary Handling**: Instead of simple clipping (which biases search to boundaries), this uses reflection to bounce particles back into the search space, maintaining the search distribution.
#3.  **Vectorized Operations**: The generation of mutant vectors, crossover, and parameter sampling is fully vectorized for maximum throughput in Python.
#4.  **Soft Restart with Archive Reset**: Detects convergence and restarts the population (preserving the elite) while clearing the history, ensuring the algorithm can escape local optima effectively.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using L-SHADE (Linear Success-History Adaptive DE)
    with Reflection Boundary Handling and Linear Population Reduction.
    """
    start_time = datetime.now()
    # Use 95% of the max_time to ensure we return the result safely before timeout
    end_time = start_time + timedelta(seconds=max_time * 0.95)

    # ---------------- Parameters ----------------
    # Population Sizing (LPR)
    # Start with a sufficiently large population for exploration
    min_pop_size = 5
    init_pop_size = int(max(min_pop_size, min(25 * dim, 250)))
    pop_size = init_pop_size

    # Memory Size (History)
    # H = dim is a common heuristic for SHADE
    H = max(5, dim)
    
    # Archive Size Rate
    arc_rate = 2.0

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # SHADE Memory Initialization (0.5 start)
    m_cr = np.full(H, 0.5)
    m_f = np.full(H, 0.5)
    k_mem = 0

    # Latin Hypercube Sampling (LHS) Initialization
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        population[:, d] = (np.random.permutation(pop_size) + np.random.rand(pop_size)) / pop_size
    population = min_b + population * diff_b

    # External Archive
    archive = np.empty((0, dim))

    # Initial Evaluation
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_sol = np.zeros(dim)

    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_fitness if best_fitness != float('inf') else func(population[0])
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # ---------------- Main Optimization Loop ----------------
    while True:
        current_time = datetime.now()
        if current_time >= end_time:
            return best_fitness

        # Calculate Progress (0.0 -> 1.0)
        progress = (current_time - start_time).total_seconds() / max_time
        
        # 1. Linear Population Reduction (LPR)
        # Smoothly reduce population size from init_pop_size to min_pop_size
        plan_pop_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)

        if pop_size > plan_pop_size:
            # Reduce population: keep best individuals
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices[:plan_pop_size]]
            fitness = fitness[sorted_indices[:plan_pop_size]]
            pop_size = plan_pop_size

            # Resize Archive to match new population limit
            max_arc_size = int(pop_size * arc_rate)
            if len(archive) > max_arc_size:
                np.random.shuffle(archive)
                archive = archive[:max_arc_size]

        # 2. Restart Mechanism (Stagnation Check)
        # If fitness variance is negligible, restart search but keep elite
        if (np.max(fitness) - np.min(fitness)) < 1e-8:
            # Re-initialize population using LHS (except elite)
            new_pop = np.zeros((pop_size, dim))
            for d in range(dim):
                new_pop[:, d] = (np.random.permutation(pop_size) + np.random.rand(pop_size)) / pop_size
            population = min_b + new_pop * diff_b
            
            # Restore elite solution at index 0
            population[0] = best_sol.copy()
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            # Reset Memory and Archive to allow fresh adaptation
            m_cr = np.full(H, 0.5)
            m_f = np.full(H, 0.5)
            archive = np.empty((0, dim))
            
            # Evaluate new population
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_fitness
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # 3. Parameter Generation (SHADE Strategy)
        # Randomly select memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        
        # Generate CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(M_F, 0.1)
        f = m_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Constraint handling for F (must be > 0)
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            # Regenerate bad values
            f[bad_mask] = m_f[r_idx[bad_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)

        # 4. Mutation: current-to-pbest/1
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # Dynamic p value (decreases over time to encourage exploitation)
        p_val = max(0.05, 0.2 * (1 - progress))
        p_count = max(2, int(pop_size * p_val))
        top_indices = sorted_indices[:p_count]
        
        # Select x_pbest
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = population[pbest_indices]
        
        # Select x_r1 (from population)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # Select x_r2 (from Population U Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutant Vectors
        f_col = f[:, None]
        mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        
        # 5. Boundary Handling: Reflection
        # If value exceeds bounds, reflect it back inside
        mutant = np.where(mutant < min_b, 2 * min_b - mutant, mutant)
        mutant = np.where(mutant > max_b, 2 * max_b - mutant, mutant)
        # Safety clip in case reflection is still out
        mutant = np.clip(mutant, min_b, max_b)

        # 6. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)

        # 7. Selection and Memory Update
        succ_scr = []
        succ_sf = []
        succ_diff = []
        new_archive_candidates = []

        for i in range(pop_size):
            if datetime.now() >= end_time: return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Improvement or equal
                diff = fitness[i] - f_trial
                
                # Store successful parameters and parent for archive
                succ_scr.append(cr[i])
                succ_sf.append(f[i])
                succ_diff.append(diff)
                new_archive_candidates.append(population[i].copy())
                
                # Update Population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()

        # 8. Update Archive
        if new_archive_candidates:
            cand = np.array(new_archive_candidates)
            if len(archive) == 0:
                archive = cand
            else:
                archive = np.vstack((archive, cand))
            
            # Cap archive size
            current_cap = int(pop_size * arc_rate)
            if len(archive) > current_cap:
                np.random.shuffle(archive)
                archive = archive[:current_cap]

        # 9. Update SHADE Memory (Weighted Lehmer Mean)
        if succ_diff:
            diffs = np.array(succ_diff)
            total_diff = np.sum(diffs)
            
            if total_diff > 0:
                weights = diffs / total_diff
                
                # Update M_CR (Weighted Mean)
                m_scr = np.sum(weights * np.array(succ_scr))
                m_cr[k_mem] = m_scr
                
                # Update M_F (Weighted Lehmer Mean)
                sf_arr = np.array(succ_sf)
                mean_lehmer = np.sum(weights * (sf_arr ** 2)) / np.sum(weights * sf_arr)
                m_f[k_mem] = mean_lehmer
                
                # Advance memory pointer
                k_mem = (k_mem + 1) % H

    return best_fitness
