#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 55.49079297264958
#2. output value is: 14.192092790710518
#3. output value is: 1.3677205408857276
#
#The best previously generated algorithm was **L-SHADE-jSO with Reflection Boundary Handling and Soft Restart** (Attempt 3), which achieved the lowest output value (1.36). This result confirms that the core L-SHADE-jSO logic combined with reflection boundaries is highly effective. However, the use of a simple Gaussian random walk for local polishing was likely too inefficient to fully exploit the basin of attraction, preventing the algorithm from reaching the absolute minimum (0.0).
#
#The following algorithm improves upon Attempt 3 by replacing the random Gaussian polish with a structured **Coordinate Descent Local Search (MTS-LS1)**. This local search is deterministic and adaptive, allowing for precise refinement of the solution dimension by dimension. Additionally, the initialization is upgraded to **Latin Hypercube Sampling (LHS)** to ensure better initial coverage, and the **Archive size** is increased to store more historical diversity.
#
##### Key Improvements:
#1.  **LHS Initialization**: Replaces random uniform sampling with Latin Hypercube Sampling (stratified) to guarantee that the initial population covers the search space more evenly.
#2.  **MTS-LS1 Local Search**: Instead of random perturbations, the algorithm uses a modified Coordinate Descent method (MTS-LS1) when stagnation is detected or in the final phase of optimization. This systematically adjusts each dimension using adaptive step sizes to descend to the local minimum efficiently.
#3.  **Adaptive Restart Strategy**: If stagnation occurs early (`progress < 0.8`), a soft restart is triggered (keeping the elite solution). If stagnation occurs late or the time is nearly up (`progress > 0.95`), the budget is dedicated to the MTS-LS1 local search to aggressively polish the best solution.
#4.  **Refined L-SHADE-jSO Core**:
#    *   **Reflection Boundary Handling**: Essential for maintaining population distribution.
#    *   **jSO Constraints**: Linearly decreasing `p` (greediness) and clamping `F` in early stages.
#    *   **Weighted Lehmer Mean**: For memory updates.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using L-SHADE-jSO with Latin Hypercube Sampling,
    Reflection Boundary Handling, and MTS-LS1 Local Search.
    """
    start_time = datetime.now()
    # Reserve a 2% time buffer to ensure we return a result before the strict timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    def check_timeout():
        return datetime.now() >= end_time

    # ---------------- Parameters ----------------
    # Population Sizing: Linear Population Reduction (LPR)
    min_pop_size = 4
    # Heuristic: ~25 * dim for robust initialization, capped at 250
    init_pop_size = int(max(min_pop_size, min(25 * dim, 250)))
    pop_size = init_pop_size
    
    # SHADE Parameters
    H = 5             # Memory size (History length)
    arc_rate = 2.0    # Archive size relative to population (increased for diversity)
    
    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # SHADE Memory Init
    # M_CR starts high (0.8) to encourage diversity
    # M_F starts at 0.5 (neutral)
    m_cr = np.full(H, 0.8)
    m_f = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
    archive = np.empty((0, dim))
    
    # Latin Hypercube Sampling (LHS) Initialization
    # Ensures stratified coverage of each dimension
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        samples = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(samples)
        population[:, d] = samples
        
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if i % 10 == 0 and check_timeout():
            return best_fitness if best_fitness != float('inf') else func(population[0])
            
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # MTS-LS1 Local Search State
    # Maintain adaptive step sizes for each dimension
    ls_step = diff_b * 0.05  # Start with 5% of domain width
    
    # ---------------- Main Optimization Loop ----------------
    while not check_timeout():
        # Calculate Progress
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        
        # 1. Linear Population Reduction (LPR)
        plan_pop_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)
        
        if pop_size > plan_pop_size:
            # Sort by fitness and truncate weakest
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:plan_pop_size]]
            fitness = fitness[sorted_idx[:plan_pop_size]]
            
            # Resize Archive
            curr_arc_cap = int(plan_pop_size * arc_rate)
            if len(archive) > curr_arc_cap:
                np.random.shuffle(archive)
                archive = archive[:curr_arc_cap]
                
            pop_size = plan_pop_size
            
        # 2. Stagnation Detection & Local Search
        fit_std = np.std(fitness)
        # Conditions: Population has converged (low std) OR Final 5% of time
        stagnated = fit_std < 1e-8
        end_game = progress > 0.95
        
        if stagnated or end_game:
            # --- MTS-LS1 Coordinate Descent ---
            # Apply local search to the elite solution to refine precision
            improved_ls = False
            curr_x = best_sol.copy()
            curr_f = best_fitness
            
            # Iterate through all dimensions
            for d in range(dim):
                if check_timeout(): return best_fitness
                
                # Enforce minimum step size to prevent underflow
                min_step = 1e-15 * max(1.0, abs(curr_x[d]))
                if ls_step[d] < min_step: ls_step[d] = min_step * 2
                
                # Try Negative Step
                x_new = curr_x.copy()
                x_new[d] = np.clip(curr_x[d] - ls_step[d], min_b[d], max_b[d])
                f_new = func(x_new)
                
                if f_new < curr_f:
                    curr_f = f_new
                    curr_x = x_new.copy()
                    ls_step[d] *= 1.5 # Expansion on success
                    improved_ls = True
                else:
                    # Try Positive Step
                    x_new[d] = np.clip(curr_x[d] + ls_step[d], min_b[d], max_b[d])
                    f_new = func(x_new)
                    
                    if f_new < curr_f:
                        curr_f = f_new
                        curr_x = x_new.copy()
                        ls_step[d] *= 1.5 # Expansion on success
                        improved_ls = True
                    else:
                        # Contraction on failure
                        ls_step[d] *= 0.5
                        
            if improved_ls:
                best_fitness = curr_f
                best_sol = curr_x.copy()
                # Update the best individual in the population to guide DE
                best_idx = np.argmin(fitness)
                population[best_idx] = best_sol
                fitness[best_idx] = best_fitness
            
            # --- Soft Restart Strategy ---
            # If stagnated early (before 80% time), restart population to escape local optimum.
            # Don't restart in 'end_game' phase (focus on exploitation).
            if stagnated and not end_game and progress < 0.8:
                # Keep elite (already at population[best_idx]), regenerate others
                # Random generation is sufficient for restart
                n_regen = pop_size - 1
                if n_regen > 0:
                    new_pop = min_b + np.random.rand(n_regen, dim) * diff_b
                    
                    # Sort to put elite at index 0
                    idx = np.argsort(fitness)
                    population = population[idx]
                    fitness = fitness[idx]
                    
                    # Replace all but top 1
                    population[1:] = new_pop
                    fitness[1:] = np.full(n_regen, float('inf'))
                    
                    # Reset SHADE Memory and Archive
                    m_cr.fill(0.8)
                    m_f.fill(0.5)
                    k_mem = 0
                    archive = np.empty((0, dim))
                    
                    # Reset LS step sizes
                    ls_step = diff_b * 0.05
                    
                    # Evaluate new individuals
                    for i in range(1, pop_size):
                        if check_timeout(): return best_fitness
                        val = func(population[i])
                        fitness[i] = val
                        if val < best_fitness:
                            best_fitness = val
                            best_sol = population[i].copy()
                    
                    continue # Skip standard DE evolution this iteration
        
        # 3. Parameter Generation (L-SHADE-jSO)
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        # Regenerate non-positive F
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = m_f[r_idx[bad_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # jSO Constraint: Clamp F to 0.7 during first 60% of search
        if progress < 0.6:
            f = np.minimum(f, 0.7)
            
        # 4. Mutation: current-to-pbest/1
        # p decreases linearly from 0.25 (exploration) to 0.05 (exploitation)
        p_val = 0.25 - (0.20 * progress)
        p_val = max(0.05, p_val)
        p_count = max(2, int(pop_size * p_val))
        
        # Select p-best
        sorted_indices = np.argsort(fitness)
        pbest_pool = sorted_indices[:p_count]
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        x_pbest = population[pbest_idx]
        
        # Select r1 (random from pop)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        # Fast self-exclusion check: if r1==i, shift to i+1 (wrapping)
        # Note: In Python/Numpy this is a heuristic; simpler to accept occasional collision
        x_r1 = population[r1_idx]
        
        # Select r2 (random from Union(Population, Archive))
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        # Compute Mutant
        f_v = f[:, None]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # Boundary Handling: Reflection
        # Reflect back into bounds: if x < min, new = 2*min - x
        mutant = np.where(mutant < min_b, 2 * min_b - mutant, mutant)
        mutant = np.where(mutant > max_b, 2 * max_b - mutant, mutant)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # 6. Selection & Memory Update
        succ_cr = []
        succ_f = []
        succ_diff = []
        new_archive_cands = []
        
        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                diff = fitness[i] - f_trial
                
                # Record success
                succ_cr.append(cr[i])
                succ_f.append(f[i])
                succ_diff.append(diff)
                new_archive_cands.append(population[i].copy())
                
                # Update Population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    
        # Update Archive
        if new_archive_cands:
            cands = np.array(new_archive_cands)
            if len(archive) == 0:
                archive = cands
            else:
                archive = np.vstack((archive, cands))
            
            # Limit archive size
            cap = int(pop_size * arc_rate)
            if len(archive) > cap:
                np.random.shuffle(archive)
                archive = archive[:cap]
                
        # Update SHADE Memory (Weighted Lehmer Mean)
        if succ_diff:
            wdiff = np.array(succ_diff)
            total_diff = np.sum(wdiff)
            
            if total_diff > 0:
                weights = wdiff / total_diff
                
                # Update M_CR
                m_scr = np.sum(weights * np.array(succ_cr))
                m_cr[k_mem] = m_scr
                
                # Update M_F
                sf_arr = np.array(succ_f)
                denom = np.sum(weights * sf_arr)
                if denom > 1e-12:
                    m_f[k_mem] = np.sum(weights * (sf_arr ** 2)) / denom
                else:
                    m_f[k_mem] = 0.5
                    
                k_mem = (k_mem + 1) % H
                
    return best_fitness
