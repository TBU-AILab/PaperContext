#The best previously generated algorithm was **L-SHADE-jSO with Reflection Boundary Handling** (Attempt 2), which achieved a significantly lower output value (1.36) compared to other attempts (14.19 and 5.56). This indicates that the core mechanism of L-SHADE-jSO (Linear Population Reduction, Weighted Lehmer Mean, jSO parameter constraints) combined with Reflection Boundary Handling is the most effective approach for this problem.
#
#The following algorithm improves upon Attempt 2 by introducing **Opposition-Based Initialization**, a **Coordinate Descent (MTS-LS1) Local Search**, and refined **Stagnation Detection**.
#
#### Key Improvements:
#
#1.  **Opposition-Based Initialization (OBL)**: Instead of starting with just random samples, the algorithm generates a set of random individuals and their *opposite* counterparts within the bounds ($x' = a + b - x$). The best $N$ individuals from this combined pool of $2N$ are selected. This provides a much higher quality initial population and covers the search space corners more effectively.
#2.  **MTS-LS1 Local Search**: When the population stagnates (low variance) or before a restart, a lightweight **Coordinate Descent** is applied to the elite solution. Unlike the random Gaussian perturbation in Attempt 2, this method systematically explores each dimension with adaptive step sizes, efficiently refining the solution to the bottom of the local basin.
#3.  **L-SHADE-jSO Core**:
#    *   **Linear Population Reduction (LPR)**: Reduces population size linearly to shift from exploration to exploitation.
#    *   **Weighted Lehmer Mean**: Memory updates prioritize parameters that produce larger fitness gains.
#    *   **jSO Constraints**: Mutation factor $F$ is clamped at 0.7 during the first 60% of evaluations to prevent early chaotic divergence.
#4.  **Reflection Boundary Handling**: Preserves population diversity better than clipping by bouncing particles back into the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using L-SHADE-jSO with Opposition-Based Initialization,
    Coordinate Descent Local Search (MTS-LS1), and Linear Population Reduction.
    """
    start_time = datetime.now()
    # Reserve a 5% time buffer to ensure safe return before strict timeout
    end_time = start_time + timedelta(seconds=max_time * 0.95)

    def check_timeout():
        return datetime.now() >= end_time

    # ---------------- Parameters ----------------
    min_pop_size = 4
    # Initial population size: heuristic ~18*dim, clamped between min and 200
    # Slightly smaller than 25*dim to allow for OBL overhead and more generations
    init_pop_size = int(max(min_pop_size, min(18 * dim, 200)))
    pop_size = init_pop_size
    
    # SHADE Memory Parameters
    H = 6  # Memory size
    arc_rate = 2.6 # Archive capacity relative to population size
    
    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Memory Initialization
    # M_CR starts high (0.8) to encourage component mixing
    # M_F starts at 0.5 (neutral)
    m_cr = np.full(H, 0.8)
    m_f = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
    archive = np.empty((0, dim))
    
    # --- Opposition-Based Learning (OBL) Initialization ---
    # 1. Generate Random Population
    p_rand = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # 2. Generate Opposite Population
    # x' = min + max - x
    p_opp = min_b + max_b - p_rand
    p_opp = np.clip(p_opp, min_b, max_b)
    
    # 3. Evaluate Combined Pool (2 * pop_size)
    all_pop = np.vstack((p_rand, p_opp))
    all_fit = np.full(2 * pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = np.zeros(dim)
    
    for i in range(2 * pop_size):
        if check_timeout():
            # If timeout immediately, return best found or evaluate one
            if best_fitness == float('inf'):
                return func(all_pop[0])
            return best_fitness
            
        val = func(all_pop[i])
        all_fit[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = all_pop[i].copy()
            
    # 4. Select N best individuals
    sorted_idx = np.argsort(all_fit)
    population = all_pop[sorted_idx[:pop_size]]
    fitness = all_fit[sorted_idx[:pop_size]]
    
    # ---------------- Main Optimization Loop ----------------
    while not check_timeout():
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        
        # 1. Linear Population Reduction (LPR)
        # Smoothly reduce population size from init_pop_size to min_pop_size
        plan_pop_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)
        
        if pop_size > plan_pop_size:
            # Sort and truncate population
            s_idx = np.argsort(fitness)
            population = population[s_idx[:plan_pop_size]]
            fitness = fitness[s_idx[:plan_pop_size]]
            
            # Resize Archive
            current_arc_cap = int(plan_pop_size * arc_rate)
            if len(archive) > current_arc_cap:
                np.random.shuffle(archive)
                archive = archive[:current_arc_cap]
            
            pop_size = plan_pop_size
            
        # 2. Stagnation Detection -> Local Search -> Restart
        # If fitness standard deviation is very low, the population has converged.
        fit_std = np.std(fitness)
        if fit_std < 1e-10:
            # --- Phase A: Coordinate Descent Local Search (MTS-LS1 style) ---
            # Try to refine the elite solution dimension by dimension
            
            # Budget for local search: proportional to dim, capped at 100
            ls_limit = max(20, min(100, 5 * dim))
            ls_evals = 0
            
            # Initial step size relative to domain
            step_sz = diff_b * 0.005
            
            improved_ls = True
            while improved_ls and ls_evals < ls_limit:
                improved_ls = False
                for d in range(dim):
                    if check_timeout(): return best_fitness
                    if ls_evals >= ls_limit: break
                    
                    # Try Negative Step
                    x_temp = best_sol.copy()
                    x_temp[d] = np.clip(best_sol[d] - step_sz[d], min_b[d], max_b[d])
                    val = func(x_temp)
                    ls_evals += 1
                    
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = x_temp.copy()
                        population[0] = x_temp # Update elite in pop
                        fitness[0] = val
                        improved_ls = True
                    else:
                        # Try Positive Step
                        x_temp[d] = np.clip(best_sol[d] + step_sz[d], min_b[d], max_b[d])
                        val = func(x_temp)
                        ls_evals += 1
                        
                        if val < best_fitness:
                            best_fitness = val
                            best_sol = x_temp.copy()
                            population[0] = x_temp
                            fitness[0] = val
                            improved_ls = True
                
                # Adaptation of step size
                if not improved_ls:
                    step_sz *= 0.5
                    # Continue if step size is still significant
                    if np.max(step_sz) > 1e-15:
                        improved_ls = True
            
            # --- Phase B: Soft Restart ---
            # If we still have time (progress < 90%), restart to find new basins
            if progress < 0.9:
                # Keep elite (index 0), regenerate others
                n_regen = pop_size - 1
                if n_regen > 0:
                    # Generate random candidates
                    new_pop = min_b + np.random.rand(n_regen, dim) * diff_b
                    
                    for i in range(n_regen):
                        if check_timeout(): return best_fitness
                        val = func(new_pop[i])
                        population[i+1] = new_pop[i]
                        fitness[i+1] = val
                        if val < best_fitness:
                            best_fitness = val
                            best_sol = new_pop[i].copy()
                    
                    # Ensure elite is at index 0
                    population[0] = best_sol.copy()
                    fitness[0] = best_fitness
                    
                    # Reset SHADE Memory & Archive to clear bias
                    m_cr = np.full(H, 0.5)
                    m_f = np.full(H, 0.5)
                    archive = np.empty((0, dim))
                    k_mem = 0
                    
                    continue # Skip standard DE evolution this iteration

        # 3. Parameter Generation
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        # Regenerate F <= 0
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = m_f[r_idx[bad_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # jSO Constraint: Clamp F to 0.7 during first 60% of search
        if progress < 0.6:
            f = np.minimum(f, 0.7)
            
        # 4. Mutation: current-to-pbest/1
        # p linearly decreases from 0.2 to 0.05 to increase exploitation pressure
        p_val = max(0.05, 0.2 - 0.15 * progress)
        p_count = max(2, int(pop_size * p_val))
        
        # Select p-best
        sorted_indices = np.argsort(fitness)
        pbest_pool = sorted_indices[:p_count]
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        x_pbest = population[pbest_idx]
        
        # Select r1 (random from pop)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_idx]
        
        # Select r2 (Union of Pop and Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        # Compute Mutant
        f_v = f[:, None]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # Reflection Boundary Handling
        # If out of bounds, reflect back: lower - (val - lower) -> 2*lower - val
        mutant = np.where(mutant < min_b, 2 * min_b - mutant, mutant)
        mutant = np.where(mutant > max_b, 2 * max_b - mutant, mutant)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # 6. Selection and Memory Update
        succ_cr = []
        succ_f = []
        succ_diff = []
        new_archive_candidates = []
        
        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Successful update
                diff = fitness[i] - f_trial
                succ_cr.append(cr[i])
                succ_f.append(f[i])
                succ_diff.append(diff)
                new_archive_candidates.append(population[i].copy())
                
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    
        # 7. Update Memory (Weighted Lehmer Mean)
        if succ_diff:
            weights = np.array(succ_diff)
            total_diff = np.sum(weights)
            
            if total_diff > 0:
                weights /= total_diff
                
                # M_CR Update
                m_scr = np.sum(weights * np.array(succ_cr))
                m_cr[k_mem] = m_scr
                
                # M_F Update
                sf_arr = np.array(succ_f)
                denom = np.sum(weights * sf_arr)
                if denom > 1e-12:
                    m_f[k_mem] = np.sum(weights * (sf_arr ** 2)) / denom
                else:
                    m_f[k_mem] = 0.5
                    
                k_mem = (k_mem + 1) % H
                
        # 8. Update Archive
        if new_archive_candidates:
            cands = np.array(new_archive_candidates)
            if len(archive) == 0:
                archive = cands
            else:
                archive = np.vstack((archive, cands))
            
            cap = int(pop_size * arc_rate)
            if len(archive) > cap:
                np.random.shuffle(archive)
                archive = archive[:cap]
                
    return best_fitness
