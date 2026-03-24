#The best generated algorithm in the previous attempts was **L-SHADE with Reflection Boundary Handling**. The following algorithm builds upon it by incorporating key features from the **jSO (iL-SHADE)** algorithm, which is a highly successful variant of L-SHADE.
#
#**Key Improvements:**
#1.  **Refined Parameter Adaptation**:
#    *   **P-value adaptation**: The 'greediness' of the mutation strategy (`current-to-pbest`) now scales linearly from $0.25$ down to $0.05$. This promotes exploration early on and strong exploitation in later stages.
#    *   **jSO F-Clamp**: For the first 60% of the optimization, the mutation factor $F$ is capped at 0.7. This prevents chaotic divergence in the early phase, a common issue in high-dimensional DE.
#2.  **Weighted Memory Updates**: The history memory for $F$ and $CR$ is updated using a **Weighted Lehmer Mean** based on fitness improvements, prioritizing parameters that yield larger gains.
#3.  **Hybrid Restart with Local Polishing**:
#    *   **Stagnation Detection**: Checks if the population variance drops below a threshold.
#    *   **Local Polish**: Before restarting, the algorithm attempts a quick local search (Gaussian perturbations) around the best solution to squeeze out maximum precision from the current basin.
#    *   **Soft Restart**: If stagnation persists, the population is re-initialized (preserving the elite) and memory is reset to allow adaptation to new landscapes.
#4.  **Robust Time Management**: Evaluation loops check the timeout frequently to ensures the result is returned exactly within the allocated budget.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using an improved L-SHADE-jSO algorithm
    (Linear Success-History Adaptive DE) with local polishing and reflection.
    """
    start_time = datetime.now()
    # Reserve a small buffer to ensure clean return before hard timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    def check_timeout():
        return datetime.now() >= end_time

    # ---------------- Parameters ----------------
    # Population Sizing: Linear Population Reduction (LPR)
    # Start with a healthy population size for exploration, reduce for efficiency.
    min_pop_size = 4
    # Heuristic: init ~ 25*dim, but capped to ensure speed
    init_pop_size = int(max(min_pop_size, min(25 * dim, 250)))
    pop_size = init_pop_size

    # Memory Size (History length)
    H = 5
    
    # Archive Size Rate (Capacity relative to population)
    arc_rate = 1.0

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # SHADE Memory Initialization
    # M_CR starts high (0.8) to encourage diversity, M_F at 0.5
    m_cr = np.full(H, 0.8)
    m_f = np.full(H, 0.5)
    k_mem = 0

    # Latin Hypercube Sampling (LHS) for uniform initial coverage
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
        # Check timeout periodically inside initialization
        if i % 10 == 0 and check_timeout():
            return best_fitness if best_fitness != float('inf') else func(population[0])
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # ---------------- Main Optimization Loop ----------------
    while True:
        if check_timeout():
            return best_fitness

        # Calculate Progress (0.0 -> 1.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        
        # 1. Linear Population Reduction (LPR)
        # Smoothly reduce population size to focus computational budget on best individuals
        plan_pop_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)

        if pop_size > plan_pop_size:
            # Sort by fitness and truncate
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx[:plan_pop_size]]
            fitness = fitness[sorted_idx[:plan_pop_size]]
            
            # Resize Archive
            curr_arc_cap = int(plan_pop_size * arc_rate)
            if len(archive) > curr_arc_cap:
                np.random.shuffle(archive)
                archive = archive[:curr_arc_cap]
            
            pop_size = plan_pop_size

        # 2. Restart & Stagnation Handling
        # If population variance is negligible, we are likely stuck in a local optimum.
        fit_std = np.std(fitness)
        if fit_std < 1e-12:
            # --- Phase A: Local Polish (Exploitation) ---
            # Before giving up, try small Gaussian perturbations around the best solution
            # to see if we can refine the minimum further.
            polish_steps = 15
            for _ in range(polish_steps):
                if check_timeout(): return best_fitness
                # Generate neighbor with small variance proportional to domain
                neighbor = best_sol + np.random.normal(0, 1e-5 * diff_b, dim)
                neighbor = np.clip(neighbor, min_b, max_b)
                val = func(neighbor)
                if val < best_fitness:
                    best_fitness = val
                    best_sol = neighbor.copy()
            
            # --- Phase B: Soft Restart (Exploration) ---
            # If still early enough, restart the population to find other basins.
            if progress < 0.85:
                # Generate new random population
                new_pop = np.zeros((pop_size, dim))
                for d in range(dim):
                    new_pop[:, d] = (np.random.permutation(pop_size) + np.random.rand(pop_size)) / pop_size
                population = min_b + new_pop * diff_b
                
                # Keep the elite solution
                population[0] = best_sol.copy()
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_fitness
                
                # Reset Memory & Archive to remove bias from previous local optimum
                m_cr = np.full(H, 0.8)
                m_f = np.full(H, 0.5)
                archive = np.empty((0, dim))
                
                # Evaluate new population (skipping elite)
                for i in range(1, pop_size):
                    if i % 10 == 0 and check_timeout(): return best_fitness
                    val = func(population[i])
                    fitness[i] = val
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = population[i].copy()
                continue
        
        # 3. Parameter Generation
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # F constraints: positive and <= 1. Regenerate <= 0.
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = m_f[r_idx[bad_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # jSO Feature: Clamp F to 0.7 during first 60% of search to prevent early divergence
        if progress < 0.6:
            f = np.minimum(f, 0.7)

        # 4. Mutation: current-to-pbest/1
        # p decreases linearly from 0.25 (exploration) to 0.05 (exploitation)
        p_curr = 0.25 - (0.20) * progress
        p_count = max(2, int(pop_size * p_curr))
        
        sorted_idx = np.argsort(fitness)
        top_p_idx = sorted_idx[:p_count]
        
        # Select x_pbest
        pbest_idx = np.random.choice(top_p_idx, pop_size)
        x_pbest = population[pbest_idx]
        
        # Select x_r1
        r1_idx = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_idx]
        
        # Select x_r2 from Union(Population, Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        # Compute Mutant Vector V
        f_v = f[:, None]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # 5. Boundary Handling: Reflection
        # Bouncing back from boundaries preserves population distribution better than clipping
        mutant = np.where(mutant < min_b, 2 * min_b - mutant, mutant)
        mutant = np.where(mutant > max_b, 2 * max_b - mutant, mutant)
        # Safety clip in case of extreme reflection
        mutant = np.clip(mutant, min_b, max_b)
        
        # 6. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # 7. Selection & Update
        succ_cr = []
        succ_f = []
        succ_diff = []
        new_arc_cands = []
        
        for i in range(pop_size):
            if i % 10 == 0 and check_timeout(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                diff = fitness[i] - f_trial
                
                # Store successful parameters
                succ_cr.append(cr[i])
                succ_f.append(f[i])
                succ_diff.append(diff)
                new_arc_cands.append(population[i].copy())
                
                # Update Population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
        
        # 8. Update Archive
        if new_arc_cands:
            cands = np.array(new_arc_cands)
            if len(archive) == 0:
                archive = cands
            else:
                archive = np.vstack((archive, cands))
            
            cap = int(pop_size * arc_rate)
            if len(archive) > cap:
                np.random.shuffle(archive)
                archive = archive[:cap]
        
        # 9. Update Memory (Weighted Lehmer Mean)
        # Prioritize parameters that caused larger fitness improvements
        if succ_diff:
            wdiff = np.array(succ_diff)
            total_diff = np.sum(wdiff)
            
            if total_diff > 0:
                weights = wdiff / total_diff
                
                # M_CR update (Weighted Arithmetic Mean)
                m_scr = np.sum(weights * np.array(succ_cr))
                m_cr[k_mem] = m_scr
                
                # M_F update (Weighted Lehmer Mean)
                s_f = np.array(succ_f)
                denom = np.sum(weights * s_f)
                if denom > 1e-12:
                    m_f[k_mem] = np.sum(weights * (s_f ** 2)) / denom
                else:
                    m_f[k_mem] = 0.5
                
                k_mem = (k_mem + 1) % H

    return best_fitness
