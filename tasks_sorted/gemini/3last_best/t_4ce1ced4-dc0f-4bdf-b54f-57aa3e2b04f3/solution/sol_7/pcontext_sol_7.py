#The best previously generated algorithm was **L-SHADE-jSO**, which achieved a significantly lower output value (1.36) compared to other attempts. The key to its success lies in the **L-SHADE** adaptive framework combined with **jSO** parameter control rules (e.g., linear p-best reduction) and **Linear Population Reduction (LPR)**.
#
#The following algorithm improves upon this by integrating **Opposition-Based Learning (OBL)** and a **Coordinate Descent Local Search**.
#
#### Key Improvements:
#1.  **Opposition-Based Initialization (OBL)**: Instead of starting with a purely random population, the algorithm generates a set of random individuals and their opposite counterparts (symmetric within bounds). It then selects the fittest individuals from this combined pool. This dramatically improves the quality of the initial basin of attraction.
#2.  **Smart Restart with OBL**: When stagnation is detected (low population variance), the algorithm performs a "Soft Restart". It keeps the elite solution but regenerates the rest of the population using OBL. This injects maximum diversity while retaining the best-found trait.
#3.  **Coordinate Descent Polish**: Before restarting, a lightweight coordinate descent (local search) is applied to the best solution. It perturbs each dimension individually to check for immediate improvements, effectively "polishing" the solution to finding the exact local minimum.
#4.  **L-SHADE-jSO Core**: Retains the robust adaptation logic:
#    *   **History Memory ($H=6$)** with Weighted Lehmer Mean updates.
#    *   **jSO Constraints**: Linearly reducing `p` (greediness) and clamping mutation factor $F \le 0.7$ in the early stages to prevent chaotic divergence.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using L-SHADE-jSO with Opposition-Based Initialization,
    Restart mechanism, and Coordinate Descent Local Polish.
    """
    start_time = datetime.now()
    # Reserve a small buffer to ensure we return before the strict timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    def check_timeout():
        return datetime.now() >= end_time

    # ---------------- Parameters ----------------
    # Population Sizing (Linear Reduction)
    # Start large to explore, reduce to focus.
    min_pop_size = 4
    # Heuristic: init ~ 20*dim, but clamped for safety
    init_pop_size = int(max(min_pop_size, min(20 * dim, 200)))
    pop_size = init_pop_size

    # SHADE Memory Size
    H = 6
    
    # Archive Size Rate (relative to population)
    arc_rate = 1.4

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # Initialize SHADE Memory
    # M_CR starts high (0.8) to encourage component mixing
    m_cr = np.full(H, 0.8)
    m_f = np.full(H, 0.5)
    k_mem = 0

    # --- Opposition-Based Initialization ---
    # 1. Generate Random Population
    p_rand = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # 2. Generate Opposite Population
    # OBL: x' = a + b - x
    p_opp = min_b + max_b - p_rand
    p_opp = np.clip(p_opp, min_b, max_b)

    # 3. Evaluate Combined Pool
    all_pop = np.vstack((p_rand, p_opp))
    all_fit = []
    
    best_fitness = float('inf')
    best_sol = np.zeros(dim)

    # Evaluation loop with timeout check
    for i in range(len(all_pop)):
        if check_timeout():
            return best_fitness if best_fitness != float('inf') else func(all_pop[0])
        
        val = func(all_pop[i])
        all_fit.append(val)
        
        if val < best_fitness:
            best_fitness = val
            best_sol = all_pop[i].copy()

    # 4. Select N best individuals
    all_fit = np.array(all_fit)
    sorted_idx = np.argsort(all_fit)
    population = all_pop[sorted_idx[:pop_size]]
    fitness = all_fit[sorted_idx[:pop_size]]

    # External Archive
    archive = np.empty((0, dim))

    # ---------------- Main Optimization Loop ----------------
    while not check_timeout():
        # Calculate Progress (0.0 -> 1.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time

        # 1. Linear Population Reduction (LPR)
        # N_t = round( (N_min - N_init) * progress + N_init )
        plan_pop_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)

        if pop_size > plan_pop_size:
            # Sort and truncate population
            sort_order = np.argsort(fitness)
            population = population[sort_order[:plan_pop_size]]
            fitness = fitness[sort_order[:plan_pop_size]]
            
            # Resize Archive
            current_arc_cap = int(plan_pop_size * arc_rate)
            if len(archive) > current_arc_cap:
                np.random.shuffle(archive)
                archive = archive[:current_arc_cap]
            
            pop_size = plan_pop_size

        # 2. Stagnation Detection & Restart
        # If fitness variance is negligible, we are likely trapped.
        fit_std = np.std(fitness)
        if fit_std < 1e-12:
            # --- Phase A: Coordinate Descent Local Polish ---
            # Try to refine the best solution by perturbing each dimension
            if not check_timeout():
                step_scale = 1e-4
                improved_polish = False
                for d in range(dim):
                    if check_timeout(): return best_fitness
                    
                    # Try negative step
                    temp = best_sol.copy()
                    step = step_scale * diff_b[d]
                    
                    temp[d] = np.clip(best_sol[d] - step, min_b[d], max_b[d])
                    val = func(temp)
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = temp.copy()
                        improved_polish = True
                    else:
                        # Try positive step
                        temp[d] = np.clip(best_sol[d] + step, min_b[d], max_b[d])
                        val = func(temp)
                        if val < best_fitness:
                            best_fitness = val
                            best_sol = temp.copy()
                            improved_polish = True

            # --- Phase B: Soft Restart with OBL ---
            # Keep elite, regenerate the rest using OBL to maximize diversity
            n_regen = pop_size - 1
            if n_regen > 0:
                # Generate Random candidates
                r_pop = min_b + np.random.rand(n_regen, dim) * diff_b
                # Generate Opposite candidates
                o_pop = min_b + max_b - r_pop
                o_pop = np.clip(o_pop, min_b, max_b)
                
                candidates = np.vstack((r_pop, o_pop))
                cand_fits = []
                
                for cand in candidates:
                    if check_timeout(): return best_fitness
                    vf = func(cand)
                    cand_fits.append(vf)
                    if vf < best_fitness:
                        best_fitness = vf
                        best_sol = cand.copy()
                
                # Select best to fill population
                cand_fits = np.array(cand_fits)
                srt_c = np.argsort(cand_fits)
                population[1:] = candidates[srt_c[:n_regen]]
                fitness[1:] = cand_fits[srt_c[:n_regen]]
            
            # Restore elite at index 0
            population[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Memory & Archive
            m_cr = np.full(H, 0.5)
            m_f = np.full(H, 0.5)
            archive = np.empty((0, dim))
            
            continue # Skip normal DE step this iteration

        # 3. Parameter Generation (SHADE)
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = m_f[r_idx[bad_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # jSO Feature: Clamp F to 0.7 during first 60% of search
        if progress < 0.6:
            f = np.minimum(f, 0.7)

        # 4. Mutation: current-to-pbest/1
        # p decreases linearly from 0.25 to 0.05
        p_val = 0.25 - (0.20 * progress)
        p_val = max(0.05, p_val)
        p_count = int(max(2, pop_size * p_val))
        
        # Select p-best
        sorted_indices = np.argsort(fitness)
        top_p_idx = sorted_indices[:p_count]
        
        pbest_idx = np.random.choice(top_p_idx, pop_size)
        x_pbest = population[pbest_idx]
        
        # Select r1 (random from pop)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_idx]
        
        # Select r2 (random from pop U archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        # Compute Mutant Vector
        f_v = f[:, None]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # Boundary Handling: Reflection
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
        succ_f = []
        succ_cr = []
        succ_diff = []
        new_arc_cands = []

        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                diff = fitness[i] - f_trial
                
                # Record success
                succ_f.append(f[i])
                succ_cr.append(cr[i])
                succ_diff.append(diff)
                new_arc_cands.append(population[i].copy())
                
                # Update Population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
        
        # Update Archive
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
        
        # Update SHADE Memory (Weighted Lehmer Mean)
        if succ_diff:
            diffs = np.array(succ_diff)
            total_diff = np.sum(diffs)
            
            if total_diff > 0:
                weights = diffs / total_diff
                
                # Update M_CR
                m_cr[k_mem] = np.sum(weights * np.array(succ_cr))
                
                # Update M_F
                sf_arr = np.array(succ_f)
                num = np.sum(weights * (sf_arr ** 2))
                den = np.sum(weights * sf_arr)
                if den > 1e-12:
                    m_f[k_mem] = num / den
                else:
                    m_f[k_mem] = 0.5
                
                k_mem = (k_mem + 1) % H

    return best_fitness
