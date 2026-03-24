#The following algorithm implements **jSO-MTS** (jSO with Multiple Trajectory Search).
#
#**Key Improvements:**
#1.  **Refined jSO (L-SHADE variant)**: Accurately implements the **Weighted Mutation** ($F_w$) and population size reduction from the jSO algorithm (a top-ranking CEC winner). This dampens early exploration greediness to prevent premature convergence and accelerates exploitation in later stages.
#2.  **Efficient Local Search (MTS-LS1)**: Replaces the heavy Eigen-crossover with a budget-aware **Coordinate Descent** (MTS-LS1) applied to the best solution when stagnation is detected. This refines solutions with high precision without the $O(D^3)$ overhead of covariance matrix operations.
#3.  **Boundary Handling**: Switches to **Midpoint-Target** bounce-back (`(min + x_parent)/2`) instead of reflection. This keeps particles closer to the boundaries (where optima often lie in benchmarks) without trapping them.
#4.  **Smart Restart**: Uses a variance-based restart trigger that preserves the elite solution but refreshes the population with a mix of global uniform samples and local Gaussian samples around the best found solution.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    jSO-MTS Algorithm (jSO with Multiple Trajectory Search Local Search).
    Efficient for black-box optimization within limited time.
    """
    start_time = datetime.now()
    # Use 99% of time to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.99)
    
    def check_time():
        return datetime.now() - start_time < time_limit

    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing: jSO guideline N = 25 * log(D) * sqrt(D)
    # Clipped to safe range [30, 400] for performance stability
    N_init = int(np.round(25 * np.log(dim) * np.sqrt(dim)))
    N_init = np.clip(N_init, 30, 400)
    N_min = 4
    
    pop_size = N_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    best_fitness = float('inf')
    best_pos = None

    # Evaluation Helper with Global Best Update
    def evaluate(x):
        nonlocal best_fitness, best_pos
        # Ensure bounds before eval
        x_c = np.clip(x, min_b, max_b)
        val = func(x_c)
        if val < best_fitness:
            best_fitness = val
            best_pos = x_c.copy()
        return val

    # Initial Evaluation
    for i in range(pop_size):
        if not check_time(): return best_fitness
        fitness[i] = evaluate(pop[i])
        
    if not check_time(): return best_fitness

    # --- Memory & Archive (L-SHADE components) ---
    H = 5
    mem_F = np.full(H, 0.5)
    mem_CR = np.full(H, 0.8)
    k_mem = 0
    archive = []
    
    # --- Local Search State ---
    mts_range = diff_b * 0.4
    stagnation_counter = 0
    last_best = best_fitness
    
    # --- Main Optimization Loop ---
    while check_time():
        # Calculate Progress (0.0 to 1.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(elapsed / max_time, 1.0)
        
        # 1. Linear Population Size Reduction (LPSR)
        N_next = int(round(N_min + (N_init - N_min) * (1.0 - progress)))
        N_next = max(N_min, N_next)
        
        if pop_size > N_next:
            n_del = pop_size - N_next
            # Remove worst individuals
            idx = np.argsort(fitness)
            pop = pop[idx[:-n_del]]
            fitness = fitness[idx[:-n_del]]
            pop_size = N_next
            # Resize archive
            if len(archive) > pop_size:
                import random
                random.shuffle(archive)
                archive = archive[:pop_size]

        # 2. Adaptation (jSO specific)
        # p (top % for mutation) reduces linearly from 0.25 to 2/N
        p_max, p_min = 0.25, 2.0 / pop_size
        p = p_max - (p_max - p_min) * progress
        
        # Generate F and CR based on Memory
        rand_mem_idx = np.random.randint(0, H, pop_size)
        mu_F = mem_F[rand_mem_idx]
        mu_CR = mem_CR[rand_mem_idx]
        
        # Cauchy distribution for F
        F = mu_F + 0.1 * np.random.standard_cauchy(pop_size)
        # Handle F constraints
        F[F > 1.0] = 1.0
        retry_F = F <= 0
        while np.any(retry_F):
            F[retry_F] = mu_F[retry_F] + 0.1 * np.random.standard_cauchy(np.sum(retry_F))
            retry_F = F <= 0
            F[F > 1.0] = 1.0
            
        # Normal distribution for CR
        CR = np.random.normal(mu_CR, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # jSO Weighted Mutation Factor (F_w)
        # Damps exploration step early to avoid greedy convergence
        F_w = F.copy()
        if progress < 0.2:
            F_w *= 0.7
        elif progress < 0.4:
            F_w *= 0.8
            
        # 3. Mutation: current-to-pbest-w/1
        sorted_idx = np.argsort(fitness)
        n_pbest = max(1, int(p * pop_size))
        pbest_pool = sorted_idx[:n_pbest]
        
        # Select p-best randomly
        r_pbest = np.random.choice(pbest_pool, pop_size)
        x_pbest = pop[r_pbest]
        
        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        hit = r1 == np.arange(pop_size)
        r1[hit] = (r1[hit] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 != r1, r2 != i (From Pop U Archive)
        pool = pop if not archive else np.vstack((pop, np.array(archive)))
        r2 = np.random.randint(0, len(pool), pop_size)
        hit2 = (r2 == r1) | (r2 == np.arange(pop_size))
        r2[hit2] = (r2[hit2] + 1) % len(pool)
        x_r2 = pool[r2]
        
        # Calculate Mutant Vector
        # v = x + F_w * (pbest - x) + F * (r1 - r2)
        mutant = pop + F_w[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) <= CR[:, None]
        cross_mask[np.arange(pop_size), j_rand] = True
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Boundary Handling (Midpoint Target)
        # Instead of reflecting, place point between parent and bound
        low_vio = trial < min_b
        high_vio = trial > max_b
        
        if np.any(low_vio):
            trial[low_vio] = (min_b[np.where(low_vio)[1]] + pop[low_vio]) / 2.0
        if np.any(high_vio):
            trial[high_vio] = (max_b[np.where(high_vio)[1]] + pop[high_vio]) / 2.0
        
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Evaluation & Selection
        new_pop = pop.copy()
        new_fit = fitness.copy()
        
        succ_F, succ_CR, diff_f = [], [], []
        
        for i in range(pop_size):
            if not check_time(): return best_fitness
            
            f_new = evaluate(trial[i])
            
            if f_new < fitness[i]:
                new_pop[i] = trial[i]
                new_fit[i] = f_new
                
                archive.append(pop[i].copy())
                succ_F.append(F[i])
                succ_CR.append(CR[i])
                diff_f.append(fitness[i] - f_new)
        
        pop = new_pop
        fitness = new_fit
        
        # Maintain Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 7. Update Memory (Weighted Lehmer Mean)
        if diff_f:
            w = np.array(diff_f)
            total = np.sum(w)
            if total > 0:
                w /= total
                sF = np.array(succ_F)
                sCR = np.array(succ_CR)
                
                mean_F = np.sum(w * (sF**2)) / (np.sum(w * sF) + 1e-15)
                mean_CR = np.sum(w * sCR)
                
                mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_F
                mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_CR
                k_mem = (k_mem + 1) % H
                
        # 8. Stagnation Check & Local Search (MTS-LS1)
        if best_fitness < last_best:
            last_best = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Trigger LS if stagnant for 15 gens or late in search
        if stagnation_counter > 15 or progress > 0.9:
            if check_time():
                # MTS Coordinate Descent on Best Solution
                # Limited budget (50 steps max per call to save time)
                ls_budget = 50
                improved_ls = False
                x_curr = best_pos.copy()
                f_curr = best_fitness
                
                # Search dimensions in random order
                perm = np.random.permutation(dim)
                for d_idx in perm:
                    if ls_budget <= 0 or not check_time(): break
                    
                    original_val = x_curr[d_idx]
                    step = mts_range[d_idx]
                    
                    # Try Negative Step
                    x_curr[d_idx] = np.clip(original_val - step, min_b[d_idx], max_b[d_idx])
                    val = func(x_curr)
                    ls_budget -= 1
                    
                    if val < f_curr:
                        f_curr = val
                        best_fitness = val
                        best_pos = x_curr.copy()
                        improved_ls = True
                    else:
                        # Try Positive Step (Half size)
                        x_curr[d_idx] = np.clip(original_val + 0.5 * step, min_b[d_idx], max_b[d_idx])
                        val = func(x_curr)
                        ls_budget -= 1
                        
                        if val < f_curr:
                            f_curr = val
                            best_fitness = val
                            best_pos = x_curr.copy()
                            improved_ls = True
                        else:
                            # Revert
                            x_curr[d_idx] = original_val
                
                if not improved_ls:
                    mts_range *= 0.5 # Shrink search radius
                else:
                    stagnation_counter = 0 # Reset stagnation

        # 9. Restart Strategy
        # If population converged (low variance) and local search radius is tiny
        if np.std(fitness) < 1e-8 and np.max(mts_range) < 1e-6:
            # Preserve Elite
            pop[0] = best_pos.copy()
            fitness[0] = best_fitness
            
            # Reset adaptation partially
            mts_range = diff_b * 0.4
            mem_F.fill(0.5)
            mem_CR.fill(0.8)
            
            # 50% Random
            n_rnd = (pop_size - 1) // 2
            pop[1:1+n_rnd] = min_b + np.random.rand(n_rnd, dim) * diff_b
            
            # 50% Small Noise around Best
            n_noise = (pop_size - 1) - n_rnd
            noise = np.random.randn(n_noise, dim) * diff_b * 0.01
            pop[1+n_rnd:] = np.clip(best_pos + noise, min_b, max_b)
            
            # Evaluate new solutions
            for k in range(1, pop_size):
                if not check_time(): return best_fitness
                fitness[k] = evaluate(pop[k])

    return best_fitness
