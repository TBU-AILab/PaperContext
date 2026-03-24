#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 1.4592126628707263
#2. output value is: 14.09976309428093
#3. output value is: 39.46349818182866
#4. output value is: 46.14857048099866
#
#Here is an improved algorithm. It implements **Memetic L-SHADE (L-SHADE with Local Search)**, also incorporating the parameter adaptation strategies from **jSO**.
#
##### Rationale for Improvement
#The previous best algorithm (L-SHADE) reached a value of `1.45`, suggesting it found the basin of the global minimum but failed to converge to `0.0` or the exact minimum within the time limit. This is a common issue with global optimizers (DE) which are good at exploration but can be slow at fine-tuning.
#
#To address this, the new algorithm introduces:
#1.  **Memetic Local Search**: When the population shows signs of convergence (low variance), an intensive **Pattern Search (Coordinate Descent)** is triggered on the best solution found so far. This rapidly exploits the local area to find the precise minimum.
#2.  **jSO-style Adaptation**: Refines L-SHADE by using a linearly decreasing `p` value for the `current-to-pbest` mutation (shifts focus from exploration to exploitation over time) and a weighted memory update.
#3.  **Linear Population Size Reduction (LPSR)**: Keeps the population size dynamic to maximize the number of generations in the early phase and converge faster in the late phase.
#
##### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Memetic L-SHADE (L-SHADE with Local Search).
    Combines global search (L-SHADE with LPSR) with local refinement 
    (Pattern Search) to achieve high precision.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial population size: 20 * dim is a robust heuristic for DE
    pop_size_init = int(max(30, 20 * dim))
    pop_size_min = 4
    
    # SHADE Memory parameters
    H = 6
    M_CR = np.full(H, 0.5)
    M_F = np.full(H, 0.5)
    k_mem = 0
    
    # Archive for diversity maintenance
    archive = []
    
    # --- Initialization ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    pop_size = pop_size_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() >= end_time: return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # Local Search Step Size (initialized relative to bounds)
    ls_step = diff_b * 0.1
    
    # --- Main Loop ---
    while datetime.now() < end_time:
        
        # 1. Time & Population Management (LPSR)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size linearly decreasing
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Reduce population (keep best)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:target_size]]
            fitness = fitness[sorted_idx[:target_size]]
            pop_size = target_size
            
            # Resize archive
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        # 2. Convergence Check & Local Search (Memetic Phase)
        # Check standard deviation to detect stagnation/convergence
        std_fit = np.std(fitness)
        fit_range = np.max(fitness) - np.min(fitness)
        
        # Trigger if population is converged or very close to best
        if (std_fit < 1e-9 or fit_range < 1e-9) and pop_size > pop_size_min:
            
            # A. Intensive Pattern Search on Best Solution
            # This refines the solution significantly if stuck in a local basin
            improved_ls = True
            while improved_ls:
                if datetime.now() >= end_time: return best_val
                improved_ls = False
                
                # Coordinate Descent-like moves
                for d in range(dim):
                    # Try stepping back
                    x_temp = best_vec.copy()
                    x_temp[d] -= ls_step[d]
                    x_temp = np.clip(x_temp, min_b, max_b)
                    v_temp = func(x_temp)
                    
                    if v_temp < best_val:
                        best_val = v_temp
                        best_vec = x_temp.copy()
                        improved_ls = True
                        continue
                    
                    # Try stepping forward (search other side)
                    x_temp = best_vec.copy()
                    x_temp[d] += 0.5 * ls_step[d] 
                    x_temp = np.clip(x_temp, min_b, max_b)
                    v_temp = func(x_temp)
                    
                    if v_temp < best_val:
                        best_val = v_temp
                        best_vec = x_temp.copy()
                        improved_ls = True
                
                if not improved_ls:
                    # Reduce step size for finer search
                    ls_step *= 0.5
                    # Stop if step is too small
                    if np.max(ls_step) > 1e-12:
                        improved_ls = True 
            
            # B. Restart Population (Soft Restart)
            # Re-initialize population to explore other basins, but keep elite
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_vec
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            
            # Reset adaptation memory
            M_CR.fill(0.5)
            M_F.fill(0.5)
            archive = []
            ls_step = diff_b * 0.1 # Reset LS step size
            continue # Restart loop

        # 3. L-SHADE Parameter Generation
        # Randomly select memory index
        r_idxs = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_CR, 0.1)
        CR = np.random.normal(M_CR[r_idxs], 0.1)
        CR = np.clip(CR, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        F = []
        for rix in r_idxs:
            f = -1
            while f <= 0:
                f = M_F[rix] + 0.1 * np.random.standard_cauchy()
            if f > 1: f = 1.0
            F.append(f)
        F = np.array(F)
        
        # 4. Evolution (current-to-pbest/1)
        sorted_indices = np.argsort(fitness)
        
        # Dynamic 'p' value (jSO strategy): decreases from 0.25 to 0.05
        p_val = 0.25 - (0.20 * progress)
        p_val = max(p_val, 2.0/pop_size)
        
        new_pop = np.copy(pop)
        new_fitness = np.copy(fitness)
        
        succ_cr = []
        succ_f = []
        diff_f = []
        
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            
            # Select p-best
            top_cnt = int(p_val * pop_size)
            if top_cnt < 1: top_cnt = 1
            p_idx = sorted_indices[np.random.randint(0, top_cnt)]
            x_pbest = pop[p_idx]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Select r2 (distinct from i, r1; from union of pop and archive)
            cand_idx = np.random.randint(0, pop_size + len(archive))
            if cand_idx < pop_size:
                r2 = cand_idx
                while r2 == i or r2 == r1:
                    cand_idx = np.random.randint(0, pop_size + len(archive))
                    if cand_idx >= pop_size: break
                    r2 = cand_idx
            
            if cand_idx < pop_size:
                x_r2 = pop[cand_idx]
            else:
                x_r2 = archive[cand_idx - pop_size]
            
            # Mutation
            mutant = pop[i] + F[i] * (x_pbest - pop[i]) + F[i] * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            mask = np.random.rand(dim) < CR[i]
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                succ_cr.append(CR[i])
                succ_f.append(F[i])
                diff_f.append(fitness[i] - f_trial)
                archive.append(pop[i].copy())
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
            else:
                # In standard DE, we keep the old individual
                pass
                
        pop = new_pop
        fitness = new_fitness
        
        # Archive Maintenance
        while len(archive) > pop_size:
            del archive[np.random.randint(0, len(archive))]
            
        # 5. Memory Update
        if len(succ_f) > 0:
            w = np.array(diff_f)
            w = w / np.sum(w)
            
            # Weighted Lehmer mean
            m_cr_new = np.sum(w * np.array(succ_cr))
            m_f_new = np.sum(w * np.array(succ_f)**2) / np.sum(w * np.array(succ_f))
            
            # Update memory (weighted average with previous)
            M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * m_cr_new
            M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * m_f_new
            
            k_mem = (k_mem + 1) % H

    return best_val
