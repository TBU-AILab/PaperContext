#The following is a fully operational, self-contained implementation of an improved optimization algorithm.
#
##### Algorithm Explanation
#This algorithm is an **Optimized Asynchronous SHADE (Success-History Adaptive Differential Evolution) with Vectorized Random Number Generation and Restart**.
#
#It builds upon the best-performing "Asynchronous SHADE" strategy but addresses its main weakness—computational overhead in Python loops—to maximize the search efficiency within the time limit.
#
#**Key Improvements:**
#1.  **Vectorized Parameter Generation**: Instead of generating random numbers ($F$, $CR$, $p$, $r1$) individually inside the loop (which incurs high overhead), this implementation pre-generates them in batches using NumPy at the start of each generation. This retains the algorithmic benefit of **Asynchronous Updates** (using the latest solutions immediately) while significantly speeding up execution.
#2.  **Robust Restart Mechanism**: It monitors the population range (`max - min` fitness). If the population converges/stagnates, it triggers a "soft restart," preserving the global best solution but scattering the rest of the population to explore new basins of attraction.
#3.  **Midpoint Bound Handling**: When solutions violate bounds, they are placed at the midpoint between the parent and the bound. This "bounce-back" behavior preserves diversity better than simple clipping.
#4.  **Optimized Hyperparameters**: The population size is adapted to the dimension ($15 \times D$) but clipped to a safe range [30, 100] to ensure the algorithm runs many generations even in high dimensions.
#
##### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Optimized Asynchronous SHADE with Vectorized RNG
    and Restart Mechanism.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Balanced for speed vs exploration
    # 15*D is robust, clipped to [30, 100] to ensure high throughput
    pop_size = int(np.clip(dim * 15, 30, 100))
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- SHADE Memory & Archive Initialization ---
    memory_size = 5
    M_CR = np.full(memory_size, 0.5)
    M_F = np.full(memory_size, 0.5)
    k_mem = 0
    archive = []
    
    # --- Main Optimization Loop ---
    while True:
        # Check time budget at start of generation
        if (time.time() - start_time) >= max_time:
            return best_val
            
        # Sort indices by fitness (required for current-to-pbest strategy)
        sorted_indices = np.argsort(fitness)
        
        # --- Stagnation Detection & Restart ---
        # Check the range of fitness values. If collapsed, restart.
        fit_range = fitness[sorted_indices[-1]] - fitness[sorted_indices[0]]
        if fit_range < 1e-8:
            # Only restart if significant time remains (>10%)
            if (time.time() - start_time) < (max_time * 0.9):
                # Soft Restart: Keep global best, re-initialize the rest
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = best_vec.copy()
                
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Reset Memory and Archive
                M_CR.fill(0.5)
                M_F.fill(0.5)
                archive = []
                
                # Re-evaluate new individuals (skipping index 0)
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time: return best_val
                    val = func(pop[k])
                    fitness[k] = val
                    if val < best_val:
                        best_val = val
                        best_vec = pop[k].copy()
                
                # Re-sort after restart
                sorted_indices = np.argsort(fitness)

        # --- Vectorized Parameter Generation (Pre-calculation) ---
        # Generating params for the whole population at once to reduce loop overhead
        
        # 1. Select Memory Indices
        r_mem_idx = np.random.randint(0, memory_size, pop_size)
        m_cr = M_CR[r_mem_idx]
        m_f = M_F[r_mem_idx]
        
        # 2. Generate CR and F
        CR = np.random.normal(m_cr, 0.1)
        np.clip(CR, 0.0, 1.0, out=CR)
        
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        # Vectorized repair for F <= 0
        while True:
            bad_f = F <= 0
            if not np.any(bad_f): break
            F[bad_f] = m_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
        np.minimum(F, 1.0, out=F)
        
        # 3. Generate p-best indices
        # Random p in [2/N, 0.2]
        p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
        n_pbest = (p_val * pop_size).astype(int)
        n_pbest = np.maximum(n_pbest, 2)
        # Select random rank up to n_pbest
        p_ranks = (np.random.rand(pop_size) * n_pbest).astype(int)
        pbest_indices = sorted_indices[p_ranks]
        
        # 4. Generate r1 indices (distinct from current i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Fix collision with self (i)
        coll_mask = (r1_indices == np.arange(pop_size))
        r1_indices[coll_mask] = (r1_indices[coll_mask] + 1) % pop_size
        
        # 5. Generate Crossover Masks
        cross_mask = np.random.rand(pop_size, dim) < CR[:, None]
        # Force at least one dimension to change
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        # Buffers for successful parameters
        succ_F = []
        succ_CR = []
        succ_diff = []
        
        # --- Asynchronous Evolution Loop ---
        # We iterate sequentially to allow immediate update of 'pop',
        # but we use the pre-calculated params (CR, F, indices) for speed.
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time: return best_val
            
            x_curr = pop[i]
            x_pbest = pop[pbest_indices[i]]
            x_r1 = pop[r1_indices[i]] # Fetches latest version (Async benefit)
            
            # Select r2: Union of Population and Archive
            n_arch = len(archive)
            if n_arch > 0:
                r2 = np.random.randint(0, pop_size + n_arch)
                if r2 < pop_size:
                    if r2 == i or r2 == r1_indices[i]:
                        r2 = (r2 + 1) % pop_size
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
            else:
                r2 = np.random.randint(0, pop_size)
                while r2 == i or r2 == r1_indices[i]:
                     r2 = np.random.randint(0, pop_size)
                x_r2 = pop[r2]
            
            # Mutation: current-to-pbest/1
            f_i = F[i]
            mutant = x_curr + f_i * (x_pbest - x_curr) + f_i * (x_r1 - x_r2)
            
            # Crossover
            trial = np.where(cross_mask[i], mutant, x_curr)
            
            # Bound Constraint Handling (Midpoint Target)
            # If out of bounds, place halfway between parent and bound
            viol_l = trial < min_b
            if np.any(viol_l):
                trial[viol_l] = (min_b[viol_l] + x_curr[viol_l]) * 0.5
            viol_u = trial > max_b
            if np.any(viol_u):
                trial[viol_u] = (max_b[viol_u] + x_curr[viol_u]) * 0.5
                
            # Evaluation
            f_trial = func(trial)
            
            # Selection
            if f_trial < fitness[i]:
                diff = fitness[i] - f_trial
                
                # Add parent to archive
                archive.append(x_curr.copy())
                if len(archive) > pop_size:
                    # Remove random member if full
                    del archive[np.random.randint(0, len(archive))]
                
                # Update Population (Asynchronous)
                pop[i] = trial
                fitness[i] = f_trial
                
                # Record Success
                succ_F.append(f_i)
                succ_CR.append(CR[i])
                succ_diff.append(diff)
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
        
        # --- Update SHADE Memory ---
        if succ_diff:
            succ_diff = np.array(succ_diff)
            succ_F = np.array(succ_F)
            succ_CR = np.array(succ_CR)
            
            weights = succ_diff / np.sum(succ_diff)
            
            # Update M_CR (Weighted Mean)
            m_cr_new = np.sum(weights * succ_CR)
            M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * m_cr_new
            
            # Update M_F (Weighted Lehmer Mean)
            den = np.sum(weights * succ_F)
            if den > 0:
                m_f_new = np.sum(weights * (succ_F**2)) / den
                M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * m_f_new
            
            k_mem = (k_mem + 1) % memory_size

    return best_val
