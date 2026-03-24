#The output values and codes for the best generated algorithms are as follows:
#1. output value is: 12.086585024030029
#   (Algorithm: Differential Evolution with 'current-to-best' strategy)
#2. output value is: 16.050773380037242
#   (Algorithm: Asynchronous Self-Adaptive DE / jDE-Async)
#3. output value is: 38.97548889993502
#   (Algorithm: L-SHADE with Linear Population Size Reduction)
#
#The following is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#### Algorithm Explanation
#This algorithm is an **Asynchronous SHADE (Success-History Adaptive Differential Evolution) with Archive and Restart**. It synthesizes the most effective components from the previous results to achieve superior performance:
#
#1.  **Asynchronous Update (from jDE-Async)**: Unlike the standard SHADE (which is generational), this algorithm updates the population immediately after a better solution is found. This drastically accelerates convergence by allowing the population to exploit improvements instantly.
#2.  **SHADE Parameter Adaptation**: It uses historical memory ($M_{CR}, M_F$) to adapt crossover and mutation parameters, which typically outperforms the simpler jDE adaptation.
#3.  **Midpoint-Target Bound Handling**: Instead of blindly clipping values to bounds (which traps solutions at the edges), it places violating solutions at the midpoint between the parent and the bound. This preserves search direction and diversity.
#4.  **Stagnation Restart**: To solve the premature convergence problem of greedy strategies, it monitors population variance. If the search stagnates, it performs a "soft restart"—keeping the global best solution but re-initializing the rest of the population to explore new areas within the remaining time budget.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Asynchronous SHADE with Archive and Restart.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adapted to dimension
    # A size of 20*D is robust, clipped to [50, 100] to ensure speed.
    pop_size = int(np.clip(dim * 20, 50, 100))
    
    # Extract bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Global Best tracking
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
            
    # --- SHADE Memory Initialization ---
    memory_size = 5
    M_CR = np.full(memory_size, 0.5)
    M_F = np.full(memory_size, 0.5)
    k_mem = 0
    
    # External Archive (stores replaced individuals to maintain diversity)
    archive = [] 
    
    # --- Main Optimization Loop ---
    while True:
        # Time Check
        if (time.time() - start_time) >= max_time:
            return best_val
            
        # Sort population indices by fitness (Best -> Worst)
        # Required for 'current-to-pbest' strategy
        sorted_indices = np.argsort(fitness)
        
        # --- Stagnation Detection & Restart ---
        # If population diversity (std dev) is negligible, we are stuck.
        if np.std(fitness) < 1e-8:
            # Only restart if we have significant time left (>10%)
            if (time.time() - start_time) < (max_time * 0.9):
                # Soft Restart: Preserve best, re-init the rest
                # The best is currently at sorted_indices[0] or tracked in best_vec
                
                # Re-initialize population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                # Restore the global best to index 0
                pop[0] = best_vec.copy()
                
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Reset Memory and Archive
                M_CR.fill(0.5)
                M_F.fill(0.5)
                archive = []
                
                # Re-evaluate new individuals (skipping index 0)
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return best_val
                    val = func(pop[k])
                    fitness[k] = val
                    if val < best_val:
                        best_val = val
                        best_vec = pop[k].copy()
                        
                # Re-sort after restart
                sorted_indices = np.argsort(fitness)
        
        # Buffers for successful parameters in this "sweep"
        succ_F = []
        succ_CR = []
        succ_diff = []
        
        # --- Asynchronous Evolution Step ---
        # Iterate through population
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            idx = i # Current individual index
            
            # 1. Parameter Generation
            # Select random memory slot
            r_idx = np.random.randint(0, memory_size)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # Generate CR (Normal dist, clipped [0, 1])
            curr_CR = np.random.normal(mu_cr, 0.1)
            curr_CR = np.clip(curr_CR, 0.0, 1.0)
            
            # Generate F (Cauchy dist, regenerate if <= 0, clip to 1)
            while True:
                curr_F = mu_f + 0.1 * np.random.standard_cauchy()
                if curr_F > 0:
                    break
            curr_F = min(curr_F, 1.0)
            
            # 2. Mutation: current-to-pbest/1
            # Select p-best: Randomly from top p% (p in [2/N, 0.2])
            p_min = 2.0 / pop_size
            p = np.random.uniform(p_min, 0.2)
            n_pbest = int(max(2, p * pop_size))
            
            pbest_rank = np.random.randint(0, n_pbest)
            pbest_idx = sorted_indices[pbest_rank]
            x_pbest = pop[pbest_idx]
            
            x_curr = pop[idx]
            
            # Select r1: Random distinct from idx
            while True:
                r1 = np.random.randint(0, pop_size)
                if r1 != idx:
                    break
            x_r1 = pop[r1]
            
            # Select r2: Random distinct from idx, r1. Taken from Union(Pop, Archive)
            n_arch = len(archive)
            while True:
                r2 = np.random.randint(0, pop_size + n_arch)
                # Map r2 to actual vector
                if r2 < pop_size:
                    if r2 != idx and r2 != r1:
                        x_r2 = pop[r2]
                        break
                else:
                    x_r2 = archive[r2 - pop_size]
                    break
            
            # Compute Mutant Vector
            mutant = x_curr + curr_F * (x_pbest - x_curr) + curr_F * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < curr_CR
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, x_curr)
            
            # 4. Bound Constraint Handling (Midpoint-Target)
            # If out of bounds, place between parent and bound
            viol_l = trial < min_b
            if np.any(viol_l):
                trial[viol_l] = (min_b[viol_l] + x_curr[viol_l]) * 0.5
                
            viol_u = trial > max_b
            if np.any(viol_u):
                trial[viol_u] = (max_b[viol_u] + x_curr[viol_u]) * 0.5
            
            # 5. Evaluation
            f_trial = func(trial)
            
            # 6. Selection
            if f_trial < fitness[idx]:
                # Improvement
                diff = fitness[idx] - f_trial
                
                # Update Archive: Add parent before replacement
                if len(archive) < pop_size:
                    archive.append(pop[idx].copy())
                else:
                    # Replace random member if full
                    rem_loc = np.random.randint(0, pop_size)
                    archive[rem_loc] = pop[idx].copy()
                
                # Asynchronous Update: Replace immediately
                pop[idx] = trial
                fitness[idx] = f_trial
                
                # Record success for memory update
                succ_F.append(curr_F)
                succ_CR.append(curr_CR)
                succ_diff.append(diff)
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
        
        # --- Update SHADE Memory ---
        # Performed once per full pass over the population
        if len(succ_diff) > 0:
            succ_diff = np.array(succ_diff)
            succ_F = np.array(succ_F)
            succ_CR = np.array(succ_CR)
            
            # Weights based on fitness improvement amount
            total_diff = np.sum(succ_diff)
            if total_diff > 0:
                weights = succ_diff / total_diff
                
                # Update M_CR (Weighted Mean)
                mean_cr = np.sum(weights * succ_CR)
                M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                
                # Update M_F (Weighted Lehmer Mean)
                # sum(w * f^2) / sum(w * f)
                num = np.sum(weights * (succ_F ** 2))
                den = np.sum(weights * succ_F)
                mean_f = num / den if den > 0 else 0.5
                M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % memory_size

    return best_val
