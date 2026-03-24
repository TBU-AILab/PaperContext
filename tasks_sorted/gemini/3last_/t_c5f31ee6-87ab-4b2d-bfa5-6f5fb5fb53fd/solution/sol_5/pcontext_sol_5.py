#The proposed algorithm is an improved version of **SHADE** (Success-History based Adaptive Differential Evolution) enhanced with **Latin Hypercube Sampling (LHS)** for initialization, **Reflection-based Boundary Handling**, and a robust **Soft Restart** mechanism.
#
#### Key Improvements:
#1.  **Latin Hypercube Sampling (LHS):** Replaces random initialization to ensure a more even coverage of the search space, reducing the chance of starting all individuals in a poor region.
#2.  **Reflection Boundary Handling:** Instead of simply clipping values to bounds (which piles up solutions at the edges), this method reflects violations back into the search space. This helps maintain diversity and avoids getting stuck on boundaries.
#3.  **Soft Restart Strategy:** When the population stagnates (no improvement for a dynamic number of generations) or converges (low variance), the search restarts. However, the best solution found so far is injected into the new population ("Soft Restart") to ensure the search continues to refine the global optimum or explore new basins without losing progress.
#4.  **Dynamic Stagnation Limit:** The restart trigger adapts to the dimensionality of the problem (`max(40, dim * 2)`), allowing more time for convergence in higher dimensions.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History Adaptive Differential Evolution) with 
    Latin Hypercube Sampling, Reflection Boundary Handling, and Soft Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Balance between exploration (high) and generation count (low).
    # We clip the size to ensure reasonable performance across various dimensions
    # within the time limit.
    pop_size = int(np.clip(10 * dim, 30, 70))
    
    # SHADE Parameters
    memory_size = 5
    
    # --- Bounds Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Helper Functions ---
    def check_limit():
        return datetime.now() - start_time >= time_limit
    
    def latin_hypercube(n, d, low, diff):
        # Generate LHS samples for better initial coverage
        result = np.zeros((n, d))
        for i in range(d):
            perm = np.random.permutation(n)
            # Add random jitter within each interval
            jitter = np.random.rand(n)
            result[:, i] = (perm + jitter) / n
        return low + result * diff

    # --- Main Optimization Loop (Restarts) ---
    while not check_limit():
        
        # 1. Initialization
        # Use Latin Hypercube Sampling
        pop = latin_hypercube(pop_size, dim, min_b, diff_b)
        fitness = np.full(pop_size, float('inf'))
        
        # Soft Restart: Inject best found solution into the new population
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if check_limit(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # Memory Initialization (Historical Success)
        m_cr = np.full(memory_size, 0.5)
        m_f = np.full(memory_size, 0.5)
        k_mem = 0
        
        # Archive for maintaining diversity
        archive = []
        
        # Stagnation Counter
        stag_count = 0
        # Dynamic limit allows more persistence in higher dimensions
        stag_limit = max(40, dim * 2)
        
        # 2. Evolutionary Cycle
        while not check_limit():
            
            # --- Parameter Generation ---
            # Select random memory index
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            # Standard Cauchy = tan(pi * (rand - 0.5))
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Retry if F <= 0, Clip if F > 1
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.clip(f, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort population to find p-best
            sorted_idx = np.argsort(fitness)
            
            # Random p in [0.05, 0.2]
            p = np.random.uniform(0.05, 0.2)
            num_pbest = max(2, int(pop_size * p))
            pbest_indices = sorted_idx[:num_pbest]
            
            # Select x_pbest randomly from top p%
            x_pbest = pop[np.random.choice(pbest_indices, pop_size)]
            
            # Select x_r1 randomly from population
            r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1]
            
            # Select x_r2 from Union(Population, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2]
            
            # Compute Mutant Vector
            # v = x + F*(pbest - x) + F*(r1 - r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross = rand_vals < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # --- Boundary Handling: Reflection ---
            # Instead of clipping, reflect violations back to preserve distribution
            
            # Lower bound reflection
            mask_l = trial < min_b
            trial[mask_l] = 2 * min_b[mask_l] - trial[mask_l]
            
            # Upper bound reflection
            mask_u = trial > max_b
            trial[mask_u] = 2 * max_b[mask_u] - trial[mask_u]
            
            # Final safety clip in case reflection is still out-of-bounds
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            succ_f = []
            succ_cr = []
            diff_f = []
            improved_any = False
            
            for i in range(pop_size):
                if check_limit(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    # If strictly better, update memory info
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_f.append(fitness[i] - f_trial)
                        improved_any = True
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
                        stag_count = 0 # Reset stagnation on global improvement
            
            # Resize Archive randomly if it exceeds population size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # --- Memory Update (Weighted Lehmer Mean) ---
            if len(succ_f) > 0:
                w = np.array(diff_f)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    f_arr = np.array(succ_f)
                    
                    # Lehmer mean for F
                    mean_f = np.sum(w * (f_arr ** 2)) / np.sum(w * f_arr)
                    
                    # Arithmetic mean for CR
                    mean_cr = np.sum(w * np.array(succ_cr))
                    
                    m_f[k_mem] = mean_f
                    m_cr[k_mem] = mean_cr
                    k_mem = (k_mem + 1) % memory_size
            
            # --- Restart Check ---
            if not improved_any:
                stag_count += 1
                
            # Restart if population variance is negligible or stagnation is prolonged
            if np.std(fitness) < 1e-8 or stag_count > stag_limit:
                break
                
    return best_val
