#The output values and codes for the last generated algorithms indicate that a **JADE-based** Differential Evolution (DE) outperformed **L-SHADE** and **CMA-ES**. This suggests the problem landscape benefits significantly from the restart mechanism (which JADE had and L-SHADE lacked in the previous iteration) and efficient exploration strategies.
#
#However, **SHADE** (Success-History Adaptive DE) is theoretically superior to JADE because it maintains a richer history of successful parameters ($H$ memory slots) rather than a single mean. The failure of the previous L-SHADE code was likely due to the lack of a **Restart Mechanism** (crucial for multimodal functions in limited time) and potentially aggressive population reduction.
#
#The following algorithm implements **SHADE-R (SHADE with Restarts)**. It combines the advanced adaptive memory of SHADE/L-SHADE with the robust restart capability of the successful JADE implementation.
#
#### Key Features:
#1.  **Success-History Adaptation (SHADE):** Uses a memory bank ($H=6$) to adapt Mutation ($F$) and Crossover ($CR$) parameters based on recent successful improvements, allowing the algorithm to learn the landscape's topology better than JADE.
#2.  **Restart Mechanism:** Detects convergence (stagnation) and restarts the population with random diversity while preserving the global best solution (Elitism). This prevents getting stuck in local optima.
#3.  **External Archive:** Maintains an archive of superseded solutions to increase diversity during mutation (`current-to-pbest/1` strategy).
#4.  **Vectorized Operations:** Maximizes the number of function evaluations within `max_time` by using NumPy vectorization for mutation, crossover, and bounding.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using SHADE-R (Success-History Adaptive Differential Evolution with Restarts).
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: SHADE typically uses N=100. 
    # We use a balanced size based on dimension to ensure speed in Python.
    pop_size = int(np.clip(10 * dim, 50, 100))
    
    # Archive parameters: A stores superseded successful parents
    # Archive size equals population size
    archive = [] 
    
    # Memory parameters (SHADE)
    H = 6
    mem_M_cr = np.full(H, 0.5)
    mem_M_f = np.full(H, 0.5)
    k_mem = 0
    
    # --- Bounds Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    best_fit = float('inf')
    best_sol = None
    
    # --- Main Optimization Loop (with Restarts) ---
    while True:
        # Check time budget before starting a new run
        if time.time() - start_time >= max_time:
            return best_fit
            
        # Initialize Population for this Restart
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Inject global best into new population to retain knowledge (Elitism)
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.zeros(pop_size)
        
        # Initial Evaluation
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_fit
            fitness[i] = func(pop[i])
            if fitness[i] < best_fit:
                best_fit = fitness[i]
                best_sol = pop[i].copy()
                
        # Sort for p-best selection (Best at index 0)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Reset Memory/Archive for new restart (fresh exploration)
        mem_M_cr.fill(0.5)
        mem_M_f.fill(0.5)
        k_mem = 0
        archive = [] 
        
        # --- Inner Generation Loop ---
        while True:
            if time.time() - start_time >= max_time:
                return best_fit
                
            # 1. Parameter Generation
            # Random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_M_cr[r_idx]
            m_f = mem_M_f[r_idx]
            
            # CR ~ Normal(M_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(M_f, 0.1)
            # Vectorized generation with retry logic for F <= 0
            f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Fix F <= 0 (retry)
            # Since numpy doesn't have a vectorized "re-sample specific indices",
            # we iterate just for the non-positive values (usually few)
            for i in range(pop_size):
                while f_vals[i] <= 0:
                    f_vals[i] = m_f[i] + 0.1 * np.random.standard_cauchy()
            
            f = np.minimum(f_vals, 1.0) # Clip F > 1 to 1.0
            
            # 2. Mutation: current-to-pbest/1
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Determine p (top %)
            # SHADE uses random p in [2/N, 0.2]
            p_min = 2.0 / pop_size
            p_curr = np.random.uniform(p_min, 0.2)
            top_p_count = int(max(2, pop_size * p_curr))
            
            # Select pbest
            pbest_indices = np.random.randint(0, top_p_count, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (from Pop, != i)
            r1 = np.random.randint(0, pop_size, pop_size)
            # Fix self-collision with simple shift
            hit_self = (r1 == np.arange(pop_size))
            if np.any(hit_self):
                r1[hit_self] = (r1[hit_self] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2 (from Union(Pop, Archive), != i, != r1)
            if len(archive) > 0:
                # Converting list to array can be slow if large, 
                # but archive size is limited to pop_size (~100), so it's fast.
                arr_archive = np.array(archive)
                pop_union = np.vstack((pop, arr_archive))
            else:
                pop_union = pop
                
            n_union = len(pop_union)
            r2 = np.random.randint(0, n_union, pop_size)
            
            # Fix collisions for r2 (simple retry logic)
            hit = (r2 == np.arange(pop_size)) | (r2 == r1)
            # Retry twice max to resolve collisions, usually sufficient
            for _ in range(2):
                if not np.any(hit): break
                r2[hit] = np.random.randint(0, n_union, np.sum(hit))
                hit = (r2 == np.arange(pop_size)) | (r2 == r1)
                
            x_r2 = pop_union[r2]
            
            # Compute Mutant Vectors
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            # mask = rand < CR
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # 4. Bound Constraint (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection & Evaluation
            diffs = []
            s_cr = []
            s_f = []
            
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return best_fit
                
                f_trial = func(trial[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    # If strictly better, record success info
                    if f_trial < fitness[i]:
                        # Add replaced parent to archive
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            # Replace random individual in archive
                            archive[np.random.randint(0, pop_size)] = pop[i].copy()
                            
                        diffs.append(fitness[i] - f_trial)
                        s_cr.append(cr[i])
                        s_f.append(f[i])
                        
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    if f_trial < best_fit:
                        best_fit = f_trial
                        best_sol = trial[i].copy()
            
            # 6. Memory Update
            if len(diffs) > 0:
                diffs = np.array(diffs)
                s_cr = np.array(s_cr)
                s_f = np.array(s_f)
                
                # Weighted improvement
                w = diffs / np.sum(diffs)
                
                # Weighted Mean for CR
                mean_cr = np.sum(w * s_cr)
                
                # Weighted Lehmer Mean for F: sum(w*f^2) / sum(w*f)
                mean_f = np.sum(w * (s_f**2)) / (np.sum(w * s_f) + 1e-15)
                
                mem_M_cr[k_mem] = mean_cr
                mem_M_f[k_mem] = mean_f
                k_mem = (k_mem + 1) % H
                
            # 7. Sort for p-best (SHADE requirement)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # 8. Convergence Check (Restart Trigger)
            # If population diversity is lost, we restart
            if np.std(fitness) < 1e-9 or (fitness[-1] - fitness[0]) < 1e-9:
                break
                
    return best_fit
