#The algorithm below implements **L-SHADE** (Linear Success-History based Adaptive Differential Evolution). This is a state-of-the-art improvement over JADE/DE that dynamically adapts population size and evolutionary parameters over time.
#
#### Key Features:
#1.  **Linear Population Size Reduction (LPSR):** The algorithm starts with a large population for exploration and linearly reduces it to a minimum size as time progresses. This forces convergence and shifts focus from exploration to exploitation naturally.
#2.  **History-based Parameter Adaptation:** It maintains a memory of successful Mutation Factors ($F$) and Crossover Rates ($CR$) and generates new parameters based on this history, allowing the algorithm to "learn" the landscape.
#3.  **Current-to-pbest Mutation:** Guides the search towards the top $p\%$ best solutions while maintaining diversity via an external archive.
#4.  **Time-Aware Budgeting:** The reduction schedule and loops are strictly controlled by the `max_time` parameter, ensuring the algorithm utilizes the available time budget efficiently without overrunning.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # --- Configuration ---
    # Population sizing: Adaptive based on dimension
    # N_init: Start high for exploration. Capped at 200 for performance on short deadlines.
    pop_size_init = int(np.clip(18 * dim, 30, 200))
    pop_size_min = 4
    
    # Memory for adaptive parameters (History size H=5)
    mem_size = 5
    m_cr = np.full(mem_size, 0.5)
    m_f = np.full(mem_size, 0.5)
    mem_k = 0
    
    # Archive for diversity
    archive = []
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    pop_size = pop_size_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    best_val = float('inf')
    best_vec = None
    
    # Evaluate Initial Population safely
    n_evaluated = 0
    for i in range(pop_size):
        if time.time() - start_time > max_time * 0.98:
            break
        val = func(pop[i])
        fitness[i] = val
        n_evaluated += 1
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # Handle early timeout during initialization
    if n_evaluated < pop_size:
        pop_size = n_evaluated
        pop = pop[:pop_size]
        fitness = fitness[:pop_size]
        
    if pop_size < 4:
        return best_val
        
    # --- Main Optimization Loop ---
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            break
            
        # 1. Linear Population Size Reduction
        # Calculate Target Population Size based on time progress
        progress = elapsed / max_time
        target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            n_reduce = pop_size - target_size
            # Remove worst individuals (highest fitness)
            sort_idx = np.argsort(fitness)
            keep_idx = sort_idx[:target_size]
            
            pop = pop[keep_idx]
            fitness = fitness[keep_idx]
            pop_size = target_size
            
            # Resize Archive to match new population size
            if len(archive) > pop_size:
                import random
                random.shuffle(archive)
                archive = archive[:pop_size]
                
        # 2. Adaptive Parameter Generation
        # Select memory slots
        r_idx = np.random.randint(0, mem_size, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        
        # Repair F: if F <= 0, regenerate. if F > 1, clip.
        bad_f = f <= 0
        retry_limit = 10
        count = 0
        while np.any(bad_f) and count < retry_limit:
            n_bad = np.sum(bad_f)
            f[bad_f] = np.random.standard_cauchy(n_bad) * 0.1 + mu_f[bad_f]
            bad_f = f <= 0
            count += 1
        f = np.clip(f, 0, 1) # Clip remaining negatives to 0, positives > 1 to 1
        
        # 3. Mutation: current-to-pbest/1
        # Select p-best individuals (top 11%)
        n_pbest = max(2, int(0.11 * pop_size))
        sorted_indices = np.argsort(fitness)
        pbest_candidates = sorted_indices[:n_pbest]
        pbest_idx = np.random.choice(pbest_candidates, pop_size)
        x_pbest = pop[pbest_idx]
        
        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        for i in range(pop_size):
            while r1[i] == i:
                r1[i] = np.random.randint(0, pop_size)
                
        # Select r2 != i and r2 != r1 from Union(Pop, Archive)
        if len(archive) > 0:
            arc_np = np.array(archive)
            union_pop = np.vstack((pop, arc_np))
        else:
            union_pop = pop
            
        r2 = np.random.randint(0, len(union_pop), pop_size)
        for i in range(pop_size):
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, len(union_pop))
        
        x_r1 = pop[r1]
        x_r2 = union_pop[r2]
        
        # Calculate Mutant Vector V
        f_col = f[:, None]
        v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # Boundary Correction (Midpoint method)
        # If out of bounds, place halfway between parent and bound
        mask_l = v < min_b
        mask_h = v > max_b
        
        rows_l, cols_l = np.where(mask_l)
        v[rows_l, cols_l] = (pop[rows_l, cols_l] + min_b[cols_l]) / 2.0
        
        rows_h, cols_h = np.where(mask_h)
        v[rows_h, cols_h] = (pop[rows_h, cols_h] + max_b[cols_h]) / 2.0
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) <= cr[:, None]
        cross_mask[np.arange(pop_size), j_rand] = True
        
        u = np.where(cross_mask, v, pop)
        
        # 5. Evaluation and Selection
        succ_f = []
        succ_cr = []
        diff_fit = []
        
        for i in range(pop_size):
            # Frequent time check
            if (i % 5 == 0) and (time.time() - start_time > max_time):
                return best_val
            
            # Enforce bounds strictly for function call
            u_eval = np.clip(u[i], min_b, max_b)
            val = func(u_eval)
            
            if val < fitness[i]:
                # Improvement
                archive.append(pop[i].copy())
                succ_f.append(f[i])
                succ_cr.append(cr[i])
                diff_fit.append(fitness[i] - val)
                
                fitness[i] = val
                pop[i] = u_eval
                
                if val < best_val:
                    best_val = val
                    best_vec = u_eval.copy()
            
            elif val == fitness[i]:
                pop[i] = u_eval
                
        # 6. Memory Update (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            diff_fit = np.array(diff_fit)
            
            denom = np.sum(diff_fit)
            if denom == 0: 
                weights = np.ones(len(diff_fit)) / len(diff_fit)
            else: 
                weights = diff_fit / denom
            
            mean_f = np.sum(weights * (succ_f**2)) / (np.sum(weights * succ_f) + 1e-15)
            mean_cr = np.sum(weights * succ_cr)
            
            m_f[mem_k] = mean_f
            m_cr[mem_k] = mean_cr
            
            mem_k = (mem_k + 1) % mem_size
            
    return best_val
