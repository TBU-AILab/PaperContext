#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution)** with a **Restart Mechanism**.
#
#**Key Improvements:**
#1.  **History-Based Adaptation**: Unlike simple jDE, SHADE uses a historical memory ($M_{CR}, M_F$) to guide the generation of control parameters $F$ and $CR$. This learns from successful updates over generations, effectively tuning the algorithm to the specific function landscape.
#2.  **Current-to-pbest Mutation**: Utilizes the `current-to-pbest/1` mutation strategy. This balances greediness (moving towards the top $p\%$ of best solutions) with diversity, preventing premature convergence better than `current-to-best`.
#3.  **External Archive**: Maintains an archive of inferior solutions recently replaced by better ones. These are used in the mutation step to maintain diversity and prevent the population from clustering too tightly too quickly.
#4.  **Convergence Restart**: Includes a safety restart mechanism. If the population diversity (standard deviation of fitness) drops below a threshold, the population is re-initialized (keeping the best solution) to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with a Restart Mechanism for robust global optimization within a time limit.
    """
    # Time management
    start_time = datetime.now()
    # Subtract a small buffer (100ms) to ensure safe return before timeout
    end_time = start_time + timedelta(seconds=max_time) - timedelta(milliseconds=100)

    # 1. Hyperparameters for SHADE
    # Population size: Standard is around 10*dim, but we cap it to [30, 80]
    # to ensure enough generations run within the time limit while maintaining diversity.
    pop_size = int(np.clip(10 * dim, 30, 80))
    
    # External Archive size (equal to pop_size)
    archive_size = pop_size
    
    # Memory size for historical parameters
    H = 6
    
    # 2. Setup Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # 3. Initialization
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Memory M_F and M_CR initialized to 0.5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0 # Memory index pointer
    
    archive = [] # List to store replaced solutions (maintains diversity)
    
    best_fitness = float('inf')
    best_sol = None

    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # If we timeout during init, return best found so far
            return best_fitness if best_fitness != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # Sort population indices by fitness for p-best selection
    sorted_idx = np.argsort(fitness)

    # 4. Optimization Loop
    while True:
        if datetime.now() >= end_time:
            return best_fitness

        # --- A. Adaptive Parameter Generation ---
        # Select random memory index k for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_f_k = mem_f[r_idx]
        m_cr_k = mem_cr[r_idx]
        
        # Generate CR ~ Normal(M_CR, 0.1)
        CR = np.random.normal(m_cr_k, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F ~ Cauchy(M_F, 0.1)
        # Handle F <= 0 (retry) and F > 1 (clip)
        F = m_f_k + 0.1 * np.random.standard_cauchy(pop_size)
        
        retry_mask = F <= 0
        while np.any(retry_mask):
            n_retry = np.sum(retry_mask)
            F[retry_mask] = m_f_k[retry_mask] + 0.1 * np.random.standard_cauchy(n_retry)
            retry_mask = F <= 0
        F[F > 1] = 1.0
        
        # --- B. Mutation: current-to-pbest/1 ---
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # 1. Select X_pbest (randomly from top p%)
        # p is random in [2/pop_size, 0.2]
        p_val = np.random.uniform(2/pop_size, 0.2)
        top_n = int(p_val * pop_size)
        if top_n < 2: top_n = 2
        
        top_indices = sorted_idx[:top_n]
        pbest_indices = np.random.choice(top_indices, pop_size)
        
        # 2. Select X_r1 (from Pop, r1 != i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        for i in range(pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, pop_size)
                
        # 3. Select X_r2 (from Pop U Archive, r2 != i, r2 != r1)
        # Create pool for selection
        if len(archive) > 0:
            pool = np.vstack((pop, np.array(archive)))
        else:
            pool = pop
        pool_size = len(pool)
        
        r2_indices = np.random.randint(0, pool_size, pop_size)
        for i in range(pop_size):
            # Collision check: if r2 is in current pop (idx < pop_size), it cannot be i or r1
            # If r2 is in Archive (idx >= pop_size), it is safe (indices don't overlap conceptually)
            while (r2_indices[i] < pop_size) and (r2_indices[i] == i or r2_indices[i] == r1_indices[i]):
                r2_indices[i] = np.random.randint(0, pool_size)
                
        # Compute Mutation Vectors
        X = pop
        X_pbest = pop[pbest_indices]
        X_r1 = pop[r1_indices]
        X_r2 = pool[r2_indices]
        
        F_col = F[:, None]
        mutant = X + F_col * (X_pbest - X) + F_col * (X_r1 - X_r2)
        
        # --- C. Crossover ---
        mask = np.random.rand(pop_size, dim) < CR[:, None]
        # Ensure at least one dimension is inherited from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, X)
        
        # Boundary constraints (Clip)
        trial = np.clip(trial, min_b, max_b)
        
        # --- D. Selection ---
        new_archive_candidates = []
        successful_F = []
        successful_CR = []
        improvements = []
        
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement
                if f_trial < fitness[i]:
                    new_archive_candidates.append(pop[i].copy())
                    successful_F.append(F[i])
                    successful_CR.append(CR[i])
                    improvements.append(fitness[i] - f_trial)
                
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
                    
        # --- E. Updates ---
        
        # 1. Update Archive
        for cand in new_archive_candidates:
            if len(archive) < archive_size:
                archive.append(cand)
            else:
                idx = np.random.randint(0, archive_size)
                archive[idx] = cand
                
        # 2. Update Memory (Weighted Lehmer Mean)
        if len(improvements) > 0:
            w = np.array(improvements)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                
                s_f = np.array(successful_F)
                s_cr = np.array(successful_CR)
                
                # Weighted Lehmer mean for F
                mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                
                # Weighted Arithmetic mean for CR
                mean_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                k_mem = (k_mem + 1) % H
                
        # 3. Resort indices for next generation
        sorted_idx = np.argsort(fitness)
        
        # --- F. Restart Mechanism ---
        # If converged (std dev very low), restart to escape local optimum
        if np.std(fitness) < 1e-6:
            # Reinitialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            fitness = np.full(pop_size, float('inf'))
            
            # Elitism: Keep best solution
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Archive and Memory
            archive = []
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            
            # Re-evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if datetime.now() >= end_time:
                    return best_fitness
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            sorted_idx = np.argsort(fitness)

    return best_fitness
