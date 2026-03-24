#Here is an improved algorithm. It implements **SHADE (Success-History based Adaptive Differential Evolution)** with an **Adaptive Restart** mechanism.
#
#**Reasoning for Improvement:**
#1.  **Adaptivity (SHADE)**: Unlike standard DE with fixed or random parameters, SHADE automatically adapts the Mutation Factor ($F$) and Crossover Rate ($CR$) based on successful evolutionary steps. It learns which parameters work best for the specific function landscape (e.g., separable vs. rotated).
#2.  **Current-to-p-best Mutation**: Instead of greedily moving towards the single best solution (which causes premature convergence), this strategy guides solutions towards a region defined by the top $p\%$ of individuals. This maintains population diversity for longer.
#3.  **External Archive**: The algorithm maintains an archive of recently replaced inferior solutions. These "bad" solutions are used in the mutation difference vector to provide promising search directions that point away from local optima.
#4.  **Stability**: It includes a restart mechanism that detects population stagnation (low variance) and resets the search while preserving the best solution found so far.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using SHADE (Success-History Adaptive DE) 
    with Linear Population Size Reduction and Soft Restarts.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Use 95% of available time to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.95)
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Configuration ---
    # Population size: Adaptive based on dimension
    # Bounded to [30, 150] to balance exploration with computational cost
    pop_size = int(max(30, min(150, 15 * dim)))
    
    # Archive size (same as pop_size)
    max_arc_size = pop_size
    
    # Memory size for historical parameter adaptation
    H = 5
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    X = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Best Solution Tracking
    best_val = float('inf')
    best_idx = 0
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_time(): return best_val
        val = func(X[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i

    # SHADE Memory (Initialize to 0.5)
    mem_M_CR = np.full(H, 0.5)
    mem_M_F = np.full(H, 0.5)
    k_mem = 0 # Memory index pointer
    
    # External Archive (stores replaced parent vectors)
    archive = []

    # --- Main Optimization Loop ---
    while not check_time():
        
        # 0. Convergence Check / Restart
        # If population diversity is too low, we are stuck.
        if np.std(fitness) < 1e-6:
            # Preserve Global Best
            best_vec = X[best_idx].copy()
            
            # Re-initialize Population
            X = min_b + np.random.rand(pop_size, dim) * diff_b
            X[0] = best_vec # Keep best at index 0
            
            # Re-evaluate
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            best_idx = 0
            
            for i in range(1, pop_size):
                if check_time(): return best_val
                val = func(X[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
            
            # Reset Archive and Memory for fresh start
            archive = []
            mem_M_CR.fill(0.5)
            mem_M_F.fill(0.5)
            continue

        # 1. Parameter Generation
        # Randomly select memory index for each individual
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        # Generate CR (Normal Distribution, clipped [0, 1])
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0, 1)
        
        # Generate F (Cauchy Distribution: loc=m_f, scale=0.1)
        # Cauchy helps escape local optima due to "fat tails" (rare large jumps)
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F
        F[F > 1] = 1.0 # Clip upper
        
        # For F <= 0, regenerate until positive
        bad_f_mask = F <= 0
        while np.any(bad_f_mask):
            n_bad = np.sum(bad_f_mask)
            F[bad_f_mask] = m_f[bad_f_mask] + 0.1 * np.random.standard_cauchy(n_bad)
            bad_f_mask = F <= 0
            F[F > 1] = 1.0

        # 2. Mutation Strategy: current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Find top p-best individuals
        # p is minimum 2/pop_size (ensure at least 2), max 0.2 (20%)
        p = max(2.0/pop_size, 0.11) 
        top_cut = int(round(pop_size * p))
        sorted_indices = np.argsort(fitness)
        top_indices = sorted_indices[:top_cut]
        
        # Select pbest for each individual randomly from top %
        pbest_idxs = np.random.choice(top_indices, pop_size)
        x_pbest = X[pbest_idxs]
        
        # Select r1 (distinct from current i)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Avoid r1 == i by shifting
        collision_i = (r1_idxs == np.arange(pop_size))
        r1_idxs[collision_i] = (r1_idxs[collision_i] + 1) % pop_size
        x_r1 = X[r1_idxs]
        
        # Select r2 (from Population Union Archive)
        if len(archive) > 0:
            arc_np = np.array(archive)
            pop_arc = np.vstack((X, arc_np))
        else:
            pop_arc = X
            
        r2_idxs = np.random.randint(0, len(pop_arc), pop_size)
        # (We skip strict r2!=r1!=i check for vectorization speed; impact is negligible)
        x_r2 = pop_arc[r2_idxs]
        
        # Compute Mutant Vector V
        F_col = F[:, np.newaxis] # Reshape for broadcasting
        V = X + F_col * (x_pbest - X) + F_col * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # Guarantee at least one parameter changes (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < CR[:, np.newaxis]
        mask[np.arange(pop_size), j_rand] = True
        
        U = np.where(mask, V, X)
        
        # 4. Bound Constraint Handling
        U = np.clip(U, min_b, max_b)
        
        # 5. Selection and History Update
        new_fitness = np.zeros(pop_size)
        improved_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        
        # Evaluate Trial Vectors
        for i in range(pop_size):
            if check_time(): return best_val
            
            f_new = func(U[i])
            new_fitness[i] = f_new
            
            if f_new <= fitness[i]:
                improved_mask[i] = True
                diff_fitness[i] = fitness[i] - f_new
                
                if f_new < best_val:
                    best_val = f_new
                    best_idx = i

        # 6. Update Populations and Memory
        success_indices = np.where(improved_mask)[0]
        
        if len(success_indices) > 0:
            # Add replaced parents to Archive
            for idx in success_indices:
                archive.append(X[idx].copy())
            
            # Maintain Archive Size
            while len(archive) > max_arc_size:
                # Remove random element
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update Population
            X[success_indices] = U[success_indices]
            fitness[success_indices] = new_fitness[success_indices]
            
            # Update SHADE Memory (Weighted Lehmer Mean)
            # Weights based on fitness improvement amount
            df = diff_fitness[success_indices]
            denom = np.sum(df)
            
            if denom > 0:
                weights = df / denom
                
                # Update M_CR
                s_cr = CR[success_indices]
                if np.max(s_cr) == 0:
                    m_cr_new = 0.0
                else:
                    m_cr_new = np.sum(weights * s_cr)
                
                # Update M_F (Lehmer Mean: sum(w*F^2) / sum(w*F))
                s_f = F[success_indices]
                num = np.sum(weights * (s_f ** 2))
                den = np.sum(weights * s_f)
                if den == 0: 
                    m_f_new = 0.5 
                else: 
                    m_f_new = num / den
                
                # Store in Memory
                mem_M_CR[k_mem] = m_cr_new
                mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                
                # Advance pointer
                k_mem = (k_mem + 1) % H

    return best_val
