#The following algorithm implements **L-SHADE** (Linear Population Size Reduction Success-History based Adaptive Differential Evolution). This is a state-of-the-art improvement over standard DE and the previously generated algorithm.
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR)**: The population size starts large to explore the search space widely and linearly decreases over time. This forces the algorithm to shift from exploration to exploitation naturally, converging rapidly towards the end of the available time.
#2.  **History-Based Parameter Adaptation**: Instead of sampling from fixed distributions, the algorithm "learns" the optimal Mutation Factor ($F$) and Crossover Rate ($CR$) by maintaining a memory of successful values from previous generations (weighted by fitness improvement).
#3.  **External Archive**: It maintains an archive of recently discarded inferior solutions. This preserves diversity in the mutation equation (`current-to-pbest/1`), preventing premature convergence.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History Adaptive Differential Evolution).
    
    Features:
    - Adaptive F and CR parameters based on successful history.
    - Linearly reducing population size to force convergence.
    - 'current-to-pbest/1' mutation strategy with external archive.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Start large for exploration, reduce linearly.
    # We cap the initial size to ensure the algorithm gets running 
    # even if dimensions are high or time is short.
    pop_size_init = int(min(300, max(30, 15 * dim)))
    pop_size_min = 4
    
    # Memory for adaptive parameters (History size H=5)
    mem_size = 5
    m_cr = np.full(mem_size, 0.5)
    m_f = np.full(mem_size, 0.5)
    k_mem = 0
    
    # Archive to maintain diversity
    archive = []
    
    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Random initial population
    population = min_b + np.random.rand(pop_size_init, dim) * diff_b
    fitness = np.full(pop_size_init, float('inf'))
    
    best_val = float('inf')
    
    # Evaluate Initial Population
    # We perform a strict time check here to handle very short max_time limits
    for i in range(pop_size_init):
        if (datetime.now() - start_time) >= limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            
    # --- Main Optimization Loop ---
    while (datetime.now() - start_time) < limit:
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress ratio (0.0 to 1.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(1.0, elapsed / max_time)
        
        # Calculate target population size based on time remaining
        target_size = int(round(pop_size_init - progress * (pop_size_init - pop_size_min)))
        target_size = max(pop_size_min, target_size)
        
        curr_size = len(population)
        
        # If population needs reduction, remove worst individuals
        if curr_size > target_size:
            idxs = np.argsort(fitness)
            population = population[idxs[:target_size]]
            fitness = fitness[idxs[:target_size]]
            curr_size = target_size
            
        # 2. Adaptive Parameters Generation
        # Assign a random memory index to each individual
        r_idxs = np.random.randint(0, mem_size, curr_size)
        
        # Generate CR from Normal Distribution based on memory
        cr = np.random.normal(m_cr[r_idxs], 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F from Cauchy Distribution based on memory
        # Cauchy: location + scale * standard_cauchy
        f = m_f[r_idxs] + 0.1 * np.random.standard_cauchy(curr_size)
        
        # Repair F values (F > 0 is required)
        retry = f <= 0
        while np.any(retry):
            f[retry] = m_f[r_idxs[retry]] + 0.1 * np.random.standard_cauchy(np.sum(retry))
            retry = f <= 0
        f = np.clip(f, 0.0, 1.0) # Clip upper bound to 1.0
        
        # 3. Mutation: current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Select p-best (top 11% individuals)
        p_count = max(2, int(0.11 * curr_size))
        sorted_idxs = np.argsort(fitness)
        pbest_idxs = np.random.choice(sorted_idxs[:p_count], curr_size)
        x_pbest = population[pbest_idxs]
        
        # Select r1 (random from population, distinct from i)
        r1_idxs = np.random.randint(0, curr_size, curr_size)
        for i in range(curr_size):
            while r1_idxs[i] == i:
                r1_idxs[i] = np.random.randint(0, curr_size)
        x_r1 = population[r1_idxs]
        
        # Select r2 (random from Population UNION Archive, distinct from i and r1)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((population, archive_np))
        else:
            union_pop = population
            
        union_size = len(union_pop)
        r2_idxs = np.random.randint(0, union_size, curr_size)
        for i in range(curr_size):
            while r2_idxs[i] == i or r2_idxs[i] == r1_idxs[i]:
                r2_idxs[i] = np.random.randint(0, union_size)
        x_r2 = union_pop[r2_idxs]
        
        # Compute Mutant Vectors
        f_col = f[:, np.newaxis]
        mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial) & Bound Constraints
        rand_v = np.random.rand(curr_size, dim)
        mask = rand_v < cr[:, np.newaxis]
        
        # Ensure at least one dimension comes from the mutant
        j_rand = np.random.randint(0, dim, curr_size)
        mask[np.arange(curr_size), j_rand] = True
        
        trial = np.where(mask, mutant, population)
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Evaluation and Selection
        success_f = []
        success_cr = []
        df = [] # Fitness improvements
        
        for i in range(curr_size):
            # Strict time check inside the loop
            if (datetime.now() - start_time) >= limit:
                return best_val
            
            val = func(trial[i])
            
            # Selection: Greedy (<= allows moving on flat landscapes)
            if val <= fitness[i]:
                # If strictly better, store for memory update
                if val < fitness[i]:
                    archive.append(population[i].copy())
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    df.append(fitness[i] - val)
                    
                population[i] = trial[i]
                fitness[i] = val
                
                if val < best_val:
                    best_val = val
                    
        # 6. Archive Maintenance
        # Cap archive size to current population size
        while len(archive) > curr_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 7. Memory Update (SHADE logic)
        if len(df) > 0:
            w = np.array(df)
            w = w / np.sum(w) # Normalize weights
            sf = np.array(success_f)
            scr = np.array(success_cr)
            
            # Lehmer Mean for F
            mean_f = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-9)
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(w * scr)
            
            # Update memory slot
            m_f[k_mem] = mean_f
            m_cr[k_mem] = mean_cr
            k_mem = (k_mem + 1) % mem_size
            
    return best_val
