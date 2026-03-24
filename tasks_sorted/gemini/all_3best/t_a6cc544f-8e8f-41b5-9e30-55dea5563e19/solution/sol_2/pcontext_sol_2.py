#The following algorithm implements a **JADE (Adaptive Differential Evolution with Archive)** strategy.
#
#**Key Improvements over previous attempts:**
#1.  **Self-Adaptation of Parameters**: Instead of fixed Mutation ($F$) and Crossover ($CR$) rates, this algorithm adapts them based on the success history of the search. This removes the need for manual tuning and allows the algorithm to switch between exploration (high $F$) and exploitation (low $F$) dynamically.
#2.  **Current-to-pbest Mutation**: This strategy balances convergence speed and diversity by guiding individuals towards the top $p\%$ best solutions found so far, rather than just the single best (which causes premature convergence) or random ones (which is too slow).
#3.  **External Archive**: An archive of recently inferior solutions is maintained. The difference vectors can use these archived solutions, which preserves population diversity and prevents the search from collapsing into a single basin of attraction too quickly.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using JADE (Adaptive Differential Evolution) with Archive.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------
    # Configuration
    # -------------------------------
    # Population size: Capped to ensure generations run in limited time.
    # A size between 20 and 60 is generally robust for various dimensions under time constraints.
    pop_size = max(20, min(10 * dim, 60))
    
    # JADE Adaptive Parameters
    mu_cr = 0.5    # Initial Mean Crossover Rate
    mu_f = 0.5     # Initial Mean Mutation Factor
    c = 0.1        # Adaptation learning rate
    p = 0.05       # Top percentage for pbest selection
    
    # -------------------------------
    # Initialization
    # -------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    
    # Archive to store inferior solutions for diversity (prevents stagnation)
    archive = []
    
    # -------------------------------
    # Initial Evaluation
    # -------------------------------
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # -------------------------------
    # Main Optimization Loop
    # -------------------------------
    while True:
        # Time check at start of generation
        if (datetime.now() - start_time) >= time_limit:
            return best_val
        
        # Sort population to find top p% individuals (pbest)
        sorted_idx = np.argsort(fitness)
        num_pbest = max(1, int(pop_size * p))
        pbest_indices = sorted_idx[:num_pbest]
        
        # Lists to store successful parameter values for this generation
        succ_cr = []
        succ_f = []
        
        # Iterate through population
        for i in range(pop_size):
            # Strict time check before potential heavy func eval
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            # 1. Parameter Generation
            # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
            cr = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1), clipped to [0, 1]. Regenerate if <= 0.
            while True:
                f = mu_f + 0.1 * np.random.standard_cauchy()
                if f > 0:
                    if f > 1: f = 1.0
                    break
            
            # 2. Mutation: DE/current-to-pbest/1/bin with Archive
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select pbest
            idx_pbest = np.random.choice(pbest_indices)
            x_pbest = pop[idx_pbest]
            
            # Select r1 (distinct from i)
            while True:
                r1 = np.random.randint(0, pop_size)
                if r1 != i:
                    break
            x_r1 = pop[r1]
            
            # Select r2 (distinct from i, r1) from Population + Archive
            # Archive logic: indices >= pop_size refer to archive
            count_archive = len(archive)
            limit = pop_size + count_archive
            
            while True:
                r2 = np.random.randint(0, limit)
                if r2 != i and r2 != r1:
                    break
            
            if r2 < pop_size:
                x_r2 = pop[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # Compute difference vectors
            mutant = pop[i] + f * (x_pbest - pop[i]) + f * (x_r1 - x_r2)
            
            # 3. Crossover
            # Pick random index to ensure at least one change
            j_rand = np.random.randint(0, dim)
            cross_mask = np.random.rand(dim) < cr
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop[i])
            
            # 4. Bounds Check
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Evaluation
            f_trial = func(trial)
            
            # 6. Selection
            # Greedy selection: if trial is better or equal, replace parent
            if f_trial <= fitness[i]:
                # If strictly better, we consider it a successful parameter set
                if f_trial < fitness[i]:
                    succ_cr.append(cr)
                    succ_f.append(f)
                    # Add parent to archive before replacement
                    archive.append(pop[i].copy())
                
                pop[i] = trial
                fitness[i] = f_trial
                
                if f_trial < best_val:
                    best_val = f_trial
        
        # 7. Archive Maintenance
        # Remove random individuals if archive exceeds population size
        while len(archive) > pop_size:
            idx_rem = np.random.randint(0, len(archive))
            archive.pop(idx_rem)
            
        # 8. Parameter Adaptation
        if len(succ_cr) > 0:
            mu_cr = (1 - c) * mu_cr + c * np.mean(succ_cr)
            
            sf = np.array(succ_f)
            # Lehmer mean for F
            mu_f = (1 - c) * mu_f + c * (np.sum(sf**2) / np.sum(sf))

    return best_val
