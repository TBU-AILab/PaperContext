#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with Restart**.
#
#**Improvements over previous attempts:**
#1.  **Adaptive Parameters (History-Based)**: Unlike jDE (which resets or randomly perturbs parameters), SHADE learns from successful updates. It maintains a memory ($M_{CR}, M_F$) of parameter values that improved fitness, biasing future parameter generation towards successful regions.
#2.  **External Archive**: It keeps a history of recently replaced inferior solutions. This archive is used in the mutation operator (`current-to-pbest/1`) to maintain diversity and prevent premature convergence.
#3.  **Robust Restart**: If the population standard deviation drops below a threshold (stagnation), the algorithm preserves the best individual and scatters the rest of the population to explore new basins of attraction.
#
#This method is state-of-the-art for black-box optimization in continuous domains without gradients.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) with Restart.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: clamped to balance exploration and iteration speed
    pop_size = int(10 * dim)
    pop_size = max(30, min(pop_size, 100))
    
    # Archive parameters
    arc_rate = 2.0
    max_arc_size = int(pop_size * arc_rate)
    
    # Memory size for adaptive parameters
    H = 5
    M_CR = np.full(H, 0.5)
    M_F = np.full(H, 0.5)
    k_mem = 0
    
    # Bounds preparation
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Helper for time check
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        if is_time_up(): return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val

    archive = []
    
    # --- Main Loop ---
    while True:
        if is_time_up(): return best_val
        
        # Sort population by fitness for 'current-to-pbest' mutation
        sorted_indices = np.argsort(fitness)
        
        # Lists to store successful parameters for memory update
        S_CR = []
        S_F = []
        S_df = []
        
        new_pop = np.zeros((pop_size, dim))
        new_fitness = np.zeros(pop_size)
        
        # Iterate over population
        for i in range(pop_size):
            if is_time_up(): return best_val
            
            target = pop[i]
            
            # 1. Parameter Generation from Memory
            r_idx = np.random.randint(0, H)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # Generate CR: Normal(mu_cr, 0.1), clipped to [0, 1]
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(mu_f, 0.1)
            # If F > 1, clamp to 1. If F <= 0, regenerate.
            while True:
                f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                if f > 0:
                    break
            if f > 1: f = 1.0
            
            # 2. Mutation: current-to-pbest/1 (with Archive)
            # Select p-best: random from top p% (p in [2/N, 0.2])
            p = np.random.uniform(2/pop_size, 0.2)
            top_p_cnt = int(max(2, pop_size * p))
            pbest_idx = sorted_indices[np.random.randint(0, top_p_cnt)]
            x_pbest = pop[pbest_idx]
            
            # Select r1: random from pop, distinct from i
            while True:
                r1_idx = np.random.randint(0, pop_size)
                if r1_idx != i:
                    break
            x_r1 = pop[r1_idx]
            
            # Select r2: random from pop U archive, distinct from i, r1
            n_arc = len(archive)
            total_pool = pop_size + n_arc
            while True:
                r2_idx = np.random.randint(0, total_pool)
                if r2_idx < pop_size:
                    if r2_idx != i and r2_idx != r1_idx:
                        x_r2 = pop[r2_idx]
                        break
                else:
                    x_r2 = archive[r2_idx - pop_size]
                    break
            
            # Create Mutant vector
            mutant = target + f * (x_pbest - target) + f * (x_r1 - x_r2)
            
            # 3. Crossover: Binomial
            cross_mask = np.random.rand(dim) < cr
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True # Ensure at least one dimension changed
            
            trial = np.where(cross_mask, mutant, target)
            
            # Bound Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                # Update Best
                if f_trial < best_val:
                    best_val = f_trial
                
                # Record success
                S_CR.append(cr)
                S_F.append(f)
                S_df.append(fitness[i] - f_trial)
                
                # Add replaced parent to archive
                archive.append(target.copy())
            else:
                new_pop[i] = target
                new_fitness[i] = fitness[i]
                
        # --- End of Generation Updates ---
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size
        while len(archive) > max_arc_size:
            idx_to_del = np.random.randint(0, len(archive))
            archive.pop(idx_to_del)
            
        # Update Historical Memory (Weighted Lehmer Mean)
        if len(S_df) > 0:
            w = np.array(S_df)
            total_imp = np.sum(w)
            if total_imp > 0:
                w = w / total_imp
                
                # Update M_CR
                m_cr_new = np.sum(w * np.array(S_CR))
                M_CR[k_mem] = m_cr_new
                
                # Update M_F
                sf = np.array(S_F)
                num = np.sum(w * (sf ** 2))
                den = np.sum(w * sf)
                m_f_new = num / den if den > 0 else 0.5
                M_F[k_mem] = m_f_new
                
                k_mem = (k_mem + 1) % H
        
        # --- Restart Mechanism ---
        # If population diversity is extremely low, restart to find other optima
        if np.std(fitness) < 1e-6:
            # Preserve best
            best_vec = pop[np.argmin(fitness)].copy()
            
            # Re-initialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_vec # Keep elite
            
            # Reset Archive and Memory to adapt to new region or search phase
            archive = []
            M_CR.fill(0.5)
            M_F.fill(0.5)
            
            # Re-evaluate new individuals
            fitness[0] = best_val
            for i in range(1, pop_size):
                if is_time_up(): return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
