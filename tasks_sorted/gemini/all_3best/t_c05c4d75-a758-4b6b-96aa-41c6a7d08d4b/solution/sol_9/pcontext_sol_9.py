#The following algorithm implements **iL-SHADE-R (Improved L-SHADE with Restart and Midpoint Boundary Handling)**. 
#
#This algorithm refines the best-performing approach (L-SHADE-R) by incorporating **Midpoint Target Boundary Handling**. Instead of simply clipping solutions to the bounds (which can reduce diversity by stacking solutions on the edges) or reflecting them (which can over-disperse), this method places out-of-bound solutions at the midpoint between their parent and the bound. This preserves the evolutionary direction while keeping the solution feasible. Combined with Linear Population Size Reduction (LPSR) and Success-History Adaptation, this offers robust convergence.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using iL-SHADE-R:
    Improved Linear Success-History Adaptive Differential Evolution with Restart
    and Midpoint Target Boundary Handling.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    # End time buffer to ensure return before hard timeout
    end_time = start_time + time_limit - timedelta(seconds=0.05)

    # --- Hyperparameters ---
    # Population Size: Linear Reduction (LPSR)
    # Start with a robust size for exploration (approx 20 * dim)
    min_pop_init = 20
    max_pop_init = 250
    init_pop_size = int(np.clip(20 * dim, min_pop_init, max_pop_init))
    min_pop_size = 4  # Aggressive reduction for end-stage exploitation
    
    current_pop_size = init_pop_size
    
    # SHADE Memory Parameters
    H = 5  # History memory size
    mem_f = np.full(H, 0.5)   # Initial Mutation Factor memory
    mem_cr = np.full(H, 0.5)  # Initial Crossover Rate memory
    k_mem = 0                 # Memory index pointer
    
    # Restart Triggers
    stall_limit = 20
    tol_std = 1e-9
    
    # --- Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Archive for diversity (stores inferior solutions replaced by offspring)
    # Size limited to current population size
    archive = []
    
    # --- Initialization ---
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(current_pop_size):
        if datetime.now() >= end_time:
            return best_fitness if best_fitness != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # Sort population by fitness (required for p-best selection)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    stall_counter = 0
    
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        if now >= end_time:
            return best_fitness
            
        elapsed = (now - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # 1. Linear Population Size Reduction (LPSR)
        # N_g = round( (N_min - N_init) * progress + N_init )
        target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        if target_size < min_pop_size: target_size = min_pop_size
        
        if current_pop_size > target_size:
            current_pop_size = target_size
            # Keep best individuals (population is sorted)
            pop = pop[:current_pop_size]
            fitness = fitness[:current_pop_size]
            
            # Resize Archive to match current population size
            if len(archive) > current_pop_size:
                del archive[current_pop_size:]
                
        # 2. Adaptive Parameter Generation (SHADE)
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, current_pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(M_CR, 0.1), clipped to [0, 1]
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F ~ Cauchy(M_F, 0.1)
        # Resample if F <= 0, Clip if F > 1
        F = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
        retry_mask = F <= 0
        while np.any(retry_mask):
            F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
            retry_mask = F <= 0
        F[F > 1] = 1.0
        
        # 3. Mutation: current-to-pbest/1
        # p-value decreases linearly from 0.2 (exploration) to 0.05 (exploitation)
        p_val = 0.2 - 0.15 * progress
        p_val = max(p_val, 2.0 / current_pop_size)
        
        num_pbest = int(max(2, p_val * current_pop_size))
        
        # Select X_pbest (randomly from top num_pbest)
        pbest_indices = np.random.randint(0, num_pbest, current_pop_size)
        X_pbest = pop[pbest_indices]
        
        # Select X_r1 (distinct from i)
        r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
        conflict_r1 = (r1_indices == np.arange(current_pop_size))
        while np.any(conflict_r1):
            r1_indices[conflict_r1] = np.random.randint(0, current_pop_size, np.sum(conflict_r1))
            conflict_r1 = (r1_indices == np.arange(current_pop_size))
        X_r1 = pop[r1_indices]
        
        # Select X_r2 (distinct from i and r1, from Union(Pop, Archive))
        if len(archive) > 0:
            pool = np.vstack((pop, np.array(archive)))
        else:
            pool = pop
        len_pool = len(pool)
        
        r2_indices = np.random.randint(0, len_pool, current_pop_size)
        # Conflict check for r2
        conflict_r2 = (r2_indices < current_pop_size) & ((r2_indices == np.arange(current_pop_size)) | (r2_indices == r1_indices))
        while np.any(conflict_r2):
            r2_indices[conflict_r2] = np.random.randint(0, len_pool, np.sum(conflict_r2))
            conflict_r2 = (r2_indices < current_pop_size) & ((r2_indices == np.arange(current_pop_size)) | (r2_indices == r1_indices))
        X_r2 = pool[r2_indices]
        
        # Compute Mutant Vectors
        F_col = F[:, None]
        mutant = pop + F_col * (X_pbest - pop) + F_col * (X_r1 - X_r2)
        
        # 4. Boundary Handling: Midpoint Target
        # If a mutant is out of bounds, place it halfway between parent and bound.
        # This preserves evolutionary direction better than clipping.
        mask_l = mutant < min_b
        if np.any(mask_l):
            mutant[mask_l] = (pop[mask_l] + min_b[mask_l]) / 2.0
        
        mask_u = mutant > max_b
        if np.any(mask_u):
            mutant[mask_u] = (pop[mask_u] + max_b[mask_u]) / 2.0
            
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(current_pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        # Ensure at least one dimension is inherited from mutant
        j_rand = np.random.randint(0, dim, current_pop_size)
        cross_mask[np.arange(current_pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 6. Selection and Memory Update
        succ_F = []
        succ_CR = []
        diff_fitness = []
        new_archive_cands = []
        gen_improved = False
        
        for i in range(current_pop_size):
            if datetime.now() >= end_time:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Successful update
                if f_trial < fitness[i]:
                    new_archive_cands.append(pop[i].copy())
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    diff_fitness.append(fitness[i] - f_trial)
                    gen_improved = True
                
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
        
        # Update Archive
        for cand in new_archive_cands:
            if len(archive) < current_pop_size:
                archive.append(cand)
            else:
                if len(archive) > 0:
                    ridx = np.random.randint(0, len(archive))
                    archive[ridx] = cand
                    
        # Update History Memory (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                
                # Lehmer mean for F
                s_f = np.array(succ_F)
                sum_w_sf = np.sum(w * s_f)
                if abs(sum_w_sf) > 1e-12:
                    mean_f = np.sum(w * (s_f ** 2)) / sum_w_sf
                    mem_f[k_mem] = mean_f
                
                # Weighted arithmetic mean for CR
                s_cr = np.array(succ_CR)
                mean_cr = np.sum(w * s_cr)
                mem_cr[k_mem] = mean_cr
                
                k_mem = (k_mem + 1) % H
                
        # Resort population
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # 7. Restart Mechanism
        if gen_improved:
            stall_counter = 0
        else:
            stall_counter += 1
            
        std_fit = np.std(fitness)
        
        # Restart if stalled or converged, provided enough time remains (> 10%)
        if (stall_counter >= stall_limit or std_fit < tol_std) and progress < 0.90:
            stall_counter = 0
            
            # Re-initialize population
            pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
            fitness = np.full(current_pop_size, float('inf'))
            
            # Elitism: Inject Global Best
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Memory and Archive
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            archive = []
            
            # Evaluate new population (skip elite at index 0)
            for i in range(1, current_pop_size):
                if datetime.now() >= end_time:
                    return best_fitness
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            # Resort immediately
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

    return best_fitness
