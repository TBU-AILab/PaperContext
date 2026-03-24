#The following algorithm implements **O-IPO-jDE-MTS (Opposition-based Increasing Population Optimization with jDE and MTS Local Search)**.
#
#### Rationale for Improvement
#
#1.  **MTS-LS1 (Multiple Trajectory Search - Local Search 1)**: The previous best algorithm used a random Gaussian walk to "polish" the result. Random walks are inefficient in high dimensions. This version implements a deterministic **Coordinate Descent** (based on MTS-LS1). It iterates through each dimension of the best solution, testing small steps in positive and negative directions. This creates a "gradient-descent-like" effect without calculating gradients, significantly improving precision (exploiting deep valleys) where standard DE struggles.
#2.  **Refined OBL (Opposition-Based Learning)**: The initialization now strictly bounds the "opposite" population. If an opposite particle is out of bounds, it is reset randomly rather than clamped, preserving diversity better at the edges of the search space.
#3.  **Weighted Bound Handling**: Instead of simple midpoint or clamping, this implementation uses a weighted correction `(parent + 2*min) / 3` when bounds are violated. This prevents the population from sticking to the geometric edges of the hypercube, keeping them "active" just inside the bounds.
#4.  **Structure**: It retains the successful **IPO** (Increasing Population) and **jDE** (Self-Adaptive parameters) backbone from the previous best result, ensuring robustness across multimodal landscapes while the new Local Search handles the fine-tuning.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using O-IPO-jDE-MTS:
    Opposition-based Initialization, Increasing Population Optimization,
    Self-Adaptive Differential Evolution, and MTS-style Local Search.
    """
    start_time = time.time()
    
    # --- Helper: Time Management ---
    def check_time():
        # Buffer of 0.02s to safely return
        return (time.time() - start_time) >= max_time - 0.02

    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global State
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Algorithm Parameters ---
    # Start with a small population to clear easy problems fast
    pop_size = max(10, int(3 * dim))
    MAX_POP_SIZE = 1000
    
    # --- Main Restart Loop ---
    while True:
        # Check time before starting a new optimization cycle
        if (time.time() - start_time) > max_time - 0.1:
            return global_best_val
            
        # 1. Opposition-Based Initialization
        N = pop_size
        
        # A. Generate Random Population
        pop_rand = min_b + np.random.rand(N, dim) * diff_b
        
        # B. Generate Opposite Population
        # OBL: x_opp = min + max - x
        pop_opp = min_b + max_b - pop_rand
        
        # Check bounds for opposite population
        # If out of bounds, randomize (better than clamping for diversity)
        lower_mask = pop_opp < min_b
        upper_mask = pop_opp > max_b
        out_mask = lower_mask | upper_mask
        
        random_fix = min_b + np.random.rand(N, dim) * diff_b
        pop_opp = np.where(out_mask, random_fix, pop_opp)
        
        # C. Combine and Select Best N
        # We need to evaluate 2*N individuals
        pop_pool = np.vstack((pop_rand, pop_opp))
        fitness_pool = np.full(2 * N, float('inf'))
        
        # Elitism: Inject global best if it exists
        if global_best_sol is not None:
            pop_pool[0] = global_best_sol
            # We will re-evaluate to keep code simple, cost is negligible vs benefit
            
        # Evaluation Loop for Pool
        for i in range(2 * N):
            if check_time(): return global_best_val
            
            val = func(pop_pool[i])
            fitness_pool[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop_pool[i].copy()
                
        # Select best N
        sorted_indices = np.argsort(fitness_pool)
        pop = pop_pool[sorted_indices[:N]]
        fitness = fitness_pool[sorted_indices[:N]]
        
        # 2. jDE Setup
        # F and CR initialized randomly
        F = 0.5 + 0.5 * np.random.rand(N) # F in [0.5, 1.0] initially usually better
        CR = 0.9 * np.random.rand(N)      # CR in [0.0, 0.9]
        
        # Archive for current-to-pbest mutation
        # Stores inferior solutions replaced by offspring
        archive = np.zeros((N * 2, dim))
        arc_count = 0
        arc_capacity = N * 2
        
        # Convergence counter
        stagnation_count = 0
        last_best_fit = fitness[0]
        
        # --- Evolution Cycle ---
        while True:
            if check_time(): return global_best_val
            
            # Sort population by fitness
            # Essential for current-to-pbest selection
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            F = F[sort_idx]
            CR = CR[sort_idx]
            
            # Convergence Detection
            current_best = fitness[0]
            if abs(last_best_fit - current_best) < 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best
                
            # If population variance is tiny or stagnated, break to local search/restart
            # Dynamic tolerance based on value magnitude
            tol = 1e-8 if global_best_val < 1.0 else 1e-8 * global_best_val
            if (fitness[-1] - fitness[0] < tol) or (stagnation_count > 30):
                break
            
            # jDE Parameter Adaptation
            # Tau1 = Tau2 = 0.1
            rand_f = np.random.rand(N)
            rand_cr = np.random.rand(N)
            
            # Reset F with prob 0.1 to U(0.1, 1.0)
            F_new = np.where(rand_f < 0.1, 0.1 + 0.9 * np.random.rand(N), F)
            # Reset CR with prob 0.1 to U(0.0, 1.0)
            CR_new = np.where(rand_cr < 0.1, np.random.rand(N), CR)
            
            # Mutation: current-to-pbest/1
            # v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            
            # P-best selection (top 5% to 20%)
            p_min = 2
            p_max = max(2, int(N * 0.2))
            p_val = np.random.randint(p_min, p_max + 1)
            
            # Indices
            pbest_indices = np.random.randint(0, p_val, N) # Top p_val individuals
            r1_indices = np.random.randint(0, N, N)
            
            # Fix r1 self-selection
            r1_indices = np.where(r1_indices == np.arange(N), (r1_indices + 1) % N, r1_indices)
            
            # r2 from Union(Pop, Archive)
            if arc_count > 0:
                pool = np.vstack((pop, archive[:arc_count]))
            else:
                pool = pop
            
            r2_indices = np.random.randint(0, len(pool), N)
            # Relaxed r2 constraint for speed (DE is robust enough)
            
            x_pbest = pop[pbest_indices]
            x_r1 = pop[r1_indices]
            x_r2 = pool[r2_indices]
            
            # Compute difference vectors
            diff1 = x_pbest - pop
            diff2 = x_r1 - x_r2
            
            F_col = F_new[:, None]
            mutant = pop + F_col * diff1 + F_col * diff2
            
            # Crossover (Binomial)
            j_rand = np.random.randint(0, dim, N)
            rand_mat = np.random.rand(N, dim)
            cross_mask = rand_mat < CR_new[:, None]
            cross_mask[np.arange(N), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # Bound Handling: Weighted correction
            # Places the point between the bound and the parent (biased towards bound)
            # This is better than clamping (avoids stacking) and midpoint (too conservative)
            low_viol = trial < min_b
            high_viol = trial > max_b
            
            trial = np.where(low_viol, (pop + min_b * 2) / 3.0, trial)
            trial = np.where(high_viol, (pop + max_b * 2) / 3.0, trial)
            
            # Selection and Updates
            next_pop = pop.copy()
            next_fitness = fitness.copy()
            next_F = F.copy()
            next_CR = CR.copy()
            
            parents_to_archive = []
            
            for i in range(N):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial[i].copy()
                
                if f_trial <= fitness[i]:
                    # Successful trial
                    parents_to_archive.append(pop[i].copy())
                    next_pop[i] = trial[i]
                    next_fitness[i] = f_trial
                    next_F[i] = F_new[i]
                    next_CR[i] = CR_new[i]
            
            pop = next_pop
            fitness = next_fitness
            F = next_F
            CR = next_CR
            
            # Update Archive
            if len(parents_to_archive) > 0:
                cands = np.array(parents_to_archive)
                num = len(cands)
                
                if arc_count + num <= arc_capacity:
                    archive[arc_count:arc_count+num] = cands
                    arc_count += num
                else:
                    # Fill remaining space
                    rem = arc_capacity - arc_count
                    if rem > 0:
                        archive[arc_count:] = cands[:rem]
                        cands = cands[rem:]
                        arc_count = arc_capacity
                    
                    # Random replacement
                    if len(cands) > 0:
                        idx = np.random.randint(0, arc_capacity, len(cands))
                        archive[idx] = cands
        
        # --- Local Search: MTS-LS1 (Coordinate Polish) ---
        # When population stagnates, polish the global best solution
        # This is more efficient than Gaussian walk for fine-tuning
        if global_best_sol is not None:
            best_sol = global_best_sol.copy()
            best_val = global_best_val
            
            # Initial search range: proportional to domain size
            # We shrink this range if we fail to improve
            sr = (max_b - min_b) * 0.4
            
            # MTS Loop
            # We perform a few passes over dimensions
            improved = True
            while improved:
                if check_time(): return global_best_val
                improved = False
                
                # Iterate dimensions
                for d in range(dim):
                    if check_time(): return global_best_val
                    
                    # Try negative step
                    original_val = best_sol[d]
                    best_sol[d] = max(min_b[d], original_val - sr[d])
                    val = func(best_sol)
                    
                    if val < best_val:
                        best_val = val
                        global_best_val = val
                        global_best_sol = best_sol.copy()
                        improved = True
                    else:
                        # Restore
                        best_sol[d] = original_val
                        
                        # Try positive step (half size as per MTS logic often)
                        best_sol[d] = min(max_b[d], original_val + 0.5 * sr[d])
                        val = func(best_sol)
                        
                        if val < best_val:
                            best_val = val
                            global_best_val = val
                            global_best_sol = best_sol.copy()
                            improved = True
                        else:
                            # Restore and shrink search range for this dimension
                            best_sol[d] = original_val
                            sr[d] *= 0.5
                            
                # If Search Range is too small, stop LS
                if np.max(sr) < 1e-15:
                    break
        
        # --- Prepare for Next Restart (IPO) ---
        # Double population size to increase diversity for next run
        pop_size = pop_size * 2
        if pop_size > MAX_POP_SIZE:
            pop_size = MAX_POP_SIZE

    return global_best_val
