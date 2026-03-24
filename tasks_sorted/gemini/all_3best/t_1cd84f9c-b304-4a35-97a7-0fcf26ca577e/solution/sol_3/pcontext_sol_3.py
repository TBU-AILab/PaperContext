#The previous algorithms attempted to solve the optimization problem using various Differential Evolution (DE) strategies. The best result was achieved by a SHADE-inspired algorithm with a restart mechanism. To improve upon this, I propose an enhanced algorithm: **SHADE with Opposition-Based Learning (OBL), Bounce-Back Boundary Handling, and Stagnation Restart**.
#
#### **Improvements Explained:**
#
#1.  **Opposition-Based Learning (OBL):**
#    *   **Why:** Standard random initialization might start the population in a poor region of the search space.
#    *   **How:** During initialization (and restarts), for every random individual generated, we also calculate its "opposite" point in the search space ($x' = min + max - x$). We evaluate both and keep the better one. This significantly increases the probability of finding a good basin of attraction early.
#
#2.  **Bounce-Back Boundary Handling:**
#    *   **Why:** Simple clipping (setting $x$ to the bound if it exceeds it) often causes the population to stick to the edges of the search space, which can be a local optimum trap.
#    *   **How:** Instead of clipping, we set the violating variable to the midpoint between the bound and the parent's previous position. This preserves the search direction while keeping the value valid.
#
#3.  **Refined Stagnation Detection:**
#    *   **Why:** Relying solely on population variance can be slow to trigger a restart if the population converges to a sub-optimal non-zero variance state.
#    *   **How:** We track the number of generations without global fitness improvement (`stagnation_count`). If this exceeds a threshold (e.g., 40 generations) or the population variance drops effectively to zero, a restart is triggered.
#
#4.  **SHADE Parameter Adaptation:**
#    *   Retains the successful history-based parameter adaptation for $F$ (Scaling Factor) and $CR$ (Crossover Rate) to adapt to the function landscape dynamically.
#
#### **Algorithm Code:**
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE with Opposition-Based Learning (OBL) and Restart.
    
    Features:
    - OBL Initialization: Generates opposite points to cover the search space better.
    - SHADE Adaptation: History-based adaptation of F and CR parameters.
    - Bounce-Back Strategy: Handles boundary violations by reflecting into the feasible region 
      rather than clipping, avoiding local optima on boundaries.
    - Stagnation Restart: Restarts the population if variance drops or 
      no improvement is observed.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- 1. Parameters & Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population setup
    # Adaptive size: Balance between exploration (large) and speed (small)
    # Capped at 100 to ensure reasonable generational turnover within limited time
    pop_size = int(max(30, min(100, 18 * dim)))
    
    # SHADE Memory Parameters
    H = 5
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
    archive = []
    
    # Global Best Tracking
    best_ind = None
    best_fit = float('inf')

    # --- 2. Helper Functions ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')

    def get_obl_population(count_to_gen):
        """
        Generates 'count_to_gen' individuals using Opposition-Based Learning.
        It generates random points, computes their opposites, and keeps the best half.
        """
        candidates = []
        cand_fits = []
        
        # 1. Random Generation
        for _ in range(count_to_gen):
            if is_time_up(): break
            x = min_b + np.random.rand(dim) * diff_b
            candidates.append(x)
            cand_fits.append(safe_evaluate(x))
            
        if is_time_up() and not candidates: return [], []

        # 2. Opposition Generation
        # x' = lower + upper - x
        current_len = len(candidates)
        for i in range(current_len):
            if is_time_up(): break
            x = candidates[i]
            op_x = min_b + max_b - x
            # Clip opposite to ensure it is valid before evaluation
            op_x = np.clip(op_x, min_b, max_b)
            
            candidates.append(op_x)
            cand_fits.append(safe_evaluate(op_x))
            
        # Select best 'count_to_gen'
        candidates = np.array(candidates)
        cand_fits = np.array(cand_fits)
        
        if len(candidates) == 0: return [], []
        
        sorted_idx = np.argsort(cand_fits)
        best_n_idx = sorted_idx[:count_to_gen]
        
        return candidates[best_n_idx], cand_fits[best_n_idx]

    # --- 3. Initial Population ---
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    
    init_pop, init_fit = get_obl_population(pop_size)
    
    if len(init_fit) == 0: # Time up before single eval
        return float('inf')
        
    population[:len(init_pop)] = init_pop
    fitness[:len(init_fit)] = init_fit
    
    # Update Best
    min_idx = np.argmin(fitness)
    if fitness[min_idx] < best_fit:
        best_fit = fitness[min_idx]
        best_ind = population[min_idx].copy()
        
    if is_time_up(): return best_fit

    # State for restart logic
    stagnation_count = 0
    last_best_fit = best_fit

    # --- 4. Main Optimization Loop ---
    while not is_time_up():
        
        # A. Check for Restart
        # Conditions: Low variance (convergence) OR High stagnation (no improvement)
        pop_std = np.std(fitness)
        if pop_std < 1e-9 or stagnation_count > 40:
            # Restart: Preserve global best, regenerate the rest using OBL
            indices_needed = pop_size - 1
            new_pop, new_fit = get_obl_population(indices_needed)
            
            if len(new_pop) > 0:
                population[0] = best_ind # Keep elite
                fitness[0] = best_fit
                population[1:1+len(new_pop)] = new_pop
                fitness[1:1+len(new_fit)] = new_fit
            
            # Reset SHADE memory and archive to adapt to new region
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            stagnation_count = 0
            
            if is_time_up(): return best_fit
        
        # B. Parameter Generation (SHADE)
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idx]
        m_f = mem_M_F[r_idx]
        
        # CR: Normal distribution around memory, clipped [0, 1]
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F: Cauchy distribution around memory
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f_vals[f_vals <= 0] = 0.5 # Retry/Fix for non-positive
        f_vals = np.minimum(f_vals, 1.0)
        
        sorted_indices = np.argsort(fitness)
        
        success_sf = []
        success_scr = []
        fit_diffs = []
        
        # Create Union of Population + Archive for mutation source
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
            
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        # C. Evolution Cycle
        for i in range(pop_size):
            if is_time_up(): return best_fit
            
            # 1. Mutation: current-to-pbest/1
            # pbest is one of the top p% individuals
            p = np.random.uniform(2/pop_size, 0.2)
            top_p = int(max(2, p * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, top_p)]
            
            # r1 distinct from i
            r1_idx = i
            while r1_idx == i:
                r1_idx = np.random.randint(0, pop_size)
            
            # r2 distinct from i and r1, from union
            r2_idx = i
            while r2_idx == i or r2_idx == r1_idx:
                r2_idx = np.random.randint(0, len(pop_archive))
                
            x_i = population[i]
            x_pbest = population[pbest_idx]
            x_r1 = population[r1_idx]
            x_r2 = pop_archive[r2_idx]
            
            F = f_vals[i]
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # 2. Crossover (Binomial)
            CR = cr_vals[i]
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True # Ensure at least one dimension changes
            
            trial = np.where(mask, mutant, x_i)
            
            # 3. Bound Handling: Bounce-Back Strategy
            # If trial violates bound, set it to mean of bound and parent.
            # This is better than clipping as it maintains some search direction logic.
            lower_viol = trial < min_b
            trial[lower_viol] = (min_b[lower_viol] + x_i[lower_viol]) / 2.0
            
            upper_viol = trial > max_b
            trial[upper_viol] = (max_b[upper_viol] + x_i[upper_viol]) / 2.0
            
            # 4. Selection
            f_trial = safe_evaluate(trial)
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                
                # Record success info
                success_scr.append(CR)
                success_sf.append(F)
                fit_diffs.append(fitness[i] - f_trial)
                
                # Add replaced parent to archive
                archive.append(x_i.copy())
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_ind = trial.copy()
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
        
        population = new_population
        fitness = new_fitness
        
        # D. Post-Generation Updates
        # Check stagnation (using tolerance for float comparison)
        if (last_best_fit - best_fit) > 1e-12:
            stagnation_count = 0
            last_best_fit = best_fit
        else:
            stagnation_count += 1
            
        # Maintain Archive Size
        while len(archive) > pop_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        # Update Memory (Weighted Lehmer Mean)
        if len(success_scr) > 0:
            success_scr = np.array(success_scr)
            success_sf = np.array(success_sf)
            fit_diffs = np.array(fit_diffs)
            
            total_diff = np.sum(fit_diffs)
            if total_diff > 0:
                weights = fit_diffs / total_diff
            else:
                weights = np.ones(len(fit_diffs)) / len(fit_diffs)
                
            # M_CR Update
            m_cr_new = np.sum(weights * success_scr)
            if np.isnan(m_cr_new) or m_cr_new < 0: m_cr_new = 0.5
            mem_M_CR[k_mem] = m_cr_new
            
            # M_F Update
            denom = np.sum(weights * success_sf)
            if denom == 0:
                m_f_new = 0.5
            else:
                m_f_new = np.sum(weights * (success_sf ** 2)) / denom
                
            if np.isnan(m_f_new) or m_f_new <= 0: m_f_new = 0.5
            mem_M_F[k_mem] = m_f_new
            
            k_mem = (k_mem + 1) % H

    return best_fit
