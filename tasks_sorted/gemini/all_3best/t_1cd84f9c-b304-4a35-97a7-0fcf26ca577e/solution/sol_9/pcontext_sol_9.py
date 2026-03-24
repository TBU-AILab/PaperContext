#The best performing algorithm so far was Algorithm 2 (`2.33`), which implemented **Restarter SHADE with Time-Dependent Adaptation**. The significant improvement (from ~8.2 to 2.3) suggests that the combination of SHADE parameter adaptation, time-decaying `p` parameter (inspired by jSO), and a restart mechanism is the correct direction.
#
#To improve upon Algorithm 2, I propose **Multi-Resolution Restarter SHADE (MR-SHADE)**.
#
#**Key Improvements:**
#1.  **Multi-Resolution Restart Strategy:** Algorithm 2 used a single "local" perturbation scale (1% of domain) during restart. MR-SHADE introduces **multi-scale local exploitation**. When restarting, it generates 25% of the population around the best individual using a gradient of Gaussian scales ($10^{-2}, 10^{-3}, \dots, 10^{-6}$). This significantly improves the algorithm's ability to "drill down" into the minimum (improving precision) regardless of the basin's width, while the remaining 75% of the population performs global exploration to escape local optima.
#2.  **Optimized Stagnation Detection:** The stagnation limit is tightened from 40 to 30 generations to ensure faster reaction to non-improving states, given the limited time budget.
#3.  **Memory Reset on Restart:** Maintains the crucial logic of resetting SHADE memory ($M_F, M_{CR}$) upon restart to ensure the new search phase is not biased by the history of the previous basin.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Multi-Resolution Restarter SHADE (MR-SHADE).
    
    Features:
    - SHADE with time-dependent 'p' parameter.
    - Multi-Resolution Restart: Exploits the best solution with multiple 
      scales of Gaussian perturbations to improve precision while exploring globally.
    - Robust parameter adaptation and stagnation detection.
    """
    # Setup timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # 1. Configuration
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population parameters
    # Balanced size for speed/diversity
    pop_size = int(max(30, min(100, 16 * dim))) 
    archive_size = int(2.0 * pop_size)
    
    # SHADE parameters
    H = 6 # Memory size
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    
    # 2. Initialization
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    archive = []
    
    best_ind = None
    best_fit = float('inf')
    
    # Helper functions
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')
            
    # Generate initial population
    for i in range(pop_size):
        if is_time_up(): return best_fit
        x = min_b + np.random.rand(dim) * diff_b
        population[i] = x
        val = safe_evaluate(x)
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_ind = x.copy()
            
    if is_time_up(): return best_fit
    
    # Stagnation monitoring
    last_best_fit = best_fit
    stagnation_count = 0
    stagnation_limit = 30 
    
    # 3. Main Loop
    while not is_time_up():
        
        # --- Time Progress ---
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(1.0, elapsed / max_time)
        
        # --- Restart Check ---
        pop_std = np.std(fitness)
        
        # Check improvement
        if best_fit < last_best_fit - 1e-12:
            stagnation_count = 0
            last_best_fit = best_fit
        else:
            stagnation_count += 1
            
        # Restart Condition: Converged OR Stagnated
        # Allow restart until 95% of time is consumed
        if (pop_std < 1e-9 or stagnation_count > stagnation_limit) and progress < 0.95:
            
            # Reset Memory & Archive
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            stagnation_count = 0
            
            # Keep Best
            population[0] = best_ind
            fitness[0] = best_fit
            
            # Multi-Resolution Restart (Exploitation around best)
            # Create shells at different scales to catch gradients or improve precision
            n_local = int(0.25 * pop_size)
            scales = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            
            idx = 1
            for scale in scales:
                count = max(1, n_local // len(scales))
                for _ in range(count):
                    if idx >= pop_size: break
                    if is_time_up(): return best_fit
                    
                    # Perturb around best with diminishing scale
                    x = best_ind + np.random.normal(0, 1, dim) * (diff_b * scale)
                    x = np.clip(x, min_b, max_b)
                    
                    population[idx] = x
                    val = safe_evaluate(x)
                    fitness[idx] = val
                    
                    if val < best_fit:
                        best_fit = val
                        best_ind = x.copy()
                    idx += 1
            
            # Global Random for the rest (Exploration)
            while idx < pop_size:
                if is_time_up(): return best_fit
                x = min_b + np.random.rand(dim) * diff_b
                population[idx] = x
                val = safe_evaluate(x)
                fitness[idx] = val
                
                if val < best_fit:
                    best_fit = val
                    best_ind = x.copy()
                idx += 1
                
            continue 
            
        # --- SHADE Adaptation ---
        # Adaptive p (linear decay)
        p_min = 2.0 / pop_size
        p_max = 0.2
        p_curr = p_max - (p_max - p_min) * progress
        p_curr = max(p_min, p_curr)
        
        # Memory-based generation
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f_vals = np.minimum(f_vals, 1.0)
        f_vals[f_vals <= 0] = 0.5 
        
        # Evolution
        sorted_indices = np.argsort(fitness)
        
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
            
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if is_time_up(): return best_fit
            
            x_i = population[i]
            
            # current-to-pbest/1
            top_p = int(max(2, p_curr * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, top_p)]
            x_pbest = population[pbest_idx]
            
            r1_idx = np.random.randint(0, pop_size)
            while r1_idx == i:
                 r1_idx = np.random.randint(0, pop_size)
            x_r1 = population[r1_idx]
            
            r2_idx = np.random.randint(0, len(pop_archive))
            while r2_idx == i or (r2_idx < pop_size and r2_idx == r1_idx):
                 r2_idx = np.random.randint(0, len(pop_archive))
            x_r2 = pop_archive[r2_idx]
            
            F = f_vals[i]
            CR = cr_vals[i]
            
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # Crossover
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            trial = np.where(mask, mutant, x_i)
            trial = np.clip(trial, min_b, max_b)
            
            f_trial = safe_evaluate(trial)
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                
                if f_trial < fitness[i]:
                    success_f.append(F)
                    success_cr.append(CR)
                    diff_fitness.append(fitness[i] - f_trial)
                    archive.append(x_i.copy())
                    
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_ind = trial.copy()
            else:
                new_population[i] = x_i
                new_fitness[i] = fitness[i]
                
        population = new_population
        fitness = new_fitness
        
        # Archive maintenance
        while len(archive) > archive_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        # Update Memory
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                
                m_cr_new = np.sum(w * np.array(success_cr))
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                sf = np.array(success_f)
                denom = np.sum(w * sf)
                if denom > 0:
                    m_f_new = np.sum(w * (sf**2)) / denom
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                else:
                    mem_M_F[k_mem] = 0.5
                
                k_mem = (k_mem + 1) % H
                
    return best_fit
