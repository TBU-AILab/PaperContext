#The best performing algorithm so far was Algorithm 1 (**Multi-Resolution Restarter SHADE**), achieving a value of `1.69`. This confirmed that a combination of SHADE-based parameter adaptation and a "multi-resolution" restart strategy (which places population clusters at exponentially decreasing scales around the best solution) is the most effective approach for this problem type within the limited time.
#
#To improve upon Algorithm 1, I propose **LHS-MR-SHADE (LHS-Initialized Multi-Resolution SHADE)**.
#
#**Key Improvements:**
#1.  **Latin Hypercube Sampling (LHS):** Algorithm 1 used standard random initialization (`np.random.rand`). For black-box optimization, this can lead to clustering and poor coverage of the search space. LHS ensures that the initial population (and the global component of restarts) is maximally stratified across all dimensions, significantly increasing the probability of landing in the global basin early.
#2.  **Extended Multi-Resolution Scales:** The previous algorithm used shells down to $10^{-6}$. This version extends the gradient down to $10^{-8}$ and adjusts the distribution, allowing the algorithm to "drill down" with even higher precision into the global minimum during the restart phase.
#3.  **Refined Restart Logic:** The restart mechanism now explicitly separates the population into a "Local Exploitation" group (30% of population, distributed across 8 scale shells) and a "Global Exploration" group (70% of population, generated via LHS). This balance prevents the "global" randoms from interfering with the high-precision refinement while ensuring escape from local optima.
#4.  **Tighter Stagnation Control:** The stagnation limit is reduced to 25 generations (from 30) to react faster to non-improving states, maximizing the utility of the limited time budget.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using LHS-Initialized Multi-Resolution SHADE (LHS-MR-SHADE).
    
    Key Features:
    - Initialization: Latin Hypercube Sampling (LHS) for stratified coverage.
    - Optimizer: SHADE (Success-History Adaptive DE) with Linear 'p' reduction.
    - Restart: Hybrid strategy with Multi-Resolution Gaussian Shells (Local) 
      and LHS (Global).
    - Resolution: Gradient of shells extends to 1e-8 for high-precision convergence.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- 1. Configuration & Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing
    # Set to maximize exploration capability while maintaining speed.
    # LHS allows us to use a slightly larger population effectively.
    pop_size = int(max(30, min(120, 18 * dim)))
    archive_size = int(2.5 * pop_size) # Larger archive to store diverse history
    
    # SHADE Memory Parameters
    H = 5 # History size
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    
    # --- 2. Helper Functions ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')
            
    def lhs_sample(n_samples):
        """
        Generates N stratified samples using Latin Hypercube Sampling.
        This ensures exactly one sample in each of the N intervals per dimension.
        """
        res = np.zeros((n_samples, dim))
        for d in range(dim):
            # Create N bins
            perms = np.random.permutation(n_samples)
            offsets = np.random.rand(n_samples)
            # Map bin index + offset to continuous domain
            step = diff_b[d] / n_samples
            res[:, d] = min_b[d] + (perms + offsets) * step
        return res

    # --- 3. Initialization ---
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    archive = []
    
    best_ind = None
    best_fit = float('inf')
    
    # Initialize using LHS
    init_pop = lhs_sample(pop_size)
    
    for i in range(pop_size):
        if is_time_up(): return best_fit
        x = init_pop[i]
        population[i] = x
        val = safe_evaluate(x)
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_ind = x.copy()
            
    if is_time_up(): return best_fit
    
    # State tracking variables
    last_best_fit = best_fit
    stagnation_count = 0
    stagnation_limit = 25 # Fast reaction to stagnation
    
    # --- 4. Main Optimization Loop ---
    while not is_time_up():
        
        # Calculate time progress [0.0, 1.0]
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(1.0, elapsed / max_time)
        
        # --- A. Stagnation Detection & Restart ---
        pop_std = np.std(fitness)
        
        # Check for fitness improvement (with tolerance)
        if best_fit < last_best_fit - 1e-12:
            stagnation_count = 0
            last_best_fit = best_fit
        else:
            stagnation_count += 1
            
        # Restart Condition:
        # 1. (Converged OR Stagnated) AND 2. (Not in the final 10% of time budget)
        if (pop_std < 1e-9 or stagnation_count > stagnation_limit) and progress < 0.9:
            
            # Reset SHADE Memory (crucial for new basins)
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            stagnation_count = 0
            
            # Keep the Global Best
            population[0] = best_ind
            fitness[0] = best_fit
            
            # 1. Multi-Resolution Local Exploitation (30% of pop)
            # Create shells at exponentially decreasing scales to catch gradients
            scales = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            n_local = int(0.3 * pop_size)
            idx = 1
            
            # Distribute local population across scales
            per_scale = max(1, n_local // len(scales))
            
            for scale in scales:
                for _ in range(per_scale):
                    if idx >= pop_size: break
                    if is_time_up(): return best_fit
                    
                    # Perturb around best with specific scale
                    noise = np.random.normal(0, 1, dim) * (diff_b * scale)
                    x = best_ind + noise
                    x = np.clip(x, min_b, max_b)
                    
                    population[idx] = x
                    val = safe_evaluate(x)
                    fitness[idx] = val
                    
                    if val < best_fit:
                        best_fit = val
                        best_ind = x.copy()
                    idx += 1
            
            # 2. Global Exploration with LHS (70% of pop)
            # Fill the rest with stratified random samples
            n_global = pop_size - idx
            if n_global > 0:
                global_pop = lhs_sample(n_global)
                for k in range(n_global):
                    if is_time_up(): return best_fit
                    if idx >= pop_size: break
                    
                    x = global_pop[k]
                    population[idx] = x
                    val = safe_evaluate(x)
                    fitness[idx] = val
                    
                    if val < best_fit:
                        best_fit = val
                        best_ind = x.copy()
                    idx += 1
            
            continue # Skip standard evolution immediately after restart

        # --- B. SHADE Parameter Adaptation ---
        # Linearly decay p from 0.2 (exploration) to 2/N (exploitation)
        p_min = 2.0 / pop_size
        p_max = 0.2
        p_curr = p_max - (p_max - p_min) * progress
        p_curr = max(p_min, p_curr)
        
        sorted_indices = np.argsort(fitness)
        
        # Sample F and CR from historical memory
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        # CR ~ Normal(M_CR, 0.1)
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f_vals = np.minimum(f_vals, 1.0)
        f_vals[f_vals <= 0] = 0.5 # Fallback
        
        # --- C. Evolution Cycle ---
        # Create pool for r2 selection: Population + Archive
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
            
        new_pop = np.zeros_like(population)
        new_fit = np.zeros_like(fitness)
        
        success_f = []
        success_cr = []
        diff_f = []
        
        for i in range(pop_size):
            if is_time_up(): return best_fit
            
            x_i = population[i]
            F = f_vals[i]
            CR = cr_vals[i]
            
            # Mutation: current-to-pbest/1
            top_p_idx = int(max(2, p_curr * pop_size))
            pbest_ind = sorted_indices[np.random.randint(0, top_p_idx)]
            x_pbest = population[pbest_ind]
            
            # r1 != i
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # r2 != i, != r1, from union
            r2 = np.random.randint(0, len(pop_archive))
            while r2 == i or (r2 < pop_size and r2 == r1):
                r2 = np.random.randint(0, len(pop_archive))
            x_r2 = pop_archive[r2]
            
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # Binomial Crossover
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            trial = np.where(mask, mutant, x_i)
            
            # Bound Handling: Clip
            trial = np.clip(trial, min_b, max_b)
            
            f_trial = safe_evaluate(trial)
            
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
                
                if f_trial < fitness[i]:
                    success_f.append(F)
                    success_cr.append(CR)
                    diff_f.append(fitness[i] - f_trial)
                    archive.append(x_i.copy())
                    
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_ind = trial.copy()
            else:
                new_pop[i] = x_i
                new_fit[i] = fitness[i]
                
        population = new_pop
        fitness = new_fit
        
        # --- D. Memory & Archive Maintenance ---
        while len(archive) > archive_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                
                # Update M_CR (Weighted Mean)
                m_cr_new = np.sum(w * np.array(success_cr))
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # Update M_F (Weighted Lehmer Mean)
                sf = np.array(success_f)
                denom = np.sum(w * sf)
                if denom > 0:
                    m_f_new = np.sum(w * (sf**2)) / denom
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                else:
                    mem_M_F[k_mem] = 0.5
                    
                k_mem = (k_mem + 1) % H
                
    return best_fit
