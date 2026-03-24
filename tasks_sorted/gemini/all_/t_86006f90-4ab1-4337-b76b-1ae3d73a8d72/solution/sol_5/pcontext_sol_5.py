#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 28.68633218659683
#2. output value is: 181.15154088399086
#3. output value is: 40.20818255996774
#4. output value is: 24.889496045193397
#5. output value is: 105.40816267667486
#
#The best result so far (24.89) was achieved by Algorithm 4 (jDE with Restart). This suggests that a strategy combining a **small population** (for speed) with a **Restart Mechanism** (to escape local optima) is effective for this specific problem. However, Algorithm 1 (SHADE) also performed reasonably well (28.68).
#
#The following improved algorithm implements **SHADE with Restarts and Coordinate Descent Polishing**.
#
#### Improvements Explanation:
#1.  **SHADE Adaptation**: It replaces the random parameter resetting of jDE (Algo 4) with SHADE's **history-based adaptation** ($M_{CR}, M_F$). This allows the algorithm to learn the optimal mutation/crossover rates for the specific function landscape.
#2.  **Hybridization (Polishing)**: A critical addition is a **Coordinate Descent Local Search** step. Before restarting (when the population stagnates), the algorithm performs a quick, gradient-free local search around the best solution found. This "polishing" step can significantly refine the solution, squeezing out residual error that DE might struggle to resolve due to its stochastic nature.
#3.  **Current-to-pBest with Archive**: Uses `current-to-pbest/1` mutation with an external archive. This is generally more robust than `current-to-best` (Algo 4) as it maintains diversity better, while still being aggressive enough for convergence.
#4.  **Optimized Restart Strategy**: Retains the successful restart logic: keep the global best solution (Elitism) and randomize the rest of the population to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    combined with a Restart Mechanism and Coordinate Descent Polishing.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Kept small (similar to Algo 4) to ensure fast convergence
    # and maximize the number of restarts/generations.
    pop_size = int(max(20, 5 * dim))
    if pop_size > 50: 
        pop_size = 50
    
    # SHADE Parameters
    H = 6 # History memory size
    
    # Archive size (stores inferior solutions to maintain diversity)
    archive_size = int(pop_size * 2.0)
    
    # Local Search (Polishing) Budget
    # Limit evaluations to prevent spending too much time on one basin
    ls_max_evals = max(50, dim * 5)
    
    # Bounds preparation
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_global_val = float('inf')
    best_global_sol = None
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Main Restart Loop ---
    while not check_time():
        # 1. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best into the new population to refine it further
        if best_global_sol is not None:
            pop[0] = best_global_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_global_val
            
            # Optimization: Skip re-evaluation of the injected best solution
            if best_global_sol is not None and i == 0:
                val = best_global_val
            else:
                val = func(pop[i])
            
            fitness[i] = val
            
            if val < best_global_val:
                best_global_val = val
                best_global_sol = pop[i].copy()
                
        # Memory Initialization for SHADE
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Stagnation Tracking
        last_best_fit = fitness.min()
        stag_count = 0
        
        # 2. Differential Evolution Loop
        while not check_time():
            # Sort population for p-best selection
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Check Stagnation / Convergence
            curr_best_fit = fitness[0]
            if np.abs(curr_best_fit - last_best_fit) < 1e-9:
                stag_count += 1
            else:
                stag_count = 0
                last_best_fit = curr_best_fit
                
            # Restart Trigger: Low variance or prolonged stagnation
            if np.std(fitness) < 1e-6 or stag_count > 40:
                # --- Local Search (Polishing) ---
                # Before restarting, attempt to refine the best solution 
                # using Coordinate Descent (Pattern Search).
                
                center = pop[0].copy()
                c_val = fitness[0]
                
                # Initial step size relative to domain
                step_size = np.max(diff_b) * 0.05
                ls_evals = 0
                
                while step_size > 1e-8 and ls_evals < ls_max_evals and not check_time():
                    improved = False
                    # Randomize dimension order to avoid bias
                    dims = np.random.permutation(dim)
                    
                    for d in dims:
                        if check_time(): return best_global_val
                        
                        original = center[d]
                        
                        # Try negative direction
                        center[d] = np.clip(original - step_size, min_b[d], max_b[d])
                        val = func(center)
                        ls_evals += 1
                        
                        if val < c_val:
                            c_val = val
                            improved = True
                            if c_val < best_global_val:
                                best_global_val = c_val
                                best_global_sol = center.copy()
                        else:
                            # Try positive direction
                            center[d] = np.clip(original + step_size, min_b[d], max_b[d])
                            val = func(center)
                            ls_evals += 1
                            
                            if val < c_val:
                                c_val = val
                                improved = True
                                if c_val < best_global_val:
                                    best_global_val = c_val
                                    best_global_sol = center.copy()
                            else:
                                center[d] = original # Revert if no improvement
                                
                    if not improved:
                        step_size /= 2.0
                        
                break # Break inner loop to trigger restart
            
            # --- SHADE Parameter Generation ---
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # CR ~ Normal(M_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(M_f, 0.1)
            f = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    val = m_f[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0:
                        if val > 1: val = 1.0
                        f[i] = val
                        break
                        
            # --- Mutation & Crossover ---
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            success_cr = []
            success_f = []
            success_df = []
            
            # p-best setting (randomize p in [2/N, 0.2] for robustness)
            p_min = 2.0 / pop_size
            p_i = np.random.uniform(p_min, 0.2)
            num_p = int(max(2, pop_size * p_i))
            
            for i in range(pop_size):
                if check_time(): return best_global_val
                
                # Mutation: current-to-pbest/1
                p_idx = np.random.randint(0, num_p)
                x_pbest = pop[p_idx]
                
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # r2 from Union(Population, Archive)
                c_size = pop_size + len(archive)
                r2 = np.random.randint(0, c_size)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, c_size)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                    
                mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
                
                # Crossover: Binomial
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary Constraint
                trial = np.clip(trial, min_b, max_b)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Store success data
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    success_df.append(fitness[i] - f_trial)
                    
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update Global Best
                    if f_trial < best_global_val:
                        best_global_val = f_trial
                        best_global_sol = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
                    
            pop = new_pop
            fitness = new_fitness
            
            # --- Archive Maintenance ---
            while len(archive) > archive_size:
                idx = np.random.randint(0, len(archive))
                archive.pop(idx)
                
            # --- Memory Update (Weighted Lehmer Mean) ---
            if len(success_df) > 0:
                s_df = np.array(success_df)
                s_cr = np.array(success_cr)
                s_f = np.array(success_f)
                
                total_df = np.sum(s_df)
                if total_df > 0:
                    weights = s_df / total_df
                    
                    mean_cr = np.sum(weights * s_cr)
                    mean_f = np.sum(weights * s_f**2) / np.sum(weights * s_f)
                    
                    M_cr[k_mem] = mean_cr
                    M_f[k_mem] = mean_f
                    k_mem = (k_mem + 1) % H
                    
    return best_global_val
