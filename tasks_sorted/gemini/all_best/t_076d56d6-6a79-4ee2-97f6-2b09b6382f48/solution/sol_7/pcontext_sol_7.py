#The output value of the best generated algorithm is: 4.589970908775712
#
#The best generated algorithm code:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the function `func` using Restart-based JADE (Joint Adaptive Differential Evolution)
    with IPOP (Increasing Population Size) and an external Archive.
    
    Algorithm Features:
    1.  **JADE Adaptation**: Self-adapts mutation factor (F) and crossover rate (CR) using 
        success-based history (Lehmer mean for F, Arithmetic mean for CR).
    2.  **Archive Strategy**: Maintains a history of inferior solutions to improve diversity 
        in the mutation strategy `current-to-pbest/1`.
    3.  **IPOP (Increasing Population)**: Restarts the optimization with exponentially increasing 
        population sizes to switch from fast exploration to deep exploitation.
    4.  **Midpoint Bound Handling**: Out-of-bound solutions are corrected to the midpoint 
        between the parent and the boundary, preventing population accumulation at the edges.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---\\
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---\\
    best_val = float('inf')
    best_vec = None
    
    # --- IPOP Settings ---\\
    # Robust initial population size scaling with dimension
    base_pop_size = max(20, int(5 * dim))
    restart_count = 0
    
    # --- Main Restart Loop ---\\
    while True:
        # Check time before starting a new restart
        if datetime.now() - start_time >= time_limit:
            return best_val
            
        # IPOP: Scale population size
        pop_size = int(base_pop_size * (1.5 ** restart_count))
        
        # Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best solution
        start_idx = 0
        if best_vec is not None:
            population[0] = best_vec.copy()
            fitness[0] = best_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()
                
        # --- JADE Internal State ---\\
        mu_cr = 0.5
        mu_f = 0.5
        c_adapt = 0.1  # Adaptation rate
        
        # Archive Setup (max size = pop_size)
        archive = np.empty((pop_size, dim))
        arc_count = 0
        
        # --- Evolution Loop ---\\
        while True:
            # Check Time
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            # Check Convergence (Stagnation)
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break
            
            # 1. Parameter Generation
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Cauchy = loc + scale * tan(pi * (uniform - 0.5))
            u = np.random.rand(pop_size)
            f = mu_f + 0.1 * np.tan(np.pi * (u - 0.5))
            
            # Handle F constraints (F > 0: retry, F <= 1: clip)
            retry_mask = f <= 0
            while np.any(retry_mask):
                cnt = np.sum(retry_mask)
                f[retry_mask] = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(cnt) - 0.5))
                retry_mask = f <= 0
            f = np.clip(f, 0.0, 1.0)
            
            # 2. Mutation: current-to-pbest/1
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Select p-best (top 5%)
            sorted_idx = np.argsort(fitness)
            num_pbest = max(2, int(0.05 * pop_size))
            pbest_indices = sorted_idx[:num_pbest]
            pbest_sel = np.random.choice(pbest_indices, pop_size)
            x_pbest = population[pbest_sel]
            
            # Select r1 (from population, r1 != i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            collision_r1 = (r1_idx == np.arange(pop_size))
            r1_idx[collision_r1] = (r1_idx[collision_r1] + 1) % pop_size
            x_r1 = population[r1_idx]
            
            # Select r2 (from Population U Archive, r2 != r1, r2 != i)
            if arc_count > 0:
                # We form r2 from union virtually
                union_size = pop_size + arc_count
                r2_idx = np.random.randint(0, union_size, pop_size)
                
                # Fetch values
                x_r2 = np.empty((pop_size, dim))
                
                # Mask for values from Population
                mask_pop = r2_idx < pop_size
                x_r2[mask_pop] = population[r2_idx[mask_pop]]
                
                # Mask for values from Archive
                mask_arc = ~mask_pop
                x_r2[mask_arc] = archive[r2_idx[mask_arc] - pop_size]
            else:
                r2_idx = np.random.randint(0, pop_size, pop_size)
                collision_r2 = (r2_idx == r1_idx) | (r2_idx == np.arange(pop_size))
                r2_idx[collision_r2] = (r2_idx[collision_r2] + 1) % pop_size
                x_r2 = population[r2_idx]
                
            # Compute Mutant
            f_col = f[:, None]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_mat = np.random.rand(pop_size, dim)
            cross_mask = rand_mat < cr[:, None]
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # 4. Bound Handling (Midpoint)
            # If out of bounds, set to (parent + bound) / 2
            mask_l = trial_pop < min_b
            trial_pop[mask_l] = (population[mask_l] + min_b[mask_l]) / 2
            
            mask_u = trial_pop > max_b
            trial_pop[mask_u] = (population[mask_u] + max_b[mask_u]) / 2
            
            # 5. Selection and Adaptation
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_val
                    
                t_val = func(trial_pop[i])
                
                if t_val < fitness[i]:
                    # Successful Update
                    
                    # Update Archive
                    if arc_count < pop_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Random replacement
                        k = np.random.randint(0, pop_size)
                        archive[k] = population[i].copy()
                        
                    # Update Population
                    fitness[i] = t_val
                    population[i] = trial_pop[i]
                    
                    # Store params for adaptation
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    # Update Global Best
                    if t_val < best_val:
                        best_val = t_val
                        best_vec = trial_pop[i].copy()
                        
            # 6. Adapt Parameters (JADE Logic)
            if len(succ_f) > 0:
                sf = np.array(succ_f)
                scr = np.array(succ_cr)
                
                # Update mu_cr (Arithmetic Mean)
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(scr)
                
                # Update mu_f (Lehmer Mean)
                denom = np.sum(sf)
                if denom > 0:
                    lehmer_mean = np.sum(sf**2) / denom
                    mu_f = (1 - c_adapt) * mu_f + c_adapt * lehmer_mean
        
        # End of loop -> Restart
        restart_count += 1
        
    return best_val
