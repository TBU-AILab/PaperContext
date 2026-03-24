#The best generated algorithm is an implementation of **JADE (Adaptive Differential Evolution) with an External Archive and a Restart Mechanism**.
#
#### Algorithm Description
#This approach combines three powerful strategies to minimize the objective function efficiently within a limited time:
#1.  **JADE (Adaptive Parameters)**: Instead of fixed crossover ($CR$) and mutation ($F$) rates, the algorithm self-adapts these parameters based on the success of previous generations. It maintains mean values $\mu_{CR}$ and $\mu_{F}$ and updates them using successful parameters, allowing the algorithm to learn whether the landscape requires exploration (high $F$) or exploitation (low $F$).
#2.  **External Archive**: The algorithm maintains an archive of superior solutions that were recently replaced. The mutation strategy `current-to-pbest/1/bin` draws secondary difference vectors from the union of the population and this archive ($P \cup A$). This preserves population diversity and prevents premature convergence, addressing the main weakness of standard adaptive DE.
#3.  **Restart Mechanism with Bound Reflection**: To handle multimodal functions (many local minima), the algorithm monitors population variance and fitness stagnation. If the search converges or stagnates, it triggers a restart with fresh individuals (while preserving the global best). Additionally, it uses a reflection-based bound handling (`(bound + parent) / 2`) rather than simple clipping, which helps maintain valid gradients near boundaries.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    # Use a safe buffer to ensure we return before the strict cutoff
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    # Population Size: Adapted to dimension.
    # Literature suggests N = 18 * dim. We clamp it to [30, 200] to ensure 
    # adequate diversity for low dims and speed for high dims.
    pop_size = int(18 * dim)
    pop_size = max(30, min(pop_size, 200))

    # Adaptation parameters (JADE)
    c_adapt = 0.1      # Learning rate for mu_cr, mu_f
    p_best_rate = 0.05 # Top 5% used for p-best mutation

    # Pre-process bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    bound_diff = ub - lb
    
    # Pre-allocate bound matrices for vectorized broadcasting
    lb_mat = np.tile(lb, (pop_size, 1))
    ub_mat = np.tile(ub, (pop_size, 1))

    # Global best solution found across all restarts
    best_fitness = float('inf')

    # -------------------------------------------------------------------------
    # Main Loop (Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check time before starting a new restart
        if datetime.now() >= end_time:
            return best_fitness

        # Reset Adaptive Means for new environment
        mu_cr = 0.5
        mu_f = 0.5

        # Initialize Population
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize Archive
        # Stores good solutions replaced by better ones to maintain diversity
        archive = np.empty((pop_size, dim)) 
        arc_count = 0

        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val

        # Stagnation tracking variables
        last_best_fit = np.min(fitness)
        stagnation_count = 0

        # ---------------------------------------------------------------------
        # Evolution Loop
        # ---------------------------------------------------------------------
        while True:
            # Time Check
            if datetime.now() >= end_time: return best_fitness

            # 1. Parameter Generation
            # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)

            # F ~ Cauchy(mu_f, 0.1)
            # Use standard_cauchy and scale
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Constrain F
            f[f > 1.0] = 1.0
            # If F is too small/negative, set to conservative default to avoid stagnation
            f[f <= 0.0] = 0.4

            # 2. Mutation: current-to-pbest/1/bin with Archive
            # Strategy: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            
            # Select p-best (top p%)
            sorted_idx = np.argsort(fitness)
            num_pbest = max(1, int(p_best_rate * pop_size))
            pbest_indices = sorted_idx[:num_pbest]
            
            # Randomly assign a pbest to each individual
            pbest_selected = pop[np.random.choice(pbest_indices, pop_size)]

            # Select r1 (distinct from current i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Fix collision r1 == i
            mask_col = (r1_idx == np.arange(pop_size))
            r1_idx[mask_col] = (r1_idx[mask_col] + 1) % pop_size
            x_r1 = pop[r1_idx]

            # Select r2 (distinct from i, r1) from Union(Population, Archive)
            union_size = pop_size + arc_count
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Rough collision fix (r2 != i)
            mask_col2 = (r2_idx == np.arange(pop_size))
            r2_indices = r2_idx
            r2_indices[mask_col2] = (r2_indices[mask_col2] + 1) % union_size
            
            # Retrieve x_r2 vectors
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2_indices < pop_size
            x_r2[mask_pop] = pop[r2_indices[mask_pop]]
            
            # Indices >= pop_size refer to the archive
            mask_arc = ~mask_pop
            if np.any(mask_arc):
                arc_indices = r2_indices[mask_arc] - pop_size
                x_r2[mask_arc] = archive[arc_indices]

            # Compute Mutant Vectors
            F_col = f[:, None]
            mutant = pop + F_col * (pbest_selected - pop) + F_col * (x_r1 - x_r2)

            # 3. Crossover
            rand_j = np.random.rand(pop_size, dim)
            cross_mask = rand_j < cr[:, None]
            # Ensure at least one parameter is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)

            # 4. Bound Handling (Reflection/Mean)
            # Instead of clipping, set value to mean of bound and parent.
            # This preserves distribution better than hard clipping.
            mask_l = trial < lb_mat
            mask_u = trial > ub_mat
            trial[mask_l] = (lb_mat[mask_l] + pop[mask_l]) / 2.0
            trial[mask_u] = (ub_mat[mask_u] + pop[mask_u]) / 2.0

            # 5. Selection and Adaptation
            successful_cr = []
            successful_f = []
            parents_to_archive = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                new_val = func(trial[i])
                
                # Greedy Selection
                if new_val < fitness[i]:
                    # Parent enters archive
                    parents_to_archive.append(pop[i].copy())
                    
                    fitness[i] = new_val
                    pop[i] = trial[i]
                    
                    # Store successful parameters
                    successful_cr.append(cr[i])
                    successful_f.append(f[i])
                    
                    if new_val < best_fitness:
                        best_fitness = new_val

            # Update Archive
            for p in parents_to_archive:
                if arc_count < pop_size:
                    archive[arc_count] = p
                    arc_count += 1
                else:
                    # Random replacement if full
                    ridx = np.random.randint(0, pop_size)
                    archive[ridx] = p

            # Update Adaptive Parameters (Lehmer Mean for F, Arithmetic for CR)
            if len(successful_cr) > 0:
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(successful_cr)
                
                sf = np.array(successful_f)
                denom = np.sum(sf)
                if denom > 1e-10:
                    lehmer_mean = np.sum(sf**2) / denom
                    mu_f = (1 - c_adapt) * mu_f + c_adapt * lehmer_mean

            # 6. Restart Conditions
            current_best = np.min(fitness)
            pop_std = np.std(fitness)
            
            # Check for Convergence (low variance)
            if pop_std < 1e-8:
                break
            
            # Check for Stagnation (no improvement)
            if abs(current_best - last_best_fit) < 1e-9:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best
            
            if stagnation_count > 35:
                break

    return best_fitness
