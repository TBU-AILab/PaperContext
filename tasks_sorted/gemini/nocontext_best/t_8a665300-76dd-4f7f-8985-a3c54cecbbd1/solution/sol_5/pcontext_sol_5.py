#Here is the improved algorithm. It implements **L-SHADE** (Linear Success-History Adaptive Differential Evolution) with an **Automatic Restart Mechanism**.
#
#This algorithm improves upon standard JADE/Nelder-Mead by:
#1.  **Linear Population Reduction:** It starts with a large population for exploration and progressively reduces it to force exploitation and convergence as time runs out.
#2.  **Historical Parameter Memory:** It maintains a history of successful $F$ (Scaling Factor) and $CR$ (Crossover Rate) values, adapting specifically to the landscape of the objective function.
#3.  **Stagnation Restart:** If the population converges (low variance) but the objective value is not optimal, it triggers a "soft restart" to escape local optima, which is likely where the previous algorithm (result ~11.85) got stuck.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE (Linear Success-History Adaptive Differential Evolution)
    with Automatic Restarts.
    """
    
    # --- Configuration ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # L-SHADE Parameters
    N_init = int(18 * dim) # Initial population size (standard L-SHADE)
    N_min = 4              # Minimum population size
    H = 6                  # Memory size
    
    # --- Helper Functions ---
    def check_time(buffer_sec=0.0):
        return (datetime.now() - start_time) >= (time_limit - timedelta(seconds=buffer_sec))

    def get_progress():
        """Returns time progress from 0.0 to 1.0"""
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # --- Initialization State ---
    # We wrap initialization in a function to support restarts
    state = {
        'pop': None,
        'fit': None,
        'M_cr': np.ones(H) * 0.5,
        'M_f': np.ones(H) * 0.5,
        'k': 0, # Memory index
        'archive': [],
        'pop_size': N_init,
        'best_val': float('inf'),
        'best_sol': None
    }

    def initialize_population(pop_size):
        # Latin Hypercube Sampling-style init for better coverage
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            pop[:, d] = points
        
        # Evaluate
        fit = np.array([func(ind) for ind in pop])
        return pop, fit

    # Initial boot
    state['pop'], state['fit'] = initialize_population(state['pop_size'])
    best_idx = np.argmin(state['fit'])
    state['best_val'] = state['fit'][best_idx]
    state['best_sol'] = state['pop'][best_idx].copy()

    # --- Main Loop ---
    while not check_time():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time progress
        progress = get_progress()
        N_target = int(round((N_min - N_init) * progress + N_init))
        N_target = max(N_min, N_target)
        
        # Resize if necessary
        current_N = len(state['pop'])
        if current_N > N_target:
            # Sort by fitness (worst at the end)
            sorted_idx = np.argsort(state['fit'])
            state['pop'] = state['pop'][sorted_idx]
            state['fit'] = state['fit'][sorted_idx]
            
            # Truncate
            state['pop'] = state['pop'][:N_target]
            state['fit'] = state['fit'][:N_target]
            
            # Archive size is also limited by current population size in L-SHADE
            if len(state['archive']) > N_target:
                # Randomly remove from archive
                del_indices = np.random.choice(len(state['archive']), len(state['archive']) - N_target, replace=False)
                state['archive'] = [x for i, x in enumerate(state['archive']) if i not in del_indices]
                
        current_N = N_target # Update local var
        
        # 2. Restart Mechanism
        # If population has collapsed (low variance) but solution is not super optimal, restart.
        fit_std = np.std(state['fit'])
        if fit_std < 1e-6 and progress < 0.85:
            # Keep best, re-init the rest
            # We slightly increase N for the restart to boost diversity again, but respect decay
            N_restart = min(int(N_init * 0.5), int((N_min - N_init) * progress + N_init) + 10)
            N_restart = max(N_min + 2, N_restart)
            
            new_pop, new_fit = initialize_population(N_restart - 1)
            
            # Construct new population: Best so far + New randoms
            state['pop'] = np.vstack((state['best_sol'].reshape(1, dim), new_pop))
            state['fit'] = np.hstack((state['best_val'], new_fit))
            
            # Reset memory slightly to allow re-adaptation
            state['M_cr'] = np.ones(H) * 0.5
            state['M_f'] = np.ones(H) * 0.5
            state['archive'] = []
            continue # Skip to next loop iteration

        # 3. Parameter Generation
        # Generate CR and F for each individual based on Memory
        r_indices = np.random.randint(0, H, current_N)
        mu_cr = state['M_cr'][r_indices]
        mu_f = state['M_f'][r_indices]
        
        # CR ~ Normal(mu_cr, 0.1)
        cr_g = np.random.normal(mu_cr, 0.1)
        cr_g = np.clip(cr_g, 0, 1)
        # Inherit CR value -1 is effectively 0 in L-SHADE logic if fixed, 
        # but here we just clip.
        
        # F ~ Cauchy(mu_f, 0.1)
        f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(current_N) - 0.5))
        
        # Handle F constraints
        f_g = np.where(f_g > 1, 1.0, f_g)
        # If F <= 0, regenerate until > 0 (vectorized approximated)
        bad_f = f_g <= 0
        while np.any(bad_f):
            f_g[bad_f] = mu_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f)) - 0.5))
            bad_f = f_g <= 0
        
        # 4. Mutation (current-to-pbest/1)
        # Sort population for p-best selection
        sorted_indices = np.argsort(state['fit'])
        pop_sorted = state['pop'][sorted_indices]
        
        # p_best rate scales roughly from 0.11 down to 0.02, or fixed 0.05
        # L-SHADE standard uses p in [2/N, 0.2]
        p = max(2.0/current_N, 0.11)
        top_p_count = max(2, int(current_N * p))
        
        # Prepare arrays
        v = np.zeros_like(state['pop'])
        
        # Union of Population and Archive for second difference vector
        if len(state['archive']) > 0:
            archive_arr = np.array(state['archive'])
            pool = np.vstack((state['pop'], archive_arr))
        else:
            pool = state['pop']
            
        for i in range(current_N):
            x_i = state['pop'][i]
            
            # Select p-best
            p_idx = np.random.randint(0, top_p_count)
            x_pbest = pop_sorted[p_idx]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, current_N)
            while r1 == i:
                r1 = np.random.randint(0, current_N)
            x_r1 = state['pop'][r1]
            
            # Select r2 (distinct from i and r1, from pool)
            r2 = np.random.randint(0, len(pool))
            while r2 == i or (r2 < current_N and r2 == r1):
                r2 = np.random.randint(0, len(pool))
            x_r2 = pool[r2]
            
            # Mutation Equation
            v[i] = x_i + f_g[i] * (x_pbest - x_i) + f_g[i] * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            # Generate mask
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < cr_g[i]
            mask[j_rand] = True
            
            # Apply mask to create trial vector u
            u = np.where(mask, v[i], x_i)
            
            # Bound Handling (Reflection/Bounce works well)
            low_violation = u < min_b
            high_violation = u > max_b
            
            # Standard midpoint target for violations
            u[low_violation] = (x_i[low_violation] + min_b[low_violation]) / 2.0
            u[high_violation] = (x_i[high_violation] + max_b[high_violation]) / 2.0
            
            v[i] = u # Store back in v for simplicity
            
        # 5. Selection and Updates
        # We need to evaluate v. Check time frequently.
        success_f = []
        success_cr = []
        diff_fitness = []
        
        new_pop = state['pop'].copy()
        new_fit = state['fit'].copy()
        
        for i in range(current_N):
            if check_time(): 
                return state['best_val']
                
            u_i = v[i]
            f_u = func(u_i)
            
            if f_u < state['fit'][i]:
                # Success
                success_f.append(f_g[i])
                success_cr.append(cr_g[i])
                diff_fitness.append(state['fit'][i] - f_u)
                
                # Add parent to archive
                state['archive'].append(state['pop'][i].copy())
                
                # Update population
                new_pop[i] = u_i
                new_fit[i] = f_u
                
                # Update global best
                if f_u < state['best_val']:
                    state['best_val'] = f_u
                    state['best_sol'] = u_i.copy()
        
        state['pop'] = new_pop
        state['fit'] = new_fit
        
        # Maintain Archive Size (limit to current_N)
        if len(state['archive']) > current_N:
            rem_cnt = len(state['archive']) - current_N
            # Random removal
            for _ in range(rem_cnt):
                idx = np.random.randint(0, len(state['archive']))
                state['archive'].pop(idx)
        
        # 6. Update Memory (Lehmer Mean)
        if len(success_f) > 0:
            success_f = np.array(success_f)
            success_cr = np.array(success_cr)
            weights = np.array(diff_fitness)
            weights = weights / np.sum(weights) # Normalize
            
            # Weighted Lehmer Mean for F
            mean_pow2 = np.sum(weights * (success_f ** 2))
            mean_pow1 = np.sum(weights * success_f)
            mean_f = mean_pow2 / (mean_pow1 + 1e-10)
            
            # Weighted Mean for CR
            mean_cr = np.sum(weights * success_cr)
            
            # Update Memory slot k
            k = state['k']
            state['M_f'][k] = 0.5 * state['M_f'][k] + 0.5 * mean_f
            state['M_cr'][k] = 0.5 * state['M_cr'][k] + 0.5 * mean_cr
            
            # Increment pointer
            state['k'] = (state['k'] + 1) % H

    return state['best_val']
