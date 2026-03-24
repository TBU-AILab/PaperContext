#Here is the fully operational and self-contained Python code implementing an Adaptive Differential Evolution algorithm (variant of SHADE) with a restart mechanism. It is designed to converge faster and escape local optima better than standard Differential Evolution.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Initialize start time to manage execution duration strictly
    start_time = time.time()
    
    # --- Algorithm Hyperparameters ---
    # Population size: Adapted to dimension, clamped to [20, 100] for balance
    # Larger populations explore better but converge slower per generation.
    pop_size = int(max(20, min(100, 15 * dim)))
    
    # SHADE-like Adaptive Parameter Memory
    H = 5                   # Size of historical memory
    mem_cr = np.full(H, 0.5) # Memory for Crossover Rate (initially 0.5)
    mem_f = np.full(H, 0.5)  # Memory for Scaling Factor (initially 0.5)
    k_mem = 0               # Pointer for memory update
    
    p_best_rate = 0.11      # Top 11% used for 'current-to-pbest' mutation
    
    # Restart / Stagnation Detection
    stagnation_limit = 20   # Generations without improvement to trigger restart
    stagnation_count = 0
    
    # --- Initialization ---
    # Prepare bounds arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population: random uniform within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Global best tracking
    best_val = float('inf')
    best_idx = -1
    
    # Evaluate initial population safely within time limits
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    last_best_val = best_val

    # --- Main Optimization Loop ---
    while True:
        # Time check at the start of each generation
        if time.time() - start_time >= max_time:
            return best_val
        
        # Sort population by fitness to select 'p-best'
        sorted_indices = np.argsort(fitness)
        num_p_best = max(2, int(pop_size * p_best_rate))
        top_p_indices = sorted_indices[:num_p_best]
        
        # --- Parameter Generation ---
        # Pick random memory slots for each individual
        r_idx = np.random.randint(0, H, pop_size)
        
        # Generate CR (Crossover Rate) using Normal distribution based on memory
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F (Scaling Factor) using Cauchy distribution
        # Cauchy allows occasional large jumps (exploration)
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        f = np.clip(f, 0.1, 1.0) # Clamp to valid range [0.1, 1.0]
        
        # Storage for successful updates to update memory later
        succ_cr = []
        succ_f = []
        fit_improv = []
        
        # --- Evolution Loop ---
        for i in range(pop_size):
            # Strict time check inside the loop
            if time.time() - start_time >= max_time:
                return best_val
            
            # 1. Mutation: 'current-to-pbest/1' strategy
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select p-best individual
            p_best = pop[np.random.choice(top_p_indices)]
            
            # Select r1, r2 distinct from i
            while True:
                r1 = np.random.randint(0, pop_size)
                if r1 != i: break
            while True:
                r2 = np.random.randint(0, pop_size)
                if r2 != i and r2 != r1: break
            
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Compute mutant vector
            mutant = pop[i] + f[i] * (p_best - pop[i]) + f[i] * (x_r1 - x_r2)
            
            # 2. Crossover: Binomial
            # Replace dimensions with mutant's based on CR
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < cr[i]
            mask[j_rand] = True # Ensure at least one dimension is changed
            trial = np.where(mask, mutant, pop[i])
            
            # 3. Boundary Handling: Clip to bounds
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Improvement found
                df = fitness[i] - f_trial
                
                # Store successful parameters if there was strict improvement
                if df > 0:
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    fit_improv.append(df)
                
                fitness[i] = f_trial
                pop[i] = trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
        
        # --- Memory Update (Weighted Lehmer Mean) ---
        if len(succ_cr) > 0:
            succ_cr = np.array(succ_cr)
            succ_f = np.array(succ_f)
            fit_improv = np.array(fit_improv)
            
            total_improv = np.sum(fit_improv)
            # Weights proportional to fitness improvement
            if total_improv > 0:
                weights = fit_improv / total_improv
                
                # Update CR memory
                m_cr_new = np.sum(weights * succ_cr)
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * m_cr_new
                
                # Update F memory (Lehmer mean)
                m_f_num = np.sum(weights * succ_f**2)
                m_f_den = np.sum(weights * succ_f)
                if m_f_den > 0:
                    m_f_new = m_f_num / m_f_den
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * m_f_new
                
                k_mem = (k_mem + 1) % H
        
        # --- Restart Mechanism ---
        # Check if we are stagnating or converged
        if best_val < last_best_val - 1e-8:
            stagnation_count = 0
            last_best_val = best_val
        else:
            stagnation_count += 1
            
        pop_std = np.std(fitness)
        
        # Trigger restart if stagnant for too long or population variance is effectively zero
        if stagnation_count >= stagnation_limit or pop_std < 1e-8:
            # Soft Restart: Keep best, reset others
            idxs = [z for z in range(pop_size) if z != best_idx]
            
            if idxs:
                # Split remainder: 50% pure random, 50% small perturbation around best
                cutoff = len(idxs) // 2
                reset_rnd = idxs[:cutoff]
                reset_pert = idxs[cutoff:]
                
                # Random reset (Exploration)
                if len(reset_rnd) > 0:
                    pop[reset_rnd] = min_b + np.random.rand(len(reset_rnd), dim) * diff_b
                    fitness[reset_rnd] = float('inf') # Force acceptance in next gen
                
                # Perturbation reset (Exploitation/Local Search)
                if len(reset_pert) > 0:
                    scale = 0.05 * diff_b # 5% of domain size
                    noise = np.random.randn(len(reset_pert), dim) * scale
                    pop[reset_pert] = pop[best_idx] + noise
                    pop[reset_pert] = np.clip(pop[reset_pert], min_b, max_b)
                    fitness[reset_pert] = float('inf')
                
                # Reset memories to re-adapt to new landscape features
                mem_cr.fill(0.5)
                mem_f.fill(0.5)
                stagnation_count = 0

    return best_val
