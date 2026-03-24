#The output value of the last generated algorithm was ~212, which indicates a regression compared to the best result of ~17.19. The best performing algorithm (Algorithm 1) utilized a robust **Restart Mechanism** and **Linear Population Size Reduction (LPSR)**. Algorithm 3 (L-SHADE) failed likely due to complexity or ineffective restart triggers for the specific landscape.
#
#To improve the results, I propose a **Self-Adaptive Ensemble Differential Evolution (SaDE) with LPSR and Restart**. This algorithm combines the strengths of the previous approaches:
#1.  **Ensemble Strategies**: Instead of relying on a single mutation strategy, it concurrently uses **`rand/1/bin`** (good for exploration/multimodal functions) and **`current-to-best/1/bin`** (good for exploitation/convergence). The algorithm automatically adapts the probability of using each strategy based on their recent success rates.
#2.  **Restart Mechanism**: Aggressively restarts the population when convergence is detected (low variance), injecting the global best solution to refine it or escape local optima.
#3.  **Learned Parameters**: It maintains the strategy success rates across restarts to "learn" the global properties of the function landscape.
#4.  **Robust Boundary Handling**: Uses reflection (bounce-back) to handle boundary violations, which preserves search momentum better than clipping.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Self-Adaptive Ensemble Differential Evolution (SaDE)
    with Linear Population Size Reduction (LPSR) and Restart Mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population: High initial size for exploration, reduces linearly
    # 20*dim is a standard baseline, capped for efficiency
    pop_size_init = min(500, max(50, 20 * dim))
    pop_size_min = 5
    
    # SaDE Strategy Adaptation
    # Strat 0: DE/rand/1/bin (Exploration)
    # Strat 1: DE/current-to-best/1/bin (Exploitation)
    prob_strat = 0.5 # Initial probability of choosing Strat 0
    succ_counts = np.array([0.0, 0.0])  # Successes per strategy
    total_counts = np.array([0.0, 0.0]) # Attempts per strategy
    
    # jDE Control Parameter Adaptation
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Helper: Bounce-Back Boundary Handling ---
    def handle_bounds(u):
        # Reflect lower bound violations
        viol_l = u < min_b
        if np.any(viol_l):
            u[viol_l] = 2 * min_b[viol_l] - u[viol_l]
            # Clip if reflection is still out of bounds (rare double bounce)
            u[u < min_b] = min_b[np.where(u < min_b)[1]]
            
        # Reflect upper bound violations
        viol_u = u > max_b
        if np.any(viol_u):
            u[viol_u] = 2 * max_b[viol_u] - u[viol_u]
            u[u > max_b] = max_b[np.where(u > max_b)[1]]
        return u

    # --- Main Optimization Loop (Restart Mechanism) ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Initialization for new Restart
        pop_size = pop_size_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous runs to refine or guide
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol
            fitness[0] = best_val
            start_eval_idx = 1
            
        # Initialize jDE parameters (F and CR)
        # F ~ U(0.1, 0.9), CR ~ U(0.0, 1.0)
        F = 0.1 + 0.8 * np.random.rand(pop_size)
        CR = np.random.rand(pop_size)
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
                
        # 2. Evolutionary Cycle
        while True:
            # Check Global Time
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_time:
                return best_val
            
            # --- Linear Population Size Reduction (LPSR) ---
            progress = elapsed / max_time
            target_pop = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                # Sort by fitness and truncate the worst
                idxs = np.argsort(fitness)
                keep_idxs = idxs[:target_pop]
                population = population[keep_idxs]
                fitness = fitness[keep_idxs]
                F = F[keep_idxs]
                CR = CR[keep_idxs]
                pop_size = target_pop
                
            # --- Convergence Check (Trigger Restart) ---
            # If population has converged (low variance or range), restart
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                break
                
            # --- Strategy Selection (SaDE) ---
            # Generate mask based on prob_strat
            # Strat 0: rand/1 (exploration), Strat 1: current-to-best/1 (exploitation)
            mask_s0 = np.random.rand(pop_size) < prob_strat
            mask_s1 = ~mask_s0
            
            # Record strategy usage
            total_counts[0] += np.sum(mask_s0)
            total_counts[1] += np.sum(mask_s1)
            
            # --- Parameter Adaptation (jDE) ---
            # 10% chance to reset F or CR
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            if np.any(mask_F):
                F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            if np.any(mask_CR):
                CR[mask_CR] = np.random.rand(np.sum(mask_CR))
                
            # --- Mutation ---
            v = np.zeros_like(population)
            idxs = np.arange(pop_size)
            
            # Generate random indices r1, r2, r3
            r1 = np.random.randint(0, pop_size, pop_size)
            hit = r1 == idxs
            r1[hit] = (r1[hit] + 1) % pop_size
            
            r2 = np.random.randint(0, pop_size, pop_size)
            hit = (r2 == idxs) | (r2 == r1)
            r2[hit] = (r2[hit] + 2) % pop_size
            
            r3 = np.random.randint(0, pop_size, pop_size)
            hit = (r3 == idxs) | (r3 == r1) | (r3 == r2)
            r3[hit] = (r3[hit] + 3) % pop_size
            
            F_col = F[:, None]
            
            # Strategy 0: DE/rand/1/bin
            if np.any(mask_s0):
                v[mask_s0] = population[r1[mask_s0]] + \
                             F_col[mask_s0] * (population[r2[mask_s0]] - population[r3[mask_s0]])
            
            # Strategy 1: DE/current-to-best/1/bin
            if np.any(mask_s1):
                best_idx = np.argmin(fitness)
                best_vec = population[best_idx]
                v[mask_s1] = population[mask_s1] + \
                             F_col[mask_s1] * (best_vec - population[mask_s1]) + \
                             F_col[mask_s1] * (population[r1[mask_s1]] - population[r2[mask_s1]])
                             
            # --- Crossover ---
            mask_cross = np.random.rand(pop_size, dim) < CR[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[idxs, j_rand] = True # Ensure at least one dim from mutant
            
            u = np.where(mask_cross, v, population)
            
            # --- Boundary Handling ---
            u = handle_bounds(u)
            
            # --- Selection ---
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                
                f_trial = func(u[i])
                
                if f_trial <= fitness[i]:
                    # Trial is better or equal
                    population[i] = u[i]
                    fitness[i] = f_trial
                    
                    # Track success for strategy adaptation
                    if mask_s0[i]:
                        succ_counts[0] += 1
                    else:
                        succ_counts[1] += 1
                        
                    # Update global best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = u[i].copy()
            
            # --- Update Strategy Probability ---
            # Calculate success rates
            denom0 = total_counts[0] + 1e-10
            denom1 = total_counts[1] + 1e-10
            rate0 = succ_counts[0] / denom0
            rate1 = succ_counts[1] / denom1
            
            total_rate = rate0 + rate1
            if total_rate > 0:
                prob_strat = rate0 / total_rate
                # Clamp probability to [0.05, 0.95] to prevent strategy extinction
                prob_strat = max(0.05, min(0.95, prob_strat))
                
            # Decay counts to adapt to changing landscape phases
            succ_counts *= 0.95
            total_counts *= 0.95
            
    return best_val
