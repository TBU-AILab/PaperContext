#The following Python algorithm implements **SHADE (Success-History based Adaptive Differential Evolution)** enhanced with **LHS (Latin Hypercube Sampling) Initialization** and a **Stagnation-Based Restart Mechanism**.
#
##### Key Improvements
#1.  **LHS Initialization**: Instead of purely random initialization, the algorithm uses Latin Hypercube Sampling to ensure the initial population is evenly stratified across the search space. This maximizes coverage and improves the probability of landing in a promising basin of attraction early on.
#2.  **SHADE Algorithm**: Utilizing the state-of-the-art SHADE strategy, the algorithm adapts mutation ($F$) and crossover ($CR$) parameters based on historical success data, effectively "learning" the landscape's properties.
#3.  **Stagnation Detection & Restart**: The algorithm monitors both population diversity (standard deviation of fitness) and the progress of the global best solution. If the population converges to a local optimum or fails to improve for a set number of generations (`patience`), a restart is triggered.
#4.  **Soft Restart Strategy**: During a restart, the best solution (elite) is preserved, but the rest of the population is re-initialized using LHS. The historical memory is reset to remove bias from the previous local optimum, allowing unbiased exploration of the new search phase.
#5.  **Robust Bound Handling**: Uses clipping to strictly enforce bounds, which is generally robust for a wide range of benchmark functions when combined with the `current-to-pbest` mutation strategy.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using SHADE with LHS Initialization and Stagnation-Based Restart.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Set a strict deadline with a small buffer (0.1s) for safe return
    deadline = start_time + timedelta(seconds=max_time - 0.1)

    # --- Configuration ---
    # Population size: Adaptive to dimension
    # 20*dim provides good diversity, clamped to [50, 200] to balance speed/exploration
    pop_size = int(np.clip(20 * dim, 50, 200))
    
    # SHADE Parameters
    H = 5                   # History Memory Size
    mem_cr = np.full(H, 0.5)# Memory for Crossover Rate
    mem_f = np.full(H, 0.5) # Memory for Scaling Factor
    k_mem = 0               # Memory index pointer
    p_best_rate = 0.11      # Top 11% for p-best selection
    arc_rate = 2.0          # Archive size relative to population
    
    # Stagnation Parameters
    patience = 50           # Max generations without improvement before restart
    no_improv_count = 0     # Counter for non-improving generations
    
    # Pre-process Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Helper: Latin Hypercube Sampling (LHS) ---
    def generate_population(n_samples):
        """Generates a stratified population using LHS."""
        pop_lhs = np.zeros((n_samples, dim))
        for d in range(dim):
            # Divide dimension range into n_samples intervals
            edges = np.linspace(min_b[d], max_b[d], n_samples + 1)
            lower = edges[:-1]
            upper = edges[1:]
            
            # Sample uniformly within each interval
            samples = np.random.uniform(lower, upper)
            # Shuffle to mix dimensions
            np.random.shuffle(samples)
            pop_lhs[:, d] = samples
        return pop_lhs

    # --- Initialization ---
    pop = generate_population(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # External Archive
    archive_size = int(pop_size * arc_rate)
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Global Best Tracking
    best_fit = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= deadline:
            return best_fit if best_sol is not None else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while datetime.now() < deadline:
        
        # 1. Restart Check
        # Check convergence (low diversity) and stagnation (no improvement)
        std_fit = np.std(fitness)
        is_converged = std_fit < 1e-6
        is_stagnant = no_improv_count >= patience
        
        if is_converged or is_stagnant:
            # --- Perform Restart ---
            no_improv_count = 0
            
            # Preserve the Elite (Global Best)
            elite_sol = best_sol.copy()
            elite_val = best_fit
            
            # Re-initialize Population using LHS
            pop = generate_population(pop_size)
            
            # Inject Elite at index 0
            pop[0] = elite_sol
            fitness[:] = float('inf')
            fitness[0] = elite_val
            
            # Reset SHADE Memory & Archive (New basin, new learning needed)
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            k_mem = 0
            n_archive = 0
            
            # Evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if datetime.now() >= deadline: return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_sol = pop[i].copy()
            
            # Continue immediately to next generation
            continue
            
        # 2. SHADE Parameter Generation
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(m_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(m_f, 0.1)
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        # Clamp F: >1 becomes 1, <=0 becomes 0.1
        f = np.where(f > 1.0, 1.0, f)
        f = np.where(f <= 0.0, 0.1, f)
        
        # 3. Mutation Strategy: current-to-pbest/1
        # Sort population to find p-best
        sorted_idx = np.argsort(fitness)
        pop_sorted = pop[sorted_idx]
        
        # Select p-best individuals
        num_pbest = max(2, int(pop_size * p_best_rate))
        pbest_indices = np.random.randint(0, num_pbest, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Select r1 (distinct from current i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        conflict = (r1_indices == np.arange(pop_size))
        r1_indices[conflict] = (r1_indices[conflict] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Union of Pop and Archive)
        if n_archive > 0:
            union_pop = np.vstack((pop, archive[:n_archive]))
        else:
            union_pop = pop
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutation Vectors
        f_vec = f[:, np.newaxis]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = rand_vals < cr[:, np.newaxis]
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Selection
        success_f = []
        success_cr = []
        diffs = []
        
        prev_best = best_fit
        
        for i in range(pop_size):
            if datetime.now() >= deadline: return best_fit
            
            f_trial = func(trial[i])
            
            # Greedy Selection
            if f_trial < fitness[i]:
                diff = fitness[i] - f_trial
                
                # Update Archive with old solution
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Random replacement if full
                    rep = np.random.randint(0, archive_size)
                    archive[rep] = pop[i].copy()
                
                # Update Population
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                # Record success for adaptation
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(diff)
                
                # Update Global Best
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_sol = trial[i].copy()
        
        # Stagnation Counter Update
        if best_fit < prev_best:
            no_improv_count = 0
        else:
            no_improv_count += 1
            
        # 7. Update SHADE History Memory
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_diff = np.array(diffs)
            
            # Weights based on fitness improvement magnitude
            weights = s_diff / np.sum(s_diff)
            
            # Weighted Mean for CR
            mean_cr = np.sum(weights * s_cr)
            
            # Weighted Lehmer Mean for F
            num = np.sum(weights * (s_f ** 2))
            den = np.sum(weights * s_f)
            mean_f = num / den if den > 0 else 0.5
            
            # Update Memory
            mem_cr[k_mem] = mean_cr
            mem_f[k_mem] = mean_f
            k_mem = (k_mem + 1) % H
            
    return best_fit
