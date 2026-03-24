#The following algorithm implements **L-SHADE** (Linear Population Size Reduction, Success-History based Adaptive Differential Evolution).
#
#### Improvements over the previous (JADE + Nelder-Mead) approach:
#
#1.  **Linear Population Reduction (LPSR):** Instead of a fixed population size, this algorithm starts with a large population to explore the global space and linearly reduces the population size as time progresses. This naturally transitions the algorithm from **Global Exploration** to **Local Exploitation** without needing a separate Nelder-Mead phase or arbitrary switching logic.
#2.  **History-Based Adaptation (SHADE):** Instead of a single adaptive $\mu_{CR}$ and $\mu_{F}$ (as in JADE), it maintains a history memory ($H$) of successful parameters. This allows the algorithm to remember multiple successful search strategies (e.g., one group of individuals might need high mutation, another low) and preserves diversity longer.
#3.  **Weighted Lehmer Mean:** The parameter updates are weighted by the magnitude of fitness improvement, ensuring that parameters resulting in *significant* progress have more influence on future generations.
#4.  **Robust Bound Handling:** Uses a reflection/bounce-back method closer to physical constraints rather than simple clipping, preventing the population from sticking to the edges of the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements L-SHADE: Linear Population Size Reduction with 
    Success-History based Adaptive Differential Evolution.
    """
    
    # --- Helper Functions ---
    def get_cauchy(loc, scale, size):
        """Generates Cauchy distributed values."""
        return loc + scale * np.tan(np.pi * (np.random.rand(size) - 0.5))

    def get_normal(loc, scale, size):
        """Generates Normal distributed values."""
        return np.random.normal(loc, scale, size)

    # --- Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # L-SHADE Parameters (CEC 2014 Standard Tuning)
    N_init = 18 * dim  # Initial population size
    N_min = 4          # Minimum population size
    H_size = 6         # Memory size for history adaptation
    
    # Initialize Population
    pop_size = N_init
    population = np.random.uniform(min_b, max_b, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    # Sort for convenience
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    best_fitness = fitness[0]
    best_solution = population[0].copy()
    
    # Memory Initialization
    # M_cr and M_f store historical successful parameters
    M_cr = np.ones(H_size) * 0.5
    M_f = np.ones(H_size) * 0.5
    k_mem = 0  # Memory index pointer
    
    # Archive Initialization
    # Stores inferior solutions to maintain diversity in mutation
    archive = []
    
    # Main Loop
    while True:
        # Check Time constraints
        elapsed = datetime.now() - start_time
        if elapsed >= time_limit:
            return best_fitness

        # Calculate progress ratio (0.0 to 1.0) for Linear Reduction
        # We reserve a tiny buffer (0.05s) to ensure the last loop finishes
        progress = min(1.0, elapsed.total_seconds() / max(1e-9, max_time - 0.05))
        
        # --- 1. Linear Population Size Reduction (LPSR) ---
        new_pop_size = int(round(N_init + (N_min - N_init) * progress))
        new_pop_size = max(N_min, new_pop_size)
        
        if pop_size > new_pop_size:
            # Shrink population: remove worst individuals (already sorted)
            remove_count = pop_size - new_pop_size
            # The current population is sorted by fitness at the start of loop
            population = population[:new_pop_size]
            fitness = fitness[:new_pop_size]
            
            # Resize archive allowed size (A_size = pop_size)
            curr_archive_size = len(archive)
            if curr_archive_size > new_pop_size:
                # Randomly remove from archive
                idxs = np.random.choice(curr_archive_size, new_pop_size, replace=False)
                archive = [archive[i] for i in idxs]
                
            pop_size = new_pop_size

        # --- 2. Parameter Generation ---
        # Pick random indices from memory
        r_indices = np.random.randint(0, H_size, pop_size)
        mu_cr = M_cr[r_indices]
        mu_f = M_f[r_indices]
        
        # Generate CR (Normal Distribution)
        cr = get_normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        # If CR is -1 (from specialized implementations), keep 0, but here clip 0-1 is standard.
        
        # Generate F (Cauchy Distribution)
        f = get_cauchy(mu_f, 0.1, pop_size)
        # Handle F constraints
        f[f > 1] = 1.0
        # If F <= 0, regenerate until positive
        neg_mask = f <= 0
        while np.any(neg_mask):
            f[neg_mask] = get_cauchy(mu_f[neg_mask], 0.1, np.sum(neg_mask))
            f[f > 1] = 1.0
            neg_mask = f <= 0
            
        # --- 3. Mutation (current-to-pbest/1) ---
        # Sort is required for p-best selection
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # p-best selection (random from top p%)
        p_val = max(2.0 / pop_size, 0.2 * (1 - progress) + 0.05) # Dynamic p reduces over time
        p_limit = max(2, int(pop_size * p_val))
        
        pbest_indices = np.random.randint(0, p_limit, pop_size)
        x_pbest = population[pbest_indices]
        
        # r1 selection (random from population, != i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Fix self-selection collision
        collision = (r1_indices == np.arange(pop_size))
        r1_indices[collision] = (r1_indices[collision] + 1) % pop_size
        x_r1 = population[r1_indices]
        
        # r2 selection (random from Union(Population, Archive), != i, != r1)
        # Prepare pool
        if len(archive) > 0:
            archive_arr = np.array(archive)
            pool = np.vstack((population, archive_arr))
        else:
            pool = population
            
        pool_size = len(pool)
        r2_indices = np.random.randint(0, pool_size, pop_size)
        
        # Verify r2 distinctness requires a bit of logic, simplified here for speed:
        # Just ensure r2 != r1. In high dimensions/pop, collision with i is rare.
        # Collision with r1 is critical.
        collision_r2 = (r2_indices == r1_indices) 
        if pop_size < pool_size: # If archive exists
             # Just repick
             while np.any(collision_r2):
                 r2_indices[collision_r2] = np.random.randint(0, pool_size, np.sum(collision_r2))
                 collision_r2 = (r2_indices == r1_indices)
        
        x_r2 = pool[r2_indices]
        
        # Mutation Vector calculation
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        # X_i is just population (sorted)
        diff_1 = x_pbest - population
        diff_2 = x_r1 - x_r2
        v = population + f[:, None] * diff_1 + f[:, None] * diff_2
        
        # --- 4. Crossover (Binomial) ---
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        # Force at least one dimension
        mask[np.arange(pop_size), j_rand] = True
        
        u = np.where(mask, v, population)
        
        # --- 5. Bound Constraints (Reflection) ---
        # If point is outside, reflect it back: lower + (lower - x)
        lower_mask = u < min_b
        upper_mask = u > max_b
        
        # Reflection logic
        # For lower: amount violated is (min_b - u). New pos = min_b + (min_b - u)
        u[lower_mask] = 2.0 * min_b[np.where(lower_mask)[1]] - u[lower_mask]
        u[upper_mask] = 2.0 * max_b[np.where(upper_mask)[1]] - u[upper_mask]
        
        # If still out (due to extreme mutation), clip
        u = np.clip(u, min_b, max_b)
        
        # --- 6. Selection ---
        # Arrays to store successful updates
        successful_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        
        new_fitness = np.zeros(pop_size)
        
        # Evaluate children
        for i in range(pop_size):
            # Strict time check inside loop to catch slow functions
            if (datetime.now() - start_time) >= time_limit:
                 return best_fitness
                 
            trial_fit = func(u[i])
            new_fitness[i] = trial_fit
            
            if trial_fit < fitness[i]:
                successful_mask[i] = True
                diff_fitness[i] = fitness[i] - trial_fit
                
                # Add parent to archive
                archive.append(population[i].copy())
                
                # Update population
                population[i] = u[i]
                fitness[i] = trial_fit
                
                if trial_fit < best_fitness:
                    best_fitness = trial_fit
                    best_solution = u[i].copy()
            else:
                # Keep parent (implied, as we modified population in place only on success)
                pass

        # Maintain Archive Size
        while len(archive) > pop_size:
            # Delete random elements to maintain size = pop_size
            # (In standard L-SHADE, archive size scales with pop_size)
            idx_to_remove = np.random.randint(0, len(archive))
            del archive[idx_to_remove]

        # --- 7. Memory Update (SHADE Adaptive Logic) ---
        # Only if there were successful individuals
        num_success = np.sum(successful_mask)
        if num_success > 0:
            success_f = f[successful_mask]
            success_cr = cr[successful_mask]
            w = diff_fitness[successful_mask]
            
            # Weighted Lehmer Mean
            total_w = np.sum(w)
            if total_w > 0:
                weights = w / total_w
                
                # Update Memory CR
                # If max(cr) is 0 (unlikely), set to 0, else mean
                if np.max(success_cr) == 0:
                    m_cr_new = 0
                else:
                    m_cr_new = np.sum(weights * success_cr)
                
                # Update Memory F (Lehmer Mean: sum(w*f^2) / sum(w*f))
                sum_wf = np.sum(weights * success_f)
                if sum_wf > 0:
                    m_f_new = np.sum(weights * (success_f ** 2)) / sum_wf
                else:
                    m_f_new = 0.5

                # Update cyclic memory
                M_cr[k_mem] = m_cr_new
                M_f[k_mem] = m_f_new
                k_mem = (k_mem + 1) % H_size

    return best_fitness
