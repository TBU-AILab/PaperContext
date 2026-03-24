#The following algorithm implements **L-SHADE (Linear Success-History based Adaptive Differential Evolution) with Restart**.
#
#**Improvements and Strategy:**
#1.  **L-SHADE (State-of-the-Art):** This is widely considered one of the best-performing Differential Evolution variants. It adapts the mutation factor ($F$) and crossover rate ($CR$) using a historical memory of successful parameters, tailored to the specific optimization landscape.
#2.  **Linear Population Reduction:** The population size is dynamically reduced from a large initial size (exploration) to a small size (exploitation) as time progresses. This forces the algorithm to converge efficiently within the time limit.
#3.  **External Archive:** It maintains an archive of inferior solutions to preserve diversity in the mutation equation ($current-to-pbest/1$).
#4.  **Restart Mechanism:** If the population converges (low variance) or becomes too small before the time runs out, the algorithm restarts with a fresh population. Crucially, it employs **Elitism** by injecting the best-found solution into the new population, ensuring monotonic improvement.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive 
    Differential Evolution) with Restart.
    """
    start_time = datetime.now()
    timeout = timedelta(seconds=max_time)
    
    def check_timeout():
        return (datetime.now() - start_time) >= timeout

    # 1. Bounds Processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # 2. Global State (Tracked across restarts)
    global_best_val = float('inf')
    global_best_vec = None
    
    # 3. Hyperparameters
    # Initial population size: 18 * dim is a standard heuristic for L-SHADE
    init_pop_size = 18 * dim
    min_pop_size = 4
    memory_size = 5
    
    # 4. Main Restart Loop
    # We restart if the population converges or gets too small, utilizing all available time.
    while not check_timeout():
        
        # --- Initialization for this Restart ---
        current_pop_size = init_pop_size
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        
        # Elitism: Inject global best (if any) to guide the new search
        if global_best_vec is not None:
            pop[0] = global_best_vec
            
        fitness = np.full(current_pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(current_pop_size):
            if check_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        # L-SHADE Memory Initialization
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        archive = []
        
        # --- Evolutionary Cycle ---
        while not check_timeout():
            
            # A. Population Size Reduction (L-SHADE Strategy)
            # Calculate target size based on consumed global time
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            progress = elapsed_sec / max_time
            # Linear reduction formula
            target_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
            target_size = max(min_pop_size, target_size)
            
            if current_pop_size > target_size:
                # Reduce population: Keep the best individuals
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:target_size]]
                fitness = fitness[sort_idx[:target_size]]
                current_pop_size = target_size
                
            # B. Parameter Adaptation
            # Pick memory index for each individual
            r_idx = np.random.randint(0, memory_size, current_pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            # Generate CR ~ Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(M_F, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
            # Clip F
            f[f > 1.0] = 1.0
            f[f <= 0.0] = 0.5 # Fallback for non-positive F
            
            # C. Mutation: DE/current-to-pbest/1
            # Sort population to identify top p-best
            sorted_indices = np.argsort(fitness)
            # p_best rate: top 10% (min 2 individuals)
            num_pbest = max(2, int(0.1 * current_pop_size))
            top_p_indices = sorted_indices[:num_pbest]
            
            # Select x_pbest
            pbest_choices = np.random.choice(top_p_indices, current_pop_size)
            x_pbest = pop[pbest_choices]
            
            # Select x_r1 (random from pop)
            r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
            x_r1 = pop[r1_indices]
            
            # Select x_r2 (random from Union(Pop, Archive))
            if len(archive) > 0:
                archive_arr = np.array(archive)
                pop_archive = np.vstack((pop, archive_arr))
            else:
                pop_archive = pop
            
            r2_indices = np.random.randint(0, len(pop_archive), current_pop_size)
            x_r2 = pop_archive[r2_indices]
            
            # Compute Mutant V = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # D. Crossover: Binomial
            rand_vals = np.random.rand(current_pop_size, dim)
            j_rand = np.random.randint(0, dim, current_pop_size)
            mask = rand_vals <= cr[:, np.newaxis]
            mask[np.arange(current_pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection & Memory Update
            trial_fitness = np.full(current_pop_size, float('inf'))
            improved_mask = np.zeros(current_pop_size, dtype=bool)
            diff_f = np.zeros(current_pop_size)
            
            # Evaluate Trials
            for i in range(current_pop_size):
                if check_timeout(): return global_best_val
                
                val = func(trial[i])
                trial_fitness[i] = val
                
                if val <= fitness[i]:
                    improved_mask[i] = True
                    diff_f[i] = fitness[i] - val
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_vec = trial[i].copy()
                        
            # Update Population and Archive
            if np.any(improved_mask):
                # Save overwritten parents to archive
                parents_to_archive = pop[improved_mask]
                for p_vec in parents_to_archive:
                    archive.append(p_vec.copy())
                
                # Update survivors
                pop[improved_mask] = trial[improved_mask]
                fitness[improved_mask] = trial_fitness[improved_mask]
                
                # Update Historical Memory (Weighted Lehmer Mean)
                succ_f = f[improved_mask]
                succ_cr = cr[improved_mask]
                succ_diff = diff_f[improved_mask]
                
                total_diff = np.sum(succ_diff)
                if total_diff > 0:
                    weights = succ_diff / total_diff
                    
                    # Mean F (Lehmer)
                    mean_f = np.sum(weights * (succ_f ** 2)) / np.sum(weights * succ_f)
                    # Mean CR (Weighted Arithmetic)
                    mean_cr = np.sum(weights * succ_cr)
                    
                    M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                    
                    k_mem = (k_mem + 1) % memory_size
            
            # Manage Archive Size (Keep <= current_pop_size)
            while len(archive) > current_pop_size:
                # Randomly remove elements
                del archive[np.random.randint(0, len(archive))]
                
            # F. Convergence Check
            # If standard deviation of fitness is very low, restart to explore elsewhere
            if np.std(fitness) < 1e-8:
                break
                
    return global_best_val
