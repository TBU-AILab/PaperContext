#Here is a fully operational, self-contained implementation of the **L-SHADE-E (Linear Population Size Reduction Success-History Adaptation Differential Evolution)** algorithm, adapted specifically for time-bound optimization.
#
#### Improvements over the previous JADE implementation:
#1.  **Linear Population Size Reduction (LPSR):** This is the most critical improvement for a fixed-time constraint. The algorithm starts with a large population to explore the global space and linearly reduces the population size as time elapses. This forces the algorithm to shift from exploration to intense exploitation (fine-tuning) exactly as the deadline approaches.
#2.  **Historical Memory (H):** Instead of a single mean for mutation/crossover, it maintains a history of successful parameters ($M_{CR}, M_F$). This allows the algorithm to recall different strategies that worked well for different stages of the optimization (e.g., high $F$ early on, low $F$ later).
#3.  **External Archive:** It preserves a set of recently replaced inferior solutions. These are used during mutation to maintain diversity without needing a huge active population.
#4.  **Weighted Parameter Updates:** Parameter updates are weighted by the magnitude of fitness improvement, giving more influence to parameter settings that produced significant jumps in performance.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE-E (Linear Population Size Reduction Success-History Adaptive DE)
    Adapted for time-constrained optimization.
    """
    # --- Configuration ---
    # Initial Population: Start high for exploration
    p_init = int(round(18 * dim))
    # Ensure starting population isn't too massive for slow functions, nor too small
    p_init = np.clip(p_init, 30, 200) 
    
    # Minimum population for final exploitation
    p_min = 4
    
    # Archive size (usually equal to initial population)
    arc_rate = 1.4
    arc_size = int(p_init * arc_rate)
    
    # Memory size for historical adaptation
    h_mem = 5
    
    # Time Management
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population init (Random Uniform)
    pop_size = p_init
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    best_val = float('inf')
    best_vec = None
    
    for i in range(pop_size):
        if datetime.now() >= end_time: return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()
            
    # --- L-SHADE Memory Initialization ---
    # M_cr and M_f store historical successful means
    mem_cr = np.full(h_mem, 0.5)
    mem_f = np.full(h_mem, 0.5)
    k_mem = 0  # Memory index pointer
    
    # Archive (External population)
    archive = np.empty((0, dim))
    
    # --- Main Loop ---
    while True:
        # 1. Time Check & Progress Calculation
        curr_time = datetime.now()
        if curr_time >= end_time:
            return best_val
        
        # Calculate progress ratio (0.0 to 1.0)
        elapsed = (curr_time - start_time).total_seconds()
        progress = min(1.0, elapsed / max_time)
        
        # 2. Linear Population Size Reduction (LPSR)
        # Linearly decay population size from p_init to p_min based on time
        plan_pop_size = int(round((p_min - p_init) * progress + p_init))
        
        if pop_size > plan_pop_size:
            # Reduction needed: Delete worst individuals
            n_remove = pop_size - plan_pop_size
            # argsort ascends, so worst (highest fitness) are at the end
            sort_idx = np.argsort(fitness)
            # Keep top 'plan_pop_size'
            keep_idx = sort_idx[:plan_pop_size]
            
            population = population[keep_idx]
            fitness = fitness[keep_idx]
            pop_size = plan_pop_size
            
            # Ensure archive doesn't grow indefinitely if pop shrinks
            curr_arc_limit = int(pop_size * arc_rate)
            if len(archive) > curr_arc_limit:
                # Randomly remove excess from archive
                keep_arc = np.random.choice(len(archive), curr_arc_limit, replace=False)
                archive = archive[keep_arc]

        # 3. Sort for current-to-pbest
        # We need the population sorted to pick the top p-best vectors
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Sync best (index 0 after sort is current best)
        if fitness[0] < best_val:
            best_val = fitness[0]
            best_vec = population[0].copy()
            
        # 4. Generate Parameters (F and CR)
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, h_mem, pop_size)
        m_cr_selected = mem_cr[r_idx]
        m_f_selected = mem_f[r_idx]
        
        # Generate CR ~ Normal(M_cr, 0.1)
        cr = np.random.normal(m_cr_selected, 0.1)
        cr = np.clip(cr, 0, 1)
        # Special case: if M_cr is terminal value -1 (rarely used here), keep fixed
        
        # Generate F ~ Cauchy(M_f, 0.1)
        # Cauchy: loc + scale * tan(uniform) or using standard_cauchy
        f = m_f_selected + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Check constraints for F
        # If F > 1, clamp to 1. If F <= 0, regenerate until > 0
        # Vectorized regeneration for F <= 0
        bad_f_idx = np.where(f <= 0)[0]
        while len(bad_f_idx) > 0:
            m_f_retry = m_f_selected[bad_f_idx]
            f[bad_f_idx] = m_f_retry + 0.1 * np.random.standard_cauchy(len(bad_f_idx))
            bad_f_idx = np.where(f <= 0)[0]
        f = np.clip(f, 0, 1)
        
        # 5. Mutation: current-to-pbest/1 (with Archive)
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        # X_r2 is selected from Union(Population, Archive)
        
        # p-best selection: top p percent (randomly chosen from top)
        # p reduces linearly from 0.2 to 0.05 to encourage convergence
        # or fixed p_rate = 2/pop_size is common in L-SHADE. Let's use adaptive p.
        p_val = max(2.0/pop_size, 0.2 * (1 - progress))
        p_num = max(1, int(round(pop_size * p_val)))
        
        pbest_indices = np.random.randint(0, p_num, pop_size) # Indices into sorted population
        x_pbest = population[pbest_indices]
        
        # r1: Random from population (distinct from current i ideally, ignored for speed)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # r2: Random from Union(Pop, Archive)
        n_arc = len(archive)
        union_size = pop_size + n_arc
        r2_indices = np.random.randint(0, union_size, pop_size)
        
        # Build union matrix
        if n_arc > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
            
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutation Vectors
        f_col = f[:, np.newaxis]
        v = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        
        # 6. Crossover (Binomial)
        # Mask: True if crossover happens
        rand_j = np.random.rand(pop_size, dim)
        mask = rand_j < cr[:, np.newaxis]
        
        # Force at least one dimension
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        u = np.where(mask, v, population)
        
        # Bound Constraints (Reflection/Clamping)
        # Simple clamping
        u = np.clip(u, min_b, max_b)
        
        # 7. Selection & Memory Update
        fitness_u = np.empty(pop_size)
        
        succ_scr = [] # Successful CR
        succ_sf = []  # Successful F
        diff_f = []   # Fitness Improvements
        
        # Evaluate off-loop to check time frequently
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            
            y = func(u[i])
            fitness_u[i] = y
            
            if y < fitness[i]:
                # Success!
                # 1. Add replaced parent to archive
                if len(archive) < int(pop_size * arc_rate):
                    archive = np.vstack((archive, population[i])) if len(archive) > 0 else np.array([population[i]])
                else:
                    # Archive full: Replace random
                    if len(archive) > 0:
                        arc_idx = np.random.randint(0, len(archive))
                        archive[arc_idx] = population[i]
                
                # 2. Record memory data
                diff = fitness[i] - y
                succ_scr.append(cr[i])
                succ_sf.append(f[i])
                diff_f.append(diff)
                
                # 3. Update Population
                population[i] = u[i]
                fitness[i] = y
                
                if y < best_val:
                    best_val = y
                    best_vec = u[i].copy()
                    
        # 8. Update Historical Memory
        if len(succ_scr) > 0:
            succ_scr = np.array(succ_scr)
            succ_sf = np.array(succ_sf)
            diff_f = np.array(diff_f)
            
            # Weights based on fitness improvement
            total_diff = np.sum(diff_f)
            if total_diff == 0:
                weights = np.ones(len(diff_f)) / len(diff_f)
            else:
                weights = diff_f / total_diff
            
            # Weighted Lehmer Mean for F
            # mean = sum(w * f^2) / sum(w * f)
            denom = np.sum(weights * succ_sf)
            if denom == 0:
                mean_f = 0.5
            else:
                mean_f = np.sum(weights * (succ_sf ** 2)) / denom
            
            # Weighted Mean for CR
            mean_cr = np.sum(weights * succ_scr)
            
            # Update Memory slot (k_mem)
            mem_f[k_mem] = mean_f
            mem_cr[k_mem] = mean_cr
            
            # Increment/Cycle Memory pointer
            k_mem = (k_mem + 1) % h_mem
            
    return best_val
