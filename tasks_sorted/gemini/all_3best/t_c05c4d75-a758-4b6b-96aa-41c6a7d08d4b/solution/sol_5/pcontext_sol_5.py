#The following algorithm implements **L-SHADE (Linear Success-History Adaptive Differential Evolution)**, a highly advanced variant of Differential Evolution that consistently outperforms standard DE and jDE in benchmarks (e.g., CEC competitions).
#
#### **Key Improvements over Previous Algorithms:**
#
#1.  **L-SHADE "Engine"**: Instead of the random parameter adaptation of jDE (Algorithm 1), this uses **History-Based Adaptation**. It maintains a memory ($M_{F}, M_{CR}$) of successful control parameters from previous generations and uses a weighted Lehmer mean to bias future parameters towards successful values. This dramatically improves convergence speed and robustness.
#2.  **Linear Population Size Reduction (LPSR)**: Similar to Algorithm 1, it linearly reduces the population size from a large initial value (for exploration) to a small value (for exploitation) as time progresses. This ensures the computational budget is always used optimally.
#3.  **Adaptive `current-to-pbest/1` Mutation**: The "greediness" of the mutation strategy ($p$-value) scales linearly with time. It starts high (diverse) and decreases (convergent), mimicking the LPSR curve.
#4.  **Integrated Archive**: An external archive of inferior solutions is maintained and utilized in the mutation step (`current-to-pbest/1` uses $X_{r2}$ from the union of Population and Archive). This preserves diversity without slowing down convergence.
#5.  **Smart Restart**: A restart mechanism is included to escape local optima if convergence is detected early. Unlike a blind restart, it respects the current "exploitation phase" by maintaining the reduced population size, preventing the algorithm from wasting time re-exploring the entire space when the budget is tight.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive Differential Evolution)
    with Time-Dependent Population Reduction and a Restart Mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    # Safety buffer to ensure the function returns before the hard timeout
    end_time = start_time + time_limit - timedelta(milliseconds=100)

    # --- Hyperparameters ---
    # Initial Population: Large size for exploration (approx 18 * dim)
    # Clipped to a reasonable range to ensure performance
    pop_size_init = int(np.clip(18 * dim, 30, 200))
    pop_size_min = 4  # Minimum population size at the end
    
    # SHADE Memory parameters
    H = 5  # Size of historical memory
    
    # Restart triggers
    stall_limit = 25
    tol_std = 1e-8

    # --- Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Population
    current_pop_size = pop_size_init
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # Global Best
    best_fitness = float('inf')
    best_sol = None

    # Evaluate Initial Population
    for i in range(current_pop_size):
        if datetime.now() >= end_time:
            return best_fitness if best_fitness != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()

    # Sort population by fitness (needed for p-best selection)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]

    # Initialize SHADE Memory (M_F and M_CR set to 0.5)
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer

    # External Archive
    archive = []
    
    stall_counter = 0

    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        if now >= end_time:
            return best_fitness

        # Calculate time progress (0.0 to 1.0)
        elapsed_sec = (now - start_time).total_seconds()
        progress = elapsed_sec / max_time
        if progress > 1.0: progress = 1.0

        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on remaining time
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        if target_size < pop_size_min:
            target_size = pop_size_min

        # Reduce population if needed
        if current_pop_size > target_size:
            current_pop_size = target_size
            # Since pop is sorted by fitness, slicing keeps the best individuals
            pop = pop[:current_pop_size]
            fitness = fitness[:current_pop_size]
            
            # Resize Archive to match current population size (maintain 1:1 ratio)
            if len(archive) > current_pop_size:
                del archive[current_pop_size:]

        # 2. Adaptive Parameter Generation (SHADE)
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, current_pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]

        # Generate CR ~ Normal(M_CR, 0.1)
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)

        # Generate F ~ Cauchy(M_F, 0.1)
        F = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
        
        # Handle F constraints (F > 0 is required)
        retry_mask = F <= 0
        while np.any(retry_mask):
            n_retry = np.sum(retry_mask)
            F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(n_retry)
            retry_mask = F <= 0
        F = np.clip(F, 0.0, 1.0) # Clip F > 1 to 1.0

        # 3. Mutation: current-to-pbest/1
        # p linearly decreases from 0.2 (exploration) to 0.05 (exploitation)
        p = 0.2 - 0.15 * progress
        p = max(p, 2.0 / current_pop_size) # Ensure p-best group has at least 2

        # Select X_pbest
        num_pbest = int(p * current_pop_size)
        num_pbest = max(num_pbest, 2)
        pbest_indices = np.random.randint(0, num_pbest, current_pop_size)
        X_pbest = pop[pbest_indices]

        # Select X_r1 (distinct from i)
        r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
        for i in range(current_pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, current_pop_size)
        X_r1 = pop[r1_indices]

        # Select X_r2 (distinct from i and r1, from Union(Pop, Archive))
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        
        len_union = len(union_pop)
        r2_indices = np.random.randint(0, len_union, current_pop_size)
        for i in range(current_pop_size):
            # If r2 points to current population, check distinctness
            while (r2_indices[i] < current_pop_size and (r2_indices[i] == i or r2_indices[i] == r1_indices[i])):
                r2_indices[i] = np.random.randint(0, len_union)
        X_r2 = union_pop[r2_indices]

        # Compute Mutant Vectors
        F_col = F[:, None]
        V = pop + F_col * (X_pbest - pop) + F_col * (X_r1 - X_r2)
        V = np.clip(V, min_b, max_b)

        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(current_pop_size, dim)
        mask = rand_vals < CR[:, None]
        j_rand = np.random.randint(0, dim, current_pop_size)
        mask[np.arange(current_pop_size), j_rand] = True
        
        U = np.where(mask, V, pop)

        # 5. Selection and Memory Update
        succ_F = []
        succ_CR = []
        diff_fit = []
        new_archive_cands = []
        improved_any = False

        for i in range(current_pop_size):
            if datetime.now() >= end_time:
                return best_fitness

            f_new = func(U[i])

            if f_new <= fitness[i]:
                # Collect successful parameters
                if f_new < fitness[i]:
                    new_archive_cands.append(pop[i].copy())
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    diff_fit.append(fitness[i] - f_new)
                    improved_any = True

                fitness[i] = f_new
                pop[i] = U[i]

                if f_new < best_fitness:
                    best_fitness = f_new
                    best_sol = U[i].copy()

        # Update Archive (Random Replacement)
        for cand in new_archive_cands:
            if len(archive) < current_pop_size:
                archive.append(cand)
            else:
                if current_pop_size > 0:
                    ridx = np.random.randint(0, current_pop_size)
                    archive[ridx] = cand

        # Update History Memory (Weighted Lehmer Mean)
        if len(diff_fit) > 0:
            w = np.array(diff_fit)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                s_f = np.array(succ_F)
                s_cr = np.array(succ_CR)
                
                mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                mean_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                k_mem = (k_mem + 1) % H

        # Sort population for next generation
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

        # 6. Restart Mechanism
        if not improved_any:
            stall_counter += 1
        else:
            stall_counter = 0

        std_fit = np.std(fitness)
        
        # Trigger restart if stalled or converged, but only if enough time remains (>10%)
        if (stall_counter >= stall_limit or std_fit < tol_std) and progress < 0.9:
            stall_counter = 0
            
            # Soft Restart: Keep the current reduced population size to avoid exploding cost
            pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
            fitness = np.full(current_pop_size, float('inf'))
            
            # Elitism: Keep the global best
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset History and Archive
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            archive = []
            
            # Re-evaluate new population (skipping index 0)
            for i in range(1, current_pop_size):
                if datetime.now() >= end_time:
                    return best_fitness
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            # Re-sort
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

    return best_fitness
