#Here is a fully operational, self-contained implementation of **L-SHADE with Linear Population Size Reduction (LPSR)**.
#
#### Why this is better than JADE:
#1.  **Linear Population Size Reduction (LPSR):** The algorithm starts with a large population to explore the global space and linearly reduces the population size over time. This forces the algorithm to switch from exploration to aggressive exploitation as time runs out, converging much tighter than a fixed-size population.
#2.  **History-Based Adaptation (L-SHADE):** Instead of a single mean for parameters $F$ and $CR$ (like JADE), this uses a memory archive of successful parameters. This allows the algorithm to recall multiple successful search strategies (e.g., one setting for exploring valleys, another for refining peaks).
#3.  **External Archive:** It maintains an archive of inferior solutions recently replaced. This preserves diversity in the mutation equation (`current-to-pbest/1` uses the archive for the difference vector), preventing premature convergence.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Linear Population Size Reduction (LPSR) implementation.
    Minimizes a black-box function within a given time limit.
    """
    
    # --- Helper Functions ---
    def get_cauchy(loc, scale, size):
        """Generates Cauchy distributed values: loc + scale * tan(pi * (rand - 0.5))"""
        return loc + scale * np.tan(np.pi * (np.random.rand(size) - 0.5))

    def get_lehmer_mean(values):
        """Calculates Lehmer Mean (efficient for F parameter adaptation)"""
        values = np.array(values)
        numer = np.sum(values**2)
        denom = np.sum(values)
        return numer / denom if denom != 0 else 0

    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Initialization Parameters ---
    # Initial population size: Start large for exploration
    # Reduces linearly to min_pop_size (4)
    r_init = 18 
    pop_size = int(r_init * dim)
    min_pop_size = 4
    
    # Initialize Population
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    diff_b = max_b - min_b
    
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.array([float('inf')] * pop_size)
    
    # Evaluate initial population
    best_val = float('inf')
    best_vec = None
    
    # Evaluate initial batch
    # We check time strictly here to ensure we don't overrun on slow functions
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # If we timeout during init, return best found so far
            return best_val if best_vec is not None else float('inf')
            
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()
            
    # --- L-SHADE Memory Initialization ---
    memory_size = 5
    # Memory for Mean CR (Crossover Rate) and Mean F (Scaling Factor)
    m_cr = np.full(memory_size, 0.5) 
    m_f = np.full(memory_size, 0.5)
    k_mem = 0  # Memory index counter
    
    # External Archive (stores replaced individuals to maintain diversity)
    archive = []
    
    # Maximum estimated evaluations (dynamic budget based on time is harder, 
    # so we map PSR (Population Size Reduction) to Time Progress)
    initial_pop_size = pop_size
    
    # --- Main Loop ---
    while True:
        # Check Time Progress
        now = datetime.now()
        if now >= end_time:
            return best_val
        
        # Calculate progress ratio (0.0 to 1.0)
        elapsed = (now - start_time).total_seconds()
        progress = elapsed / max_time
        
        # --- 1. Linear Population Size Reduction (LPSR) ---
        # Calculate target population size based on time progress
        plan_pop_size = int(np.round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)
        
        # If current population is too big, reduce it by removing worst individuals
        if pop_size > plan_pop_size:
            n_reduce = pop_size - plan_pop_size
            sorting_idx = np.argsort(fitness)
            # Keep the best, discard the worst (at the end of sorted list)
            keep_idx = sorting_idx[:plan_pop_size]
            
            population = population[keep_idx]
            fitness = fitness[keep_idx]
            pop_size = plan_pop_size
            
            # Resize archive if it exceeds new pop_size
            if len(archive) > pop_size:
                # Randomly remove elements from archive to match pop_size
                import random
                del archive[pop_size:]
        
        # --- 2. Parameter Generation ---
        # Generate random indices to select from memory
        r_idxs = np.random.randint(0, memory_size, pop_size)
        
        # Generate CR (Normal Distribution)
        cr = np.random.normal(m_cr[r_idxs], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy Distribution)
        f = get_cauchy(m_f[r_idxs], 0.1, pop_size)
        f = np.clip(f, 0, 1) # Note: L-SHADE usually regenerates if <= 0, but clipping is faster/safer
        
        # --- 3. Mutation: current-to-pbest/1 ---
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p-best selection (top p% random)
        p = max(2.0 / pop_size, 0.11) # p is typically 0.11 in L-SHADE
        top_p_count = max(1, int(np.round(p * pop_size)))
        pbest_indices = sorted_indices[:top_p_count]
        
        # Arrays for vectorization
        pbest_idxs = np.random.choice(pbest_indices, pop_size)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        
        # r2 is selected from Union(Population, Archive)
        n_archive = len(archive)
        n_union = pop_size + n_archive
        r2_raw = np.random.randint(0, n_union, pop_size)
        
        # Prepare Union Matrix
        if n_archive > 0:
            union_pop = np.vstack((population, np.array(archive)))
        else:
            union_pop = population
            
        x_pbest = population[pbest_idxs]
        x_r1 = population[r1_idxs]
        x_r2 = union_pop[r2_raw]
        
        # Reshape f for broadcasting
        f_col = f[:, np.newaxis]
        
        # Mutation Equation
        mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        
        # Boundary Handling (Bounce back if out of bounds, preserves distribution better than clip)
        # Using simple clipping here for stability and speed
        mutant = np.clip(mutant, min_b, max_b)
        
        # --- 4. Crossover (Binomial) ---
        rand_j = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) <= cr[:, np.newaxis]
        # Ensure at least one dimension is taken from mutant
        mask[np.arange(pop_size), rand_j] = True
        
        trial_pop = np.where(mask, mutant, population)
        
        # --- 5. Selection & Memory Update ---
        new_archive_candidates = []
        success_f = []
        success_cr = []
        success_diff = []
        
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val
            
            t_val = func(trial_pop[i])
            
            if t_val < fitness[i]:
                # Improvement found
                # Add parent to archive candidates
                new_archive_candidates.append(population[i].copy())
                
                # Record successful parameters
                success_f.append(f[i])
                success_cr.append(cr[i])
                success_diff.append(fitness[i] - t_val)
                
                # Update Population
                fitness[i] = t_val
                population[i] = trial_pop[i]
                
                if t_val < best_val:
                    best_val = t_val
                    best_vec = trial_pop[i].copy()
                    
        # --- 6. Update Archive ---
        if new_archive_candidates:
            # Add new rejected parents to archive
            archive.extend(new_archive_candidates)
            # If archive too large, remove random elements
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
        # --- 7. Update Historical Memory ---
        if len(success_f) > 0:
            success_f = np.array(success_f)
            success_cr = np.array(success_cr)
            success_diff = np.array(success_diff)
            
            # Weighted Lehmer Mean based on fitness improvement amount
            weights = success_diff / np.sum(success_diff)
            
            # Update Memory F (Lehmer Mean)
            mean_sq_f = np.sum(weights * (success_f**2))
            mean_f = np.sum(weights * success_f)
            lehmer_f = mean_sq_f / mean_f if mean_f > 0 else 0.5
            
            # Update Memory CR (Weighted Arithmetic Mean)
            mean_cr = np.sum(weights * success_cr)
            
            # Update memory slot
            m_f[k_mem] = lehmer_f
            m_cr[k_mem] = mean_cr
            
            # Increment memory index
            k_mem = (k_mem + 1) % memory_size

    return best_val
