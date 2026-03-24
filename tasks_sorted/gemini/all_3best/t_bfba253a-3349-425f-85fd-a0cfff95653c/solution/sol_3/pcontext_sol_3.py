#The previous results show that the simplest algorithm (Standard Differential Evolution, Algorithm 1) performed the best (15.94), beating the more complex restart strategy (Algorithm 3) and the self-adaptive jDE (Algorithm 2).
#
#The superior performance of Algorithm 1 suggests that maintaining population diversity is crucial, but its simple "rand/1" mutation strategy might be slow to converge. Algorithm 2 tried to adapt parameters but likely suffered from the high variance of its random adaptation logic. Algorithm 3's "current-to-best" strategy likely converged prematurely.
#
#To improve upon the best result (15.94), I propose an implementation of **SHADE (Success-History based Adaptive Differential Evolution)**. SHADE is a state-of-the-art improvement over standard DE and jDE.
#
#**Key Improvements in SHADE:**
#1.  **History-Based Parameter Adaptation**: Instead of randomizing parameters or using fixed values, SHADE maintains a memory ($H$) of successful $F$ (scaling factor) and $CR$ (crossover rate) values from previous generations. New parameters are sampled from distributions centered on these successful historical means. This allows the algorithm to learn the specific requirements of the objective function landscape rapidly.
#2.  **Current-to-pBest Mutation**: This strategy balances exploration and exploitation better than "current-to-best". It steers the population towards a random individual selected from the top $p\%$ of best solutions, preventing immediate collapse into local optima while still converging faster than "rand/1".
#3.  **External Archive**: SHADE maintains an archive of recently replaced (inferior) solutions. These are used in the mutation step (selection of $r2$) to maintain diversity and prevent the difference vectors from becoming too small too quickly.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution).
    SHADE adapts the control parameters (F and CR) during the optimization process based on
    the success of previous generations and utilizes an external archive to maintain diversity.
    """
    start_time = datetime.now()
    # Safety margin to ensure we return before the strict timeout
    time_limit = timedelta(seconds=max_time * 0.98) 

    # --- Configuration ---
    # Population size: SHADE typically uses N=100 or ~18*D. 
    # We clip the size to ensure the algorithm runs sufficient generations 
    # even within a short time limit.
    pop_size = int(np.clip(18 * dim, 30, 100))
    
    # SHADE Memory Parameters
    H = 5 # Size of the historical memory
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate (starts at 0.5)
    mem_M_F = np.full(H, 0.5)  # Memory for Scaling Factor (starts at 0.5)
    k_mem = 0 # Memory index pointer
    
    # External Archive to store inferior solutions (maintains diversity)
    archive = []
    max_archive_size = pop_size

    # --- Initialization (Latin Hypercube Sampling) ---
    # LHS provides a more uniform initial coverage than random sampling
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        # Generate stratified samples for each dimension
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        points = np.random.uniform(edges[:-1], edges[1:])
        population[:, d] = np.random.permutation(points)
        
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness
        
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val

    # --- Main Optimization Loop ---
    while True:
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        # Sort population for 'current-to-pbest' selection strategy.
        # Individuals are sorted by fitness to identify the top p% best.
        sort_idx = np.argsort(fitness)
        sorted_pop = population[sort_idx]
        sorted_fitness = fitness[sort_idx]
        
        # Containers to record parameters that lead to improvement
        S_CR = []
        S_F = []
        S_df = [] # Fitness improvement amounts (weights)
        
        # --- Parameter Generation ---
        # Vectorized generation of candidate CR and F values based on Memory
        r_idxs = np.random.randint(0, H, pop_size)
        
        # CR_i ~ Normal(M_CR[r_i], 0.1)
        CR_g = np.random.normal(mem_M_CR[r_idxs], 0.1)
        CR_g = np.clip(CR_g, 0.0, 1.0)
        
        # F_i ~ Cauchy(M_F[r_i], 0.1)
        F_g = []
        for idx in r_idxs:
            f_val = -1
            # F must be positive; regenerate if <= 0 (Cauchy distribution has heavy tails)
            while f_val <= 0:
                f_val = mem_M_F[idx] + 0.1 * np.random.standard_cauchy()
            # F capped at 1.0
            F_g.append(min(1.0, f_val))
        F_g = np.array(F_g)
        
        # Prepare for evolution
        new_pop = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Union of Population and Archive for mutation vector selection (r2)
        if len(archive) > 0:
            pop_archive_pool = np.vstack((population, np.array(archive)))
        else:
            pop_archive_pool = population
        len_pool = len(pop_archive_pool)

        # --- Evolution Cycle ---
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # Mutation Strategy: current-to-pbest/1/bin
            # V = X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
            
            # Select pbest randomly from top p% (p in [0.05, 0.2])
            # We use top 10% (min 2 individuals)
            p_val = max(2, int(pop_size * 0.1)) 
            pbest_idx = np.random.randint(0, p_val)
            x_pbest = sorted_pop[pbest_idx]
            
            # Select r1 distinct from i
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select r2 distinct from i and r1, from the Pool (Population + Archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len_pool)
            x_r2 = pop_archive_pool[r2]
            
            x_i = population[i]
            
            # Calculate mutant vector
            mutant = x_i + F_g[i] * (x_pbest - x_i) + F_g[i] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # Crossover: Binomial
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < CR_g[i]
            mask[j_rand] = True # Ensure at least one dimension changes
            trial = np.where(mask, mutant, x_i)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Trial is better or equal
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                if f_trial < fitness[i]:
                    # Strict improvement: record success parameters
                    S_CR.append(CR_g[i])
                    S_F.append(F_g[i])
                    S_df.append(fitness[i] - f_trial)
                    
                    # Move the replaced parent to the archive
                    archive.append(x_i.copy())
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
            else:
                # Retain parent
                new_pop[i] = x_i
                new_fitness[i] = fitness[i]
        
        # Apply population updates
        population = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size (Remove random members if full)
        while len(archive) > max_archive_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # --- Update Historical Memory (SHADE Logic) ---
        if len(S_CR) > 0:
            S_CR = np.array(S_CR)
            S_F = np.array(S_F)
            S_df = np.array(S_df)
            
            # Calculate weights based on fitness improvement amount
            if np.sum(S_df) > 0:
                w = S_df / np.sum(S_df)
                
                # Update Memory CR (Weighted Arithmetic Mean)
                m_cr_new = np.sum(w * S_CR)
                mem_M_CR[k_mem] = m_cr_new
                
                # Update Memory F (Weighted Lehmer Mean)
                # Lehmer mean gives more weight to larger F values, helpful for exploration
                m_f_new = np.sum(w * (S_F ** 2)) / np.sum(w * S_F)
                mem_M_F[k_mem] = m_f_new
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % H

    return best_fitness
