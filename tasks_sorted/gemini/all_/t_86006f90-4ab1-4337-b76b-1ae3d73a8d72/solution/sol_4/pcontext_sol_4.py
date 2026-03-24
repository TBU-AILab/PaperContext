#The following is an improved algorithm implementing **Optimized SHADE (Success-History based Adaptive Differential Evolution) with Elitist Restart**.
#
##### Improvements Explanation:
#1.  **Optimized Population Size**: Unlike the previous SHADE implementation (which used a large population of ~18*dim and performed poorly), this version uses a smaller, tighter population (`max(20, 5*dim)`). This modification is based on the success of the jDE algorithm (Algo 3) and ensures rapid convergence within the time limit.
#2.  **Elitist Restart Mechanism**: To prevent the "stagnation" observed in standard DE, this algorithm detects convergence (low variance) and triggers a restart. Crucially, it injects the **global best solution** found so far into the new random population. This allows the algorithm to refine the best solution (exploitation) while simultaneously exploring new basins of attraction.
#3.  **Adaptive History (SHADE)**: It retains the superior parameter adaptation of SHADE (using memory $M_{CR}, M_F$) rather than the random walk of jDE. This allows the algorithm to learn the specific mutation strengths required for the function landscape.
#4.  **Current-to-pBest with Archive**: Uses the robust `current-to-pbest` mutation strategy with an external archive. This balances greediness (moving towards top solutions) with diversity (using archived inferior solutions to create difference vectors), preventing premature convergence better than the simple `current-to-best` strategy.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Optimized SHADE with Elitist Restart.
    Combines the parameter adaptation of SHADE with the aggressive 
    convergence settings of jDE and a robust restart strategy.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Tuned for limited time budgets.
    # A smaller population (similar to the successful jDE run) allows for 
    # more generations and faster convergence than the standard SHADE sizing.
    pop_size = int(max(20, 5 * dim))
    if pop_size > 50: 
        pop_size = 50
    
    # SHADE Memory Parameters
    H = 5                   # Size of the history memory
    
    # Archive Parameters
    archive_size = pop_size # Archive matches population size
    
    # Bounds processing
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Main Restart Loop ---
    # Allows the algorithm to escape local optima by resetting the population
    while True:
        # Check time before starting a new run
        if (datetime.now() - start_time) >= time_limit:
            return best_val
            
        # 1. Initialization
        # Initialize Memory for Adaptive Parameters (History)
        M_cr = np.full(H, 0.5) # Crossover Rate Memory
        M_f = np.full(H, 0.5)  # Scaling Factor Memory
        k_mem = 0              # Memory index pointer
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject the global best solution into the new population
        # This turns the restart into a "refined exploration"
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            # Optimization: Don't re-evaluate the injected best
            if best_sol is not None and i == 0:
                val = best_val
            else:
                val = func(pop[i])
            
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # Initialize Archive (stores discarded solutions to maintain diversity)
        archive = []
        
        # --- Generation Loop ---
        while True:
            # Strict Time Check
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            # 2. Check for Stagnation
            # If population variance is too low, we are stuck in a local optimum.
            # Trigger restart to explore elsewhere.
            if np.std(fitness) < 1e-8:
                break
            
            # Sort population by fitness (required for p-best selection)
            sorted_idxs = np.argsort(fitness)
            pop = pop[sorted_idxs]
            fitness = fitness[sorted_idxs]
            
            # 3. Generate Adaptive Parameters (F and CR)
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Generate CR ~ Normal(mean=m_cr, std=0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(loc=m_f, scale=0.1)
            f = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    val = m_f[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0:
                        if val > 1: val = 1.0
                        f[i] = val
                        break
                        
            # 4. Evolution Step
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            # Tracking successful updates for memory adaptation
            success_cr = []
            success_f = []
            success_df = []
            
            # "p-best" setting: select from top 10% (min 2 individuals)
            num_p_best = max(2, int(pop_size * 0.1))
            
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                    
                # --- Mutation: current-to-pbest/1 ---
                # V = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
                
                # Select x_pbest randomly from top p%
                p_idx = np.random.randint(0, num_p_best)
                x_pbest = pop[p_idx]
                
                # Select x_r1 from population (distinct from i)
                r1 = np.random.randint(0, pop_size)
                while r1 == i:
                    r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # Select x_r2 from Union(Population, Archive) (distinct from i, r1)
                union_size = pop_size + len(archive)
                r2 = np.random.randint(0, union_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, union_size)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                    
                mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
                
                # --- Crossover: Binomial ---
                j_rand = np.random.randint(dim)
                cross_mask = np.random.rand(dim) < cr[i]
                cross_mask[j_rand] = True # Ensure at least one param changes
                trial = np.where(cross_mask, mutant, pop[i])
                
                # --- Boundary Handling ---
                trial = np.clip(trial, min_b, max_b)
                
                # --- Selection ---
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    # Improved: store trial
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Record success for history update
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    success_df.append(fitness[i] - f_trial)
                    
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update Global Best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial.copy()
                else:
                    # No improvement: keep parent
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Move to next generation
            pop = new_pop
            fitness = new_fitness
            
            # 5. Archive Maintenance
            # If archive grows too big, remove random elements
            if len(archive) > archive_size:
                num_del = len(archive) - archive_size
                del_idxs = np.random.choice(len(archive), num_del, replace=False)
                del_idxs.sort()
                # Delete in reverse order to preserve indices
                for idx in del_idxs[::-1]:
                    del archive[idx]
            
            # 6. Memory Update (Weighted Lehmer Mean)
            # Updates M_cr and M_f based on successful parameters
            if len(success_cr) > 0:
                success_cr = np.array(success_cr)
                success_f = np.array(success_f)
                success_df = np.array(success_df)
                
                total_improvement = np.sum(success_df)
                
                if total_improvement > 0:
                    # Weights proportional to fitness improvement
                    weights = success_df / total_improvement
                    
                    # Update CR memory (Weighted Mean)
                    mean_cr = np.sum(weights * success_cr)
                    M_cr[k_mem] = mean_cr
                    
                    # Update F memory (Weighted Lehmer Mean)
                    mean_f = np.sum(weights * success_f**2) / np.sum(weights * success_f)
                    M_f[k_mem] = mean_f
                    
                    # Advance memory pointer
                    k_mem = (k_mem + 1) % H
    
    return best_val
