#The output values indicate that the Standard Differential Evolution (Algorithm 2) significantly outperformed the more complex DE with Restarts (Algorithm 1). This suggests that maintaining population diversity (via a continuous search process) was more beneficial than the aggressive "current-to-best" mutation strategy which likely led to premature convergence.
#
#To further improve upon the best result, the following algorithm implements **Self-Adaptive Differential Evolution (jDE)** combined with **Latin Hypercube Sampling (LHS)** initialization.
#
#**Key Improvements:**
#1.  **Self-Adaptive Parameters (jDE)**: Instead of fixed or randomly dithered parameters ($F$ and $CR$), this algorithm evolves these parameters along with the individuals. This allows the algorithm to automatically "learn" the best mutation and crossover rates for the specific function landscape during the search.
#2.  **Latin Hypercube Sampling (LHS)**: Instead of pure random initialization, LHS is used to generate the initial population. This ensures a more stratified and uniform coverage of the search space, providing a better starting point.
#3.  **Robust Strategy**: It uses the standard **DE/rand/1/bin** strategy which preserves diversity better than "current-to-best", preventing early stagnation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with Latin Hypercube Sampling (LHS) initialization.
    
    This algorithm adapts the mutation factor (F) and crossover rate (CR) 
    for each individual, allowing it to tune itself to the problem landscape 
    within the given time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 
    # Adapted to dimension to ensure sufficient diversity.
    # We clip it to [20, 60] to balance exploration with the number of generations
    # feasible within a potentially short max_time.
    pop_size = int(np.clip(10 * dim, 20, 60))
    
    # jDE Control Parameters (Probabilities to update F and CR)
    tau_F = 0.1
    tau_CR = 0.1
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Latin Hypercube Sampling (LHS) for better initial coverage
    # This ensures the initial population is spread more evenly than random sampling
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        # Sample uniformly within each stratum
        points = np.random.uniform(edges[:-1], edges[1:])
        # Shuffle to uncorrelate dimensions
        population[:, d] = np.random.permutation(points)
        
    # Initialize Adaptive Parameters
    # F starts at 0.5, CR at 0.9 (Standard DE defaults)
    # These will evolve during the optimization
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Strict time check inside evaluation loop
        if (datetime.now() - start_time) >= time_limit:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while True:
        # Check time at the start of the generation
        if (datetime.now() - start_time) >= time_limit:
            return best_val

        # 1. Parameter Adaptation (jDE Logic)
        # Vectorized generation of new candidate parameters
        rand1 = np.random.rand(pop_size)
        rand2 = np.random.rand(pop_size)
        rand3 = np.random.rand(pop_size)
        rand4 = np.random.rand(pop_size)

        # Update F with probability tau_F
        mask_F = rand1 < tau_F
        F_new = F.copy()
        # New F is chosen randomly in [0.1, 1.0]
        F_new[mask_F] = 0.1 + 0.9 * rand2[mask_F]
        
        # Update CR with probability tau_CR
        mask_CR = rand3 < tau_CR
        CR_new = CR.copy()
        # New CR is chosen randomly in [0.0, 1.0]
        CR_new[mask_CR] = rand4[mask_CR]
        
        # 2. Evolution Cycle (Mutation, Crossover, Selection)
        for i in range(pop_size):
            # Check time strictly before every function evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            # Mutation: DE/rand/1
            # Select 3 distinct random individuals excluding current i
            idxs = [k for k in range(pop_size) if k != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            x_r3 = population[r3]
            
            # Compute Mutant Vector
            # v = x_r1 + F_new * (x_r2 - x_r3)
            mutant = x_r1 + F_new[i] * (x_r2 - x_r3)
            
            # Boundary Constraint: Clip to bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # Crossover: Binomial
            cross_points = np.random.rand(dim) < CR_new[i]
            # Ensure at least one dimension is mutated to avoid cloning
            if not np.any(cross_points):
                cross_points[np.random.randint(dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # Selection
            f_trial = func(trial)
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                # Adopt new parameters only upon success
                # This reinforces parameters that work well for the current landscape
                F[i] = F_new[i]
                CR[i] = CR_new[i]
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    
    return best_val
