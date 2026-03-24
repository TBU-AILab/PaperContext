#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 35.26997894218242
#
# 1. algorithm code is:
#import numpy as np
#import time
#
#def run(func, dim, bounds, max_time):
#    """
#    Minimizes the given function within the specified time limit using 
#    Differential Evolution (DE).
#    """
#    start_time = time.time()
#    
#    # --- Hyperparameters ---
#    # Population size: Standard is 10*dim, but we cap it to ensure 
#    # the algorithm iterates sufficiently given the time constraint.
#    pop_size = max(10, 5 * dim) 
#    if pop_size > 60: 
#        pop_size = 60
#        
#    mutation_factor = 0.8  # F: Weighting factor for differential vector
#    crossover_prob = 0.7   # CR: Probability of crossover
#    
#    # --- Initialization ---
#    bounds_np = np.array(bounds)
#    min_b = bounds_np[:, 0]
#    max_b = bounds_np[:, 1]
#    diff_b = max_b - min_b
#    
#    # Initialize population randomly within bounds
#    # Shape: (pop_size, dim)
#    population = min_b + np.random.rand(pop_size, dim) * diff_b
#    
#    fitness = np.full(pop_size, np.inf)
#    best_val = np.inf
#    
#    # Evaluate initial population
#    # We iterate manually to check time constraints between function calls
#    for i in range(pop_size):
#        if (time.time() - start_time) >= max_time:
#            return best_val
#        
#        val = func(population[i])
#        fitness[i] = val
#        
#        if val < best_val:
#            best_val = val
#
#    # --- Main Loop (Generations) ---
#    while True:
#        # Check total time elapsed at the start of generation
#        if (time.time() - start_time) >= max_time:
#            return best_val
#
#        # 1. Mutation: V = X_a + F * (X_b - X_c)
#        # Generate random indices for a, b, c
#        # Note: While standard DE enforces distinct indices (a != b != c != i),
#        # allowing random selection with replacement is a robust approximation 
#        # for vectorized implementation in constrained scripts.
#        idx_a = np.random.randint(0, pop_size, pop_size)
#        idx_b = np.random.randint(0, pop_size, pop_size)
#        idx_c = np.random.randint(0, pop_size, pop_size)
#
#        mutant = population[idx_a] + mutation_factor * (population[idx_b] - population[idx_c])
#
#        # 2. Bounds Handling (Clip values to stay inside search space)
#        mutant = np.clip(mutant, min_b, max_b)
#
#        # 3. Crossover (Binomial)
#        cross_points = np.random.rand(pop_size, dim) < crossover_prob
#        trial_pop = np.where(cross_points, mutant, population)
#
#        # 4. Selection
#        # Evaluate trial vectors and accept if fitness improves
#        for i in range(pop_size):
#            if (time.time() - start_time) >= max_time:
#                return best_val
#
#            f_trial = func(trial_pop[i])
#
#            if f_trial <= fitness[i]:
#                fitness[i] = f_trial
#                population[i] = trial_pop[i]
#                if f_trial < best_val:
#                    best_val = f_trial
#
#    return best_val
#
#
#
#The following is an improved algorithm implementing **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**. This algorithm is generally superior to Differential Evolution for continuous, non-separable problems and handles correlation between variables effectively, preventing premature convergence in valleys (which likely caused the previous score of ~35). It includes a restart mechanism to escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # Pre-process bounds
    bounds = np.array(bounds)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    best_fitness = float('inf')
    
    # ---------------- CMA-ES Parameters ----------------
    # Strategy parameter setting: Selection
    # Population size
    lambda_ = 4 + int(3 * np.log(dim))
    mu = lambda_ // 2
    
    # Weights for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights) # Normalize
    mueff = 1 / np.sum(weights**2)      # Variance-effectiveness of sum w_i x_i
    
    # Strategy parameter setting: Adaptation
    cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
    
    # Expectation of ||N(0,I)||
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

    # ---------------- Restart Loop ----------------
    while True:
        # Check time before starting a new run
        if (time.time() - start_time) > max_time - 0.05:
            return best_fitness

        # Initialize dynamic strategy parameters
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        # Initial Point and Step Size
        xmean = np.random.uniform(lower, upper)
        sigma = 0.5 * (np.max(upper - lower)) 

        # Generation Loop
        gen = 0
        while True:
            if (time.time() - start_time) > max_time:
                return best_fitness
            
            gen += 1
            
            # Generate lambda offspring
            arz = np.random.randn(lambda_, dim)
            
            # Transform to x: y = B * D * z
            # Vectorized: y = (arz @ diag(D) @ B.T)
            y = arz @ np.diag(D) @ B.T
            arx = xmean + sigma * y
            
            # Evaluate fitness with penalty for bounds
            fitness_values = []
            
            for k in range(lambda_):
                if (time.time() - start_time) > max_time:
                    return best_fitness
                    
                # Boundary handling:
                # 1. Clip phenotype for evaluation (valid input)
                x_k = arx[k]
                x_eval = np.clip(x_k, lower, upper)
                
                # 2. Calculate squared distance penalty
                dist_sq = np.sum((x_k - x_eval)**2)
                
                # 3. Evaluate
                val = func(x_eval)
                if val < best_fitness:
                    best_fitness = val
                
                # 4. Penalize fitness for selection
                # This forces the evolution path back into bounds
                # Heuristic: val + 1e8 * dist_sq
                fitness_values.append(val + 1e8 * dist_sq)
            
            fitness_values = np.array(fitness_values)
            
            # Sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(fitness_values)
            best_idx = arindex[:mu]
            
            xold = xmean.copy()
            # Recombination: xmean = xold + sigma * sum(w_i * y_i)
            y_best = y[best_idx]
            y_w = weights @ y_best
            xmean = xold + sigma * y_w
            
            # Cumulation: Update evolution paths
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * B * z_w
            z_w = weights @ arz[best_idx]
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_w)
            
            # hsig flag
            ps_norm = np.linalg.norm(ps)
            hsig_cond = ps_norm / np.sqrt(1 - (1 - cs)**(2 * gen)) / chiN < 1.4 + 2/(dim+1)
            hsig = 1.0 if hsig_cond else 0.0
            
            # pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * y_w
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
            
            # Adapt covariance matrix C
            # Rank-1 update
            rank1 = np.outer(pc, pc)
            
            # Rank-mu update
            rank_mu = y_best.T @ np.diag(weights) @ y_best
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu + c1 * delta_hsig) * C + c1 * rank1 + cmu * rank_mu
            
            # Adapt step size sigma
            sigma = sigma * np.exp((cs / damps) * (ps_norm / chiN - 1))
            
            # Decomposition of C
            # Enforce symmetry
            C = np.triu(C) + np.triu(C, 1).T
            try:
                eig_vals, eig_vecs = np.linalg.eigh(C)
                if np.any(eig_vals <= 0):
                    break # Restart if matrix not positive definite
                D = np.sqrt(eig_vals)
                B = eig_vecs
            except np.linalg.LinAlgError:
                break # Restart
            
            # Stop criteria (Restart)
            # 1. Sigma too small
            if sigma < 1e-15: break
            # 2. Condition number too high
            if np.max(D) > 1e14 * np.min(D): break
            # 3. Divergence
            if sigma > 1e10: break
            
    return best_fitness
