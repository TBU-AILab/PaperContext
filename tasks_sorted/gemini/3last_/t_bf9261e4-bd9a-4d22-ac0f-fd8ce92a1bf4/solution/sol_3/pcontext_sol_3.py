#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 49.07308299118522 (L-SHADE)
#2. output value is: 30.853154225886627 (Restarting DE)
#3. output value is: 94.19612142364304 (Basic DE)
#
#The best result (30.85) was achieved by a Restarting Differential Evolution strategy. However, Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is widely regarded as superior for continuous optimization, especially in non-separable or ill-conditioned landscapes ("valleys"), provided it is implemented with a restart mechanism to handle multimodality.
#
#The following algorithm implements **IPO-CMA-ES (Increasing Population Size Covariance Matrix Adaptation Evolution Strategy)**. It improves upon previous attempts by:
#1.  **Adapting the Covariance Matrix**: It learns the correlation between variables, allowing it to navigate narrow valleys where standard DE moves slowly.
#2.  **Restarting with Increasing Population (IPO)**: If the algorithm converges to a local optimum or stagnates, it restarts with a doubled population size. This significantly increases the probability of finding the global minimum.
#3.  **Lazy Matrix Updates**: To ensure high performance, the computationally expensive eigendecomposition is only performed periodically, not every generation.
#4.  **Robust Boundary Handling**: It evaluates clipped (valid) parameters but applies a soft penalty to the internal distribution state, guiding the search naturally into the feasible region.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPO-CMA-ES (Increasing Population Size 
    Covariance Matrix Adaptation Evolution Strategy).
    """
    start_time = datetime.now()
    # Reserve 5% of time for safe return overhead
    time_limit = timedelta(seconds=max_time * 0.95)
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    mean_diff = np.mean(diff_b)
    
    best_fitness = float('inf')
    
    # Helper to check remaining time
    def has_time():
        return (datetime.now() - start_time) < time_limit

    # Initial Population size (lambda)
    # Standard heuristic: 4 + 3 * log(dim)
    lam_start = 4 + int(3 * np.log(dim))
    
    # Restart Loop
    restart_idx = 0
    while has_time():
        # IPO: Double population size each restart to improve global search capability
        lam = int(lam_start * (2 ** restart_idx))
        restart_idx += 1
        
        # Initialize Mean (randomly within bounds)
        xmean = min_b + np.random.rand(dim) * diff_b
        
        # Initialize Step-size (sigma)
        # Start with a relatively large step size (50% of domain average width)
        sigma = 0.5 * mean_diff
        
        # Initialize Covariance Matrix components
        # C = B * D^2 * B^T. Start with Identity.
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        # Evolution Paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Weights for selection (approximate recombination)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Adaptation constants (standard CMA-ES tuning)
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        counteval = 0
        
        # Generation Loop
        while has_time():
            # 1. Generate Offspring
            # z ~ N(0, I)
            arz = np.random.randn(lam, dim)
            
            # y = B * D * z  => Transform to covariance shape
            # Vectorized: ary = arz @ (diag(D) @ B.T)
            ary = arz @ (np.diag(D) @ B.T)
            
            # x = m + sigma * y
            arx = xmean + sigma * ary
            
            fitnesses = []
            
            # 2. Evaluate
            for i in range(lam):
                if not has_time():
                    return best_fitness
                
                # Boundary Handling:
                # 1. Clip parameters for actual function evaluation
                x_gen = arx[i]
                x_eval = np.clip(x_gen, min_b, max_b)
                
                val = func(x_eval)
                
                # Update global best
                if val < best_fitness:
                    best_fitness = val
                
                # 2. Penalize internal CMA-ES fitness
                # This guides the distribution mean back towards the feasible region
                # Penalty is proportional to squared distance from bounds
                dist_sq = np.sum((x_gen - x_eval)**2)
                penalty = 1e8 * dist_sq
                fitnesses.append(val + penalty)
                
                counteval += 1
            
            fitnesses = np.array(fitnesses)
            
            # 3. Sort and Select
            sorted_idx = np.argsort(fitnesses)
            best_idx = sorted_idx[:mu]
            
            # 4. Update Mean
            xold = xmean.copy()
            # zmean = weighted average of selected z-vectors
            zmean = weights @ arz[best_idx]
            # xmean = xold + sigma * (B * D * zmean)
            xmean = xold + sigma * (zmean @ (np.diag(D) @ B.T))
            
            # 5. Update Evolution Paths
            # ps (Evolution Path for Sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (counteval/lam + 1))) / chiN < 1.4 + 2/(dim+1)
            
            # pc (Evolution Path for Covariance Matrix)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
            
            # 6. Update Covariance Matrix C
            # Rank-1 update (using pc)
            pc_part = np.outer(pc, pc)
            
            # Rank-mu update (using selected y vectors)
            # y_selected = (arx_selected - xold) / sigma
            y_selected = ary[best_idx]
            rank_mu_part = y_selected.T @ np.diag(weights) @ y_selected
            
            C = (1 - c1 - cmu) * C \
                + c1 * (pc_part + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * rank_mu_part
                
            # 7. Update Step-size sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 8. Eigen Decomposition (Lazy)
            # Only update B and D every few generations to save time (O(N^3))
            # Heuristic gap: roughly 1/(c1+cmu)/N/10
            update_gap = max(1, int(1 / (c1 + cmu) / dim / 10))
            
            if (counteval // lam) % update_gap == 0:
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    vals, vecs = np.linalg.eigh(C)
                    # Numerical stability: ensure positive eigenvalues
                    vals = np.maximum(vals, 1e-30)
                    D = np.sqrt(vals)
                    B = vecs
                except np.linalg.LinAlgError:
                    # If matrix degenerates, force a restart
                    break 
            
            # 9. Stop/Restart Criteria
            # Condition 1: Condition number too high (ill-conditioned)
            if np.max(D) > 1e14 * np.min(D):
                break
            # Condition 2: Step size too small (converged)
            if sigma < 1e-16:
                break
            # Condition 3: Step size exploded (unstable)
            if sigma > 1e10:
                break
            # Condition 4: Fitness stagnation (flat fitness in population)
            if np.std(fitnesses[best_idx]) < 1e-12:
                break
            
    return best_fitness
