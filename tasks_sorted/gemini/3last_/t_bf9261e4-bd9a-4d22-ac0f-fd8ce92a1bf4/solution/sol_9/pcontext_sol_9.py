4. output value is: 166.89102866002295

 4. algorithm code is:
#The best performing algorithm so far was **IPO-CMA-ES** (Output 166.89). To improve upon this, I propose **Mirrored-IPO-CMA-ES**.
#
##### Improvements
#1.  **Mirrored Sampling**: Instead of sampling $\lambda$ independent random vectors, we sample $\lambda/2$ vectors $z_i$ and use both $z_i$ and $-z_i$ to generate offspring. This reduces the variance of the mutation distribution and usually accelerates convergence by ensuring a symmetric sampling around the mean.
#2.  **Optimized Vectorization**: The implementation uses efficient numpy vectorization for the `rank-mu` update and population generation to minimize interpreter overhead.
#3.  **Strict IPO Regimes**: We adhere to the Increasing Population Size (IPO) strategy, doubling the population size ($ \lambda \cdot 2^k $) on every restart. This transitions from fast local search to broader global exploration.
#4.  **Robust Time Management**: The algorithm checks the time budget periodically (batched) to maximize the evaluation rate while guaranteeing a safe exit before the timeout.
#5.  **Lazy Eigendecomposition**: The covariance matrix decomposition ($O(N^3)$) is only performed when necessary, based on the adaptation rate and generation counter.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Mirrored-IPO-CMA-ES (Increasing Population Size 
    CMA-ES with Mirrored Sampling).
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Reserve a small buffer (0.05s) to ensure strict adherence to max_time
    time_limit = timedelta(seconds=max_time - 0.05)

    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Problem Setup ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    scale = ub - lb
    # Avoid zero scale issues
    scale = np.maximum(scale, 1e-9)
    min_scale = np.min(scale)
    max_scale = np.max(scale)
    
    global_best = float('inf')
    
    # --- IPO-CMA-ES Configuration ---
    # Start with a small population and double it on every restart.
    # Base lambda: 4 + 3*ln(D)
    lam_base = 4 + int(3 * np.log(dim))
    restart_idx = 0

    while has_time():
        # 1. Population Size for this restart
        lam = int(lam_base * (2 ** restart_idx))
        
        # Enforce even population size for Mirrored Sampling
        if lam % 2 != 0:
            lam += 1
            
        # 2. Strategy Parameters
        mu = lam // 2
        # Weights: logarithmic, decreasing, sum to 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Adaptation constants (Hansen's formulas)
        cc = (4 + mueff/dim) / (dim + 4 + 2 * mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs
        chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # 3. Initialization
        # Random start within bounds
        xmean = lb + np.random.rand(dim) * scale
        # Initial step size: half of average domain width
        sigma = 0.5 * np.mean(scale)
        
        # Covariance Matrix Decomposition
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        counteval = 0
        eigeneval = 0
        fitness_history = []
        
        # --- Evolution Loop ---
        while has_time():
            # 4. Mirrored Sampling
            # Generate lambda/2 independent vectors, and their negations
            half_lam = lam // 2
            arz_half = np.random.randn(half_lam, dim)
            arz = np.vstack([arz_half, -arz_half])
            
            # 5. Transform to Phenotype
            # y = B * D * z  (Vectorized)
            # arx = mean + sigma * y
            # D scales the columns of arz (elementwise)
            ary = (arz * D) @ B.T
            arx = xmean + sigma * ary
            
            # 6. Evaluation with Penalty
            fitness = np.zeros(lam)
            arx_clipped = np.clip(arx, lb, ub)
            
            # Process evaluations
            # Check time periodically to reduce overhead
            batch_check = max(1, 1000 // dim)
            
            abort_restart = False
            for k in range(lam):
                if k % batch_check == 0 and not has_time():
                    abort_restart = True
                    break
                
                # Evaluate feasible solution
                val = func(arx_clipped[k])
                
                # Update Global Best
                if val < global_best:
                    global_best = val
                
                # Add soft quadratic penalty for CMA guidance
                # Penalty proportional to squared distance from bounds
                dist_sq = np.sum((arx[k] - arx_clipped[k])**2)
                fitness[k] = val + 1e8 * dist_sq
            
            if abort_restart or not has_time():
                return global_best
            
            counteval += lam
            
            # 7. Selection
            sorted_idx = np.argsort(fitness)
            best_idx = sorted_idx[:mu]
            
            # 8. Update Mean
            xold = xmean.copy()
            y_sel = ary[best_idx]
            z_sel = arz[best_idx]
            
            # Weighted recombination
            ymean = weights @ y_sel
            zmean = weights @ z_sel
            xmean = xold + sigma * ymean
            
            # 9. Update Evolution Paths
            # ps: Path for sigma (conjugate evolution path)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            
            # hsig: Stall flag to prevent C explosion
            # Correction factor accounts for initial generations
            hsig_check = np.linalg.norm(ps) / chiN / np.sqrt(1 - (1 - cs)**(2 * (counteval/lam) + 1))
            hsig = hsig_check < 1.4 + 2/(dim+1)
            
            # pc: Path for covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ymean
            
            # 10. Update Covariance Matrix
            dh = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            # Rank-mu update (using weighted selection)
            rank_mu = y_sel.T @ (y_sel * weights[:, None])
            
            C = (1 - c1 - cmu) * C + c1 * (rank_one + dh * C) + cmu * rank_mu
            
            # 11. Update Step Size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            # 12. Lazy Eigendecomposition
            # Decompose C only every ~ O(dim) generations
            gen = counteval // lam
            if gen - eigeneval > max(1, 1 / ((c1 + cmu) * dim) / 10):
                eigeneval = gen
                C = np.triu(C) + np.triu(C, 1).T # Enforce symmetry
                try:
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.maximum(vals, 1e-14) # Numerical floor
                    D = np.sqrt(vals)
                    B = vecs
                except np.linalg.LinAlgError:
                    break # Restart on matrix error
            
            # 13. Check Restart Criteria
            # Condition number
            if np.max(D) > 1e14 * np.min(D): break
            # Sigma too small (converged)
            if sigma < 1e-12 * min_scale: break
            # Sigma too large (diverged)
            if sigma > 1e10 * max_scale: break
            # Stagnation
            fitness_history.append(fitness[sorted_idx[0]])
            hist_len = 10 + int(30 * dim / lam)
            if len(fitness_history) > hist_len:
                fitness_history.pop(0)
                if max(fitness_history) - min(fitness_history) < 1e-12:
                    break
        
        # Prepare for next restart (Double population)
        restart_idx += 1
        
    return global_best