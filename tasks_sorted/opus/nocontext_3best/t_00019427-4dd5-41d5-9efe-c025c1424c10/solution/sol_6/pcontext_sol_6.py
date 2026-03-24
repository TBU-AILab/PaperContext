#Looking at the results, algorithm 1 (SHADE + multi-scale Nelder-Mead) performed best at 4.255. I'll build on its strengths while adding several key improvements:
#
#1. **CMA-ES as the main optimizer** - much better than DE for continuous optimization
#2. **Better initial sampling** with opposition-based learning
#3. **Improved local search** with adaptive step sizes
#4. **Multiple restarts from diverse basins**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    mid = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Initial sampling with opposition-based learning ---
    n_init = min(max(40, 5 * dim), 300)
    init_pop = []
    init_fit = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.05:
            break
        x = lower + np.random.random(dim) * ranges
        f = evaluate(x)
        init_pop.append(x.copy())
        init_fit.append(f)
        # Opposition-based
        x_opp = lower + upper - x
        f_opp = evaluate(x_opp)
        init_pop.append(x_opp.copy())
        init_fit.append(f_opp)
    
    init_pop = np.array(init_pop)
    init_fit = np.array(init_fit)
    
    # Sort and keep best
    sorted_idx = np.argsort(init_fit)
    
    # --- Phase 2: CMA-ES ---
    def run_cmaes(x0, sigma0, time_budget_frac):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_full_cov = (n <= 50)
        
        if use_full_cov:
            C = np.eye(n)
            eigeneval = 0
            B = np.eye(n)
            D = np.ones(n)
        else:
            diagC = np.ones(n)
        
        gen = 0
        stag = 0
        prev_best_local = best
        
        while elapsed() < max_time * time_budget_frac:
            gen += 1
            
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full_cov:
                # Update eigen decomposition periodically
                if gen == 1 or eval_count[0] - eigeneval > lam / (c1 + cmu) / n / 10:
                    try:
                        C = np.triu(C) + np.triu(C, 1).T
                        D_sq, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D_sq, 1e-20))
                        eigeneval = eval_count[0]
                    except:
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
                
                for k in range(lam):
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
            else:
                sqrtC = np.sqrt(np.maximum(diagC, 1e-20))
                for k in range(lam):
                    arx[k] = mean + sigma * sqrtC * arz[k]
            
            # Evaluate
            arfitness = np.full(lam, float('inf'))
            for k in range(lam):
                if elapsed() >= max_time * time_budget_frac:
                    return
                arx[k] = clip(arx[k])
                arfitness[k] = evaluate(arx[k])
            
            # Sort by fitness
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = np.dot(weights, selected)
            
            # Cumulation: update evolution paths
            mean_diff = mean - old_mean
            
            if use_full_cov:
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * mean_diff / (sigma * np.sqrt(np.maximum(diagC, 1e-20)))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen)) / chiN < 1.4 + 2.0/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma
            
            # Adapt covariance matrix
            if use_full_cov:
                artmp = (selected - old_mean) / sigma
                C = ((1 - c1 - cmu + (1-hsig)*c1*cc*(2-cc)) * C +
                     c1 * np.outer(pc, pc) +
                     cmu * (artmp.T @ np.diag(weights) @ artmp))
            else:
                artmp = (selected - old_mean) / sigma
                diagC = ((1 - c1 - cmu + (1-hsig)*c1*cc*(2-cc)) * diagC +
                         c1 * pc**2 +
                         cmu * np.sum(weights[:, None] * artmp**2, axis=0))
            
            # Adapt step size
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            # Check stagnation
            if abs(prev_best_local - best) < 1e-14:
                stag += 1
            else:
                stag = 0
            prev_best_local = best
            
            if stag > 20 + 3*dim or sigma < 1e-16:
                break
    
    # Run CMA-ES from best initial point
    best_init = init_pop[sorted_idx[0]]
    run_cmaes(best_init, 0.3 * np.mean(ranges), 0.55)
    
    # Run CMA-ES from second best diverse point
    if len(sorted_idx) > 5:
        for idx in sorted_idx[1:10]:
            if np.linalg.norm(init_pop[idx] - best_params) > 0.1 * np.linalg.norm(ranges):
                run_cmaes(init_pop[idx], 0.2 * np.mean(ranges), 0.72)
                break
    
    # Refined CMA-ES from current best
    run_cmaes(best_params, 0.05 * np.mean(ranges), 0.88)
    run_cmaes(best_params, 0.005 * np.mean(ranges), 0.95)
    
    # Fine local search
    step = 0.001 * ranges
    while elapsed() < max_time * 0.995:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.995:
                break
            for sign in [1, -1]:
                trial = best_params.copy()
                trial[d] += sign * step[d]
                f_trial = evaluate(trial)
                if f_trial < best:
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if np.max(step / ranges) < 1e-13:
                break
    
    return best
#
#Key improvements:
#1. **CMA-ES replaces SHADE** - adapts full covariance structure, much better for correlated landscapes
#2. **Diagonal CMA for high dimensions** (>50) to avoid O(n³) eigendecomposition
#3. **Opposition-based initial sampling** doubles coverage of search space
#4. **Multiple CMA-ES restarts** at decreasing scales (0.3→0.05→0.005 of range)
#5. **Diverse restart selection** - picks second starting point far from best to explore different basins
#6. **Coordinate descent refinement** at the end for fine-tuning
