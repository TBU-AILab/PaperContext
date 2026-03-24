#Looking at the previous algorithm, I can see it uses a DE hybrid approach with coordinate descent refinement. Let me improve it with a more sophisticated approach combining CMA-ES (which is excellent for continuous optimization), better restart strategies, and improved local search.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- CMA-ES Implementation ---
    def cma_es(x0, sigma0, budget_fraction=0.4):
        nonlocal best, best_x
        
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
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        deadline = elapsed() + max_time * budget_fraction
        
        generation = 0
        while elapsed() < deadline and elapsed() < max_time * 0.95:
            generation += 1
            
            # Generate offspring
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= deadline or elapsed() >= max_time * 0.95:
                    return
                z = np.random.randn(n)
                arx[k] = mean + sigma * (B @ (D * z))
                arx[k] = clip(arx[k])
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[arindex[:mu]]
            mean = selected.T @ weights
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * cc*(2-cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            # Update eigen decomposition
            if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1/D) @ B.T
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                break
    
    # --- Multi-restart CMA-ES ---
    # Initial run from center
    x0 = center.copy()
    sigma0 = 0.3 * np.max(ranges)
    cma_es(x0, sigma0, budget_fraction=0.35)
    
    # Restart from best
    if best_x is not None and time_left() > max_time * 0.1:
        cma_es(best_x.copy(), 0.1 * np.max(ranges), budget_fraction=0.25)
    
    # Random restarts
    restart = 0
    while time_left() > max_time * 0.1:
        restart += 1
        if restart % 2 == 0 and best_x is not None:
            x0 = best_x + 0.3 * ranges * np.random.randn(dim)
            x0 = clip(x0)
            sig = 0.05 * np.max(ranges)
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            sig = 0.2 * np.max(ranges)
        cma_es(x0, sig, budget_fraction=0.15)
    
    # --- Local refinement: Nelder-Mead style ---
    if best_x is not None and time_left() > 0.5:
        step = 0.001 * ranges
        for _ in range(5):
            if elapsed() >= max_time * 0.99:
                break
            improved = False
            for j in range(dim):
                if elapsed() >= max_time * 0.99:
                    break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[j] += direction * step[j]
                    f_trial = eval_func(trial)
                    if f_trial < best:
                        improved = True
                        step[j] *= 1.5
                        break
                else:
                    step[j] *= 0.5
            if not improved:
                step *= 0.5
    
    return best
#
#Key improvements:
#1. **Full CMA-ES implementation** - covariance matrix adaptation is state-of-the-art for black-box optimization
#2. **Multi-restart strategy** - alternates between restarts from best solution (exploitation) and random restarts (exploration)
#3. **Adaptive budget allocation** - distributes time across multiple CMA-ES runs with decreasing budgets
#4. **Robust eigendecomposition** with fallback to identity on numerical issues
#5. **Better local search** with adaptive step sizes that grow on success and shrink on failure
