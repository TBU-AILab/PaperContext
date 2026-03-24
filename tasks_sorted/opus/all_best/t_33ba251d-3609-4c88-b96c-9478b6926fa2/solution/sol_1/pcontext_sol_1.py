#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Implement proper CMA-ES** instead of just DE - CMA-ES is much better for continuous optimization
#2. **Better adaptive restart strategy** with multiple restarts at different scales
#3. **More efficient local search** with proper step-size adaptation
#4. **Vectorized operations** where possible to save time
#
#Here's my improved algorithm:
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    mid = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # CMA-ES implementation
    def run_cmaes(x0, sigma0, budget_frac):
        nonlocal best, best_x
        time_limit = elapsed() + max_time * budget_frac
        
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
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dimensions
        use_diag = n > 40
        
        if use_diag:
            C_diag = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        
        counteval = 0
        
        while elapsed() < time_limit:
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_diag:
                sqrtC = np.sqrt(C_diag)
                for k in range(lam):
                    arx[k] = clip(xmean + sigma * sqrtC * arz[k])
            else:
                for k in range(lam):
                    arx[k] = clip(xmean + sigma * (B * D) @ arz[k])
            
            # Evaluate
            arfitness = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= time_limit:
                    return
                arfitness[k] = eval_f(arx[k])
                counteval += 1
            
            # Sort
            arindex = np.argsort(arfitness)
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            # Update evolution paths
            if use_diag:
                invsqrtC_diag = 1.0 / np.sqrt(C_diag)
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC_diag * (xmean - xold) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (xmean - xold) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean - xold) / sigma
            
            # Update covariance
            artmp = (arx[arindex[:mu]] - xold) / sigma
            if use_diag:
                C_diag = (1 - c1 - cmu) * C_diag + c1 * (pc**2 + (1-hsig)*cc*(2-cc)*C_diag) + cmu * np.sum(weights[:, None] * artmp**2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
                
                if counteval - eigeneval > lam/(c1+cmu)/n/10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
            
            # Update sigma
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, max(ranges))
            
            # Check convergence
            if sigma < 1e-12:
                return
    
    # Multi-restart CMA-ES strategy
    # Initial broad search
    n_restarts = 0
    while elapsed() < max_time * 0.95:
        remaining = max_time - elapsed()
        if remaining < 0.5:
            break
        
        if n_restarts == 0:
            x0 = mid.copy()
            sigma0 = 0.3 * np.mean(ranges)
            budget = 0.3
        elif best_x is not None and np.random.rand() < 0.5:
            # Local restart around best
            x0 = best_x + 0.1 * ranges * np.random.randn(dim)
            x0 = clip(x0)
            sigma0 = 0.05 * np.mean(ranges) * (0.5 ** (n_restarts // 3))
            budget = min(0.2, remaining / max_time - 0.05)
        else:
            # Global restart
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.2 * np.mean(ranges)
            budget = min(0.15, remaining / max_time - 0.05)
        
        if budget < 0.02:
            budget = 0.02
        
        run_cmaes(x0, sigma0, budget)
        n_restarts += 1
    
    return best
#
#Key improvements:
#1. **Full CMA-ES implementation** with proper covariance matrix adaptation - much more effective than DE for many function landscapes
#2. **Diagonal covariance mode** for high dimensions (>40) to avoid expensive matrix operations
#3. **Intelligent multi-restart strategy** alternating between local restarts around the best solution and global restarts for exploration
#4. **Adaptive sigma** with proper step-size control via cumulative path length
#5. **Time-budget aware restarts** that allocate remaining time intelligently across restarts
