#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Use CMA-ES-like strategy** - more effective for continuous optimization than basic DE
#2. **Better time management** - spend more time on exploitation
#3. **Multiple restarts with shrinking search regions**
#4. **Vectorized operations** for speed
#
#Here's my improved algorithm:
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_point(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Sobol-like initialization with LHS ---
    n_init = min(max(50, 15 * dim), 500)
    for i in range(n_init):
        if elapsed() >= max_time * 0.15:
            break
        x = lower + np.random.random(dim) * ranges
        eval_point(x)

    # --- Phase 2: CMA-ES-inspired search with multiple restarts ---
    restart = 0
    while elapsed() < max_time * 0.92:
        restart += 1
        
        # Initialize CMA-ES parameters
        if best_params is not None and restart <= 2:
            mean = best_params.copy()
            sigma = 0.2 * np.max(ranges) / (restart)
        else:
            mean = lower + np.random.random(dim) * ranges
            sigma = 0.3 * np.max(ranges)
        
        lam = max(8, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        eigeneval = 0
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        gen = 0
        stag_count = 0
        local_best = float('inf')
        
        while elapsed() < max_time * 0.92:
            gen += 1
            
            # Eigen decomposition periodically
            if eigeneval <= 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                except:
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)
                eigeneval = max(1, int(lam / (10 * dim * (c1 + cmu))))
            eigeneval -= 1
            
            # Sample offspring
            arx = np.zeros((lam, dim))
            arz = np.zeros((lam, dim))
            arf = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= max_time * 0.92:
                    return best
                arz[k] = np.random.randn(dim)
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arf[k] = eval_point(arx[k])
            
            idx = np.argsort(arf)
            xold = mean.copy()
            mean = clip(np.sum(weights[:, None] * arx[idx[:mu]], axis=0))
            
            zmean = np.sum(weights[:, None] * arz[idx[:mu]], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ zmean)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*gen)) / chiN < 1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - xold) / sigma
            
            artmp = (arx[idx[:mu]] - xold) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            if arf[idx[0]] < local_best - 1e-10:
                local_best = arf[idx[0]]
                stag_count = 0
            else:
                stag_count += 1
            
            if stag_count > 10 + 30 * dim // lam or sigma < 1e-12:
                break

    # --- Phase 3: Final Nelder-Mead polish ---
    if best_params is not None and elapsed() < max_time * 0.98:
        n = dim
        scale = 0.01 * ranges
        simplex = np.zeros((n+1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i] * (1 if best_params[i] < (lower[i]+upper[i])/2 else -1)
        
        f_simplex = np.array([eval_point(simplex[i]) for i in range(n+1) if elapsed() < max_time*0.98])
        if len(f_simplex) < n+1:
            return best
        
        while elapsed() < max_time * 0.99:
            order = np.argsort(f_simplex)
            simplex = simplex[order]; f_simplex = f_simplex[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + 1.0*(centroid - simplex[-1]))
            fr = eval_point(xr)
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1]=xr; f_simplex[-1]=fr
            elif fr < f_simplex[0]:
                xe = clip(centroid + 2.0*(xr - centroid))
                fe = eval_point(xe)
                simplex[-1], f_simplex[-1] = (xe,fe) if fe<fr else (xr,fr)
            else:
                xc = clip(centroid + 0.5*(simplex[-1] - centroid))
                fc = eval_point(xc)
                if fc < f_simplex[-1]:
                    simplex[-1]=xc; f_simplex[-1]=fc
                else:
                    for i in range(1, n+1):
                        if elapsed() >= max_time*0.99: return best
                        simplex[i] = clip(simplex[0]+0.5*(simplex[i]-simplex[0]))
                        f_simplex[i] = eval_point(simplex[i])
    
    return best
