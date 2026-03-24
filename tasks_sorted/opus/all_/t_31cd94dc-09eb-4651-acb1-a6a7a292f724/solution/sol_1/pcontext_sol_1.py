#
#
#Looking at the previous algorithm, I'll improve it with several enhancements:
#
#1. **CMA-ES inspired search** - better covariance adaptation
#2. **Multiple restarts with different strategies**
#3. **Better local search (pattern search + Nelder-Mead simplex)**
#4. **Adaptive population sizing**
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
    
    def remaining():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Initial sampling with LHS ---
    n_init = min(max(10 * dim, 50), 300)
    pop = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    pop = lower + pop * ranges
    
    fit = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.90:
            return best
        fit[i] = eval_f(pop[i])
    
    # Sort and keep best
    order = np.argsort(fit)
    pop = pop[order]
    fit = fit[order]

    # --- Phase 2: CMA-ES ---
    def run_cmaes(x0, sigma0, budget_frac):
        nonlocal best, best_params
        deadline = elapsed() + max_time * budget_frac
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        eigeneval = 0
        counteval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        while elapsed() < deadline:
            # Generate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.array([eval_f(arx[k]) for k in range(lam)])
            if elapsed() >= deadline:
                break
            counteval += lam
            
            idx = np.argsort(arfitness)
            
            old_mean = mean.copy()
            mean = np.dot(weights, arx[idx[:mu]])
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(B * D, (mean - old_mean) / sigma) if sigma > 1e-20 else ps
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, max(ranges))
            
            if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                D_sq, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_sq, 1e-20))
    
    # Run CMA-ES from best points
    n_restarts = max(1, min(5, int(dim / 3) + 1))
    budget_per = 0.60 / n_restarts
    
    for r in range(n_restarts):
        if elapsed() >= max_time * 0.80:
            break
        x0 = pop[r % len(pop)].copy() if r < len(pop) else best_params.copy()
        sig = 0.3 * np.mean(ranges) if r == 0 else 0.1 * np.mean(ranges) * (0.5 ** r)
        run_cmaes(x0, sig, budget_per)

    # --- Phase 3: Nelder-Mead simplex ---
    if best_params is not None and elapsed() < max_time * 0.95:
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step = 0.05 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([eval_f(simplex[i]) for i in range(n + 1)])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while elapsed() < max_time * 0.98:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = simplex[:n].mean(axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_f(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[n], f_simplex[n] = xe, fe
                else:
                    simplex[n], f_simplex[n] = xr, fr
            elif fr < f_simplex[n - 1]:
                simplex[n], f_simplex[n] = xr, fr
            else:
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[n] - centroid))
                fc = eval_f(xc)
                if fc < min(fr, f_simplex[n]):
                    simplex[n], f_simplex[n] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_f(simplex[i])
                        if elapsed() >= max_time * 0.98:
                            break
    
    return best
