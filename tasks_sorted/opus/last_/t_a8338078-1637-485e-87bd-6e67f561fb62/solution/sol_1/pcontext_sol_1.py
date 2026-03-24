#Looking at the previous algorithm, I can see several areas for improvement:
#
#1. The CMA-ES budget allocation is suboptimal - restarts don't get enough time
#2. The initialization could be better
#3. The CMA-ES implementation has some issues with restart logic and sigma management
#4. Need better multi-restart strategy with increasing population (IPOP-CMA-ES style)
#
#Here's my improved version:
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(30 * dim, 200), 1000)
    
    # Time-check: if dim is very large, reduce init
    if dim > 50:
        n_init = min(n_init, 300)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i] + lower[i]
    
    population = []
    for i in range(n_init):
        if not time_ok(0.15):
            break
        f = eval_f(init_points[i])
        population.append((f, init_points[i].copy()))
    
    population.sort(key=lambda t: t[0])
    
    # --- Phase 2: IPOP-CMA-ES with restarts ---
    def cmaes_run(x0, sigma0, lam_mult=1):
        nonlocal best, best_x
        
        n = dim
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        lam = max(lam, 6)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 100)
        
        if use_full:
            C = np.eye(n)
            eigeneval = 0
            update_eigen_interval = max(1, lam // (10 * n))
        else:
            diagC = np.ones(n)
        
        gen = 0
        stag_count = 0
        best_local = float('inf')
        
        while time_ok(0.92):
            # Eigendecomposition
            if use_full:
                if gen == 0 or eigeneval >= update_eigen_interval:
                    try:
                        C = np.triu(C) + np.triu(C, 1).T
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-20)
                        D = np.sqrt(eigvals)
                        invsqrtC = eigvecs @ np.diag(1.0 / D) @ eigvecs.T
                        eigeneval = 0
                    except:
                        return
                eigeneval += 1
            
            # Generate and evaluate offspring
            arxs = []
            arfitness = []
            for k in range(lam):
                if not time_ok(0.92):
                    return
                z = np.random.randn(n)
                if use_full:
                    y = eigvecs @ (D * z)
                    x = xmean + sigma * y
                else:
                    x = xmean + sigma * (np.sqrt(diagC) * z)
                
                f = eval_f(x)
                arxs.append(np.clip(x, lower, upper))
                arfitness.append(f)
            
            idx = np.argsort(arfitness)
            
            local_best = arfitness[idx[0]]
            if local_best < best_local - 1e-10:
                best_local = local_best
                stag_count = 0
            else:
                stag_count += 1
            
            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arxs[idx[i]]
            
            diff = xmean - xold
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / (np.sqrt(diagC) * sigma + 1e-30)
            
            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            if use_full:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (arxs[idx[i]] - xold) / sigma
                
                rank_mu_update = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_update += weights[i] * np.outer(artmp[i], artmp[i])
                
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * rank_mu_update
            else:
                artmp_sq = np.zeros(n)
                for i in range(mu):
                    d = (arxs[idx[i]] - xold) / sigma
                    artmp_sq += weights[i] * d**2
                
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * artmp_sq
                diagC = np.maximum(diagC, 1e-20)
            
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 10 * np.max(ranges))
            
            gen += 1
            
            # Termination conditions
            if sigma < 1e-14:
                return
            if stag_count > 20 + 3 * n:
                return
            if gen > 100 + 50 * n:
                return
    
    # Multiple restarts with increasing population
    lam_mult = 1.0
    restart = 0
    
    while time_ok(0.92):
        if restart == 0 and best_x is not None:
            x0 = best_x.copy()
            sig0 = 0.2 * np.mean(ranges)
        elif restart < len(population):
            x0 = population[min(restart, len(population)-1)][1].copy()
            sig0 = 0.3 * np.mean(ranges)
        else:
            x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sig0 = 0.4 * np.mean(ranges)
        
        cmaes_run(x0, sig0, lam_mult)
        restart += 1
        lam_mult *= 2  # IPOP: double population size
        if lam_mult > 32:
            lam_mult = 1.0  # Reset
    
    # --- Phase 3: Local refinement ---
    if best_x is not None and time_ok(0.95):
        for scale in [1e-3, 1e-4, 1e-5]:
            step = scale * ranges
            improved = True
            while improved and time_ok(0.97):
                improved = False
                for i in range(dim):
                    if not time_ok(0.97):
                        return best
                    for direction in [1.0, -1.0]:
                        x_trial = best_x.copy()
                        x_trial[i] = np.clip(x_trial[i] + direction * step[i], lower[i], upper[i])
                        f = eval_f(x_trial)
                        if f < best:
                            improved = True
    
    return best
#
#Key improvements:
#1. **IPOP-CMA-ES**: Doubling population size on restarts for better global exploration
#2. **Better time management**: More budget to CMA-ES, less to init
#3. **Eigendecomposition caching**: Reduced frequency of expensive eigen updates
#4. **Stagnation detection**: Restarts when progress stalls
#5. **Smarter restart strategy**: First from best, then from diverse good points, then random
#6. **Cleaner code structure** with helper functions for time checking
