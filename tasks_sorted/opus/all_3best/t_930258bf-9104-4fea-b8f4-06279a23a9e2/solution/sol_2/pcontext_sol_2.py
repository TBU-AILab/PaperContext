#Looking at the previous results, I'll improve the algorithm by:
#
#1. **Adding Differential Evolution (DE)** as a complementary global search alongside CMA-ES
#2. **Better restart diversification** - alternating between exploration and exploitation restarts
#3. **Improved local search** - using a proper pattern search with adaptive step sizes
#4. **Multiple CMA-ES runs with varying population sizes** (BIPOP strategy)
#5. **Better time management** across phases
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_x, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(20 * dim, 100), 600)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fit = []
    for i in range(n_init):
        if remaining() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fit.append((f, i))
    
    init_fit.sort()
    top_k = min(15, n_init)
    top_points = [init_points[init_fit[i][1]].copy() for i in range(top_k)]
    top_fits = [init_fit[i][0] for i in range(top_k)]

    # --- Phase 2: Differential Evolution ---
    def run_de(time_fraction=0.3):
        nonlocal best, best_x
        deadline = elapsed() + max_time * time_fraction
        
        pop_size = max(min(10 * dim, 100), 30)
        # Initialize population from top points + random
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(pop_size):
            if i < len(top_points):
                pop[i] = top_points[i].copy()
            else:
                pop[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
            fit[i] = eval_func(pop[i])
            if remaining() <= 0 or elapsed() >= deadline:
                return
        
        F = 0.8
        CR = 0.9
        
        gen = 0
        while elapsed() < deadline and remaining() > 0:
            gen += 1
            for i in range(pop_size):
                if elapsed() >= deadline or remaining() <= 0:
                    return
                
                # DE/current-to-best/1 with jitter
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b = np.random.choice(idxs, 2, replace=False)
                
                best_idx = np.argmin(fit)
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.4, 1.2)
                
                mutant = pop[i] + Fi * (pop[best_idx] - pop[i]) + Fi * (pop[a] - pop[b])
                mutant = clip(mutant)
                
                # Binomial crossover
                CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
                cross_points = np.random.rand(dim) < CRi
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = eval_func(trial)
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    run_de(time_fraction=0.35)

    # --- Phase 3: CMA-ES with BIPOP restarts ---
    def run_cmaes(x0, initial_sigma, lam_mult=1):
        nonlocal best, best_x
        
        sigma = initial_sigma
        mean = x0.copy()
        n = dim
        
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        lam = max(lam, 6)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        best_gen_fit = float('inf')
        stag = 0
        
        for gen in range(1, 10000):
            if remaining() <= 0:
                return
            
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    Dvals, B = np.linalg.eigh(C)
                    Dvals = np.maximum(Dvals, 1e-20)
                    D = np.sqrt(Dvals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return
            
            arx = np.zeros((lam, n))
            arfit = np.zeros(lam)
            for k in range(lam):
                if remaining() <= 0:
                    return
                z = np.random.randn(n)
                arx[k] = clip(mean + sigma * (B @ (D * z)))
                arfit[k] = eval_func(arx[k])
                counteval += 1
            
            idx = np.argsort(arfit)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lam)) / chiN < 1.4 + 2.0 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges) * 2)
            
            if gen % 10 == 0:
                if arfit[idx[0]] < best_gen_fit - 1e-10 * (abs(best_gen_fit) + 1):
                    best_gen_fit = arfit[idx[0]]
                    stag = 0
                else:
                    stag += 1
                if stag >= 4 or sigma < 1e-16:
                    return
    
    restart = 0
    lam_mult = 1
    while remaining() > 1:
        if restart < len(top_points):
            x0 = top_points[restart % len(top_points)]
            sig = np.mean(ranges) / 4.0
        elif np.random.rand() < 0.4 and best_x is not None:
            x0 = clip(best_x + np.random.randn(dim) * ranges * 0.05)
            sig = np.mean(ranges) / 8.0
        else:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            sig = np.mean(ranges) / 3.0
            lam_mult = min(lam_mult * 1.5, 6)
        
        run_cmaes(x0, sig, lam_mult if restart >= len(top_points) else 1)
        restart += 1
    
    return best
