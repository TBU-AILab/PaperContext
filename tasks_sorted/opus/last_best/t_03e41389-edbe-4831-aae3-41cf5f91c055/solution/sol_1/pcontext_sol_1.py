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
    
    def time_left():
        return max_time - elapsed()
    
    def evaluate(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Initial sampling with LHS ---
    pop_size = min(max(30, 15 * dim), 300)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])

    # --- Phase 2: CMA-ES inspired search ---
    # Simple (mu, lambda) evolution strategy with covariance adaptation
    
    def run_cmaes_simple(x0, sigma0, budget_fraction):
        nonlocal best, best_x
        deadline = elapsed() + time_left() * budget_fraction
        
        n = dim
        mu = max(4, n // 2)
        lam = max(8, 2 * mu)
        
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen = 0
        while elapsed() < deadline and elapsed() < max_time * 0.92:
            gen += 1
            try:
                sqrtC = np.linalg.cholesky(C)
            except:
                C = np.eye(n)
                sqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz @ sqrtC.T)
            arx = np.clip(arx, lower, upper)
            
            fits = np.array([evaluate(arx[k]) for k in range(lam)])
            if elapsed() >= deadline or elapsed() >= max_time * 0.92:
                break
            
            idx = np.argsort(fits)
            selected = arz[idx[:mu]]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            zmean = np.sum(weights[:, None] * selected, axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN) < (1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (sqrtC @ zmean)
            
            C = (1 - c1 - cmu_val) * C + c1 * np.outer(pc, pc)
            for k in range(mu):
                yk = sqrtC @ selected[k]
                C += cmu_val * weights[k] * np.outer(yk, yk)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            if sigma < 1e-15:
                break
    
    # Run CMA-ES from best found point
    sorted_idx = np.argsort(fitness)
    
    # Multiple restarts with different starting points
    n_restarts = max(1, min(5, int(dim <= 20) * 3 + 2))
    budget_per = 0.85 / n_restarts
    
    for r in range(n_restarts):
        if elapsed() >= max_time * 0.92:
            break
        if r == 0:
            x0 = best_x.copy()
            sig = 0.3 * np.mean(ranges)
        else:
            x0 = best_x + 0.1 * ranges * (np.random.random(dim) - 0.5)
            x0 = np.clip(x0, lower, upper)
            sig = 0.1 * np.mean(ranges)
        run_cmaes_simple(x0, sig, budget_per)
    
    # --- Phase 3: Final Nelder-Mead style local search ---
    if best_x is not None:
        scale = 0.001 * ranges
        for iteration in range(20):
            if elapsed() >= max_time * 0.98:
                break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[d] += direction * scale[d]
                    f_trial = evaluate(trial)
                    if f_trial < best:
                        improved = True
            if not improved:
                scale *= 0.5
                if np.max(scale) < 1e-15:
                    break
    
    return best