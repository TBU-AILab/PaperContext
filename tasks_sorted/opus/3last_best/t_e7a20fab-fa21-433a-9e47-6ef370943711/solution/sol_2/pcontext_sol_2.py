#Looking at the previous best algorithm (score 146.4), I can identify several improvements:
#
#1. **Eigendecomposition too infrequent** - the update frequency formula can skip critical updates
#2. **No local search refinement** after CMA-ES converges
#3. **Opposition-based learning wastes budget** on poor reflections
#4. **Stagnation threshold too high** - wastes evaluations
#5. **Need better restart diversity** - use DE between CMA-ES restarts
#
import numpy as np
from datetime import datetime

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
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = [0]
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Latin Hypercube Sampling
    n_init = min(max(20 * dim, 150), 600)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() <= 0:
            return best
        init_fitness[i] = eval_func(init_pop[i])
    
    sorted_idx = np.argsort(init_fitness)
    
    # Keep archive of good solutions for diversity
    archive = []
    n_archive = min(20, n_init)
    for i in range(n_archive):
        archive.append((init_pop[sorted_idx[i]].copy(), init_fitness[sorted_idx[i]]))

    def update_archive(x, f):
        archive.append((x.copy(), f))
        archive.sort(key=lambda t: t[1])
        while len(archive) > 50:
            archive.pop()

    # CMA-ES
    def cma_es(x0, sigma0=0.3, pop_size=None, max_stag=None):
        nonlocal best, best_params
        n = dim
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(n))
        lam = pop_size
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        mean = x0.copy()
        sigma = sigma0
        
        if max_stag is None:
            max_stag = 10 + int(30 * (n / lam))
        
        stag_count = 0
        prev_median = float('inf')
        gen_count = 0
        best_local = float('inf')
        
        while time_left() > 0.5:
            arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz @ (B * D).T)
            arx = np.array([clip(x) for x in arx])
            
            fitness = np.zeros(lam)
            for k in range(lam):
                if time_left() <= 0.2:
                    return best_local
                fitness[k] = eval_func(arx[k])
            
            order = np.argsort(fitness)
            arx = arx[order]
            arz = arz[order]
            fitness = fitness[order]
            
            if fitness[0] < best_local:
                best_local = fitness[0]
                update_archive(arx[0], fitness[0])
            
            med_fit = np.median(fitness)
            if med_fit >= prev_median - 1e-12 * abs(prev_median + 1e-30):
                stag_count += 1
            else:
                stag_count = 0
            prev_median = min(prev_median, med_fit)
            
            if stag_count > max_stag:
                return best_local
            
            old_mean = mean.copy()
            mean = clip(weights @ arx[:mu])
            
            zmean = weights @ arz[:mu]
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen_count + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((mean - old_mean) / sigma)
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen_count += 1
            if gen_count % max(1, int(1.0 / (c1 + cmu_val + 1e-20) / n / 10)) == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D_sq, 1e-20))
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return best_local
            
            if sigma * np.max(D) < 1e-12:
                return best_local
        return best_local

    # BIPOP-CMA-ES
    base_pop = 4 + int(3 * np.log(dim))
    restart = 0
    large_budget_used = 0
    
    while time_left() > 1:
        if restart == 0:
            x0 = best_params.copy()
            sig = 0.25 * np.mean(ranges)
            ps = base_pop
        elif restart % 2 == 1:
            ps = int(base_pop * (2 ** ((restart + 1) // 2)))
            ps = min(ps, 512)
            idx = np.random.randint(min(5, len(archive)))
            x0 = archive[idx][0] + np.random.randn(dim) * 0.2 * ranges
            x0 = clip(x0)
            sig = 0.4 * np.mean(ranges)
        else:
            ps = int(base_pop * (0.5 + 0.5 * np.random.random()))
            x0 = best_params + np.random.randn(dim) * 0.05 * ranges * np.random.random()
            x0 = clip(x0)
            sig = 0.1 * np.mean(ranges) * (0.5 + np.random.random())
        
        cma_es(x0, sigma0=sig, pop_size=ps)
        restart += 1

    return best
