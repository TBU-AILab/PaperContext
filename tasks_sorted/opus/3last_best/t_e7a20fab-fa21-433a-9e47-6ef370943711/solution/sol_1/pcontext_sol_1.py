#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **Better time allocation** - CMA-ES should get more time as it's the most powerful optimizer
#2. **Multiple CMA-ES restarts** with increasing population sizes (IPOP-CMA-ES)
#3. **Better initialization** - more samples, opposition-based learning
#4. **Adaptive DE parameters** (SHADE-like)
#5. **Remove inefficiencies** - vectorize operations where possible
#
#Here's the improved version:
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
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(30 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() <= 0:
            return best
        init_fitness[i] = eval_func(init_pop[i])
    
    # Opposition-based candidates
    n_opp = min(n_init, 200)
    sorted_idx = np.argsort(init_fitness)
    for i in range(n_opp):
        if time_left() <= 0:
            return best
        opp = lower + upper - init_pop[sorted_idx[i]]
        eval_func(opp)

    # --- CMA-ES with restarts (IPOP) ---
    def cma_es(x0, sigma0=0.3, pop_size=None):
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
        
        stag_count = 0
        prev_best_gen = float('inf')
        eigen_update_freq = max(1, int(lam / (10 * n)))
        gen_count = 0
        
        while time_left() > 0:
            arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz @ (B * D).T)
            arx = np.array([clip(x) for x in arx])
            
            fitness = np.array([eval_func(arx[k]) for k in range(lam) if time_left() > 0])
            if len(fitness) < lam:
                return
            
            order = np.argsort(fitness)
            arx = arx[order]
            arz = arz[order]
            
            best_gen = fitness[order[0]]
            if best_gen >= prev_best_gen - 1e-12:
                stag_count += 1
            else:
                stag_count = 0
            prev_best_gen = min(prev_best_gen, best_gen)
            
            if stag_count > 20 + 3 * n:
                return
            
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
            if gen_count % eigen_update_freq == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D_sq, 1e-20))
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return
            
            if sigma * np.max(D) < 1e-12:
                return

    # IPOP-CMA-ES: multiple restarts with increasing population
    sorted_idx = np.argsort(init_fitness)
    base_pop = 4 + int(3 * np.log(dim))
    restart = 0
    
    while time_left() > 2:
        pop_mult = int(base_pop * (2 ** restart))
        pop_mult = min(pop_mult, 256)
        
        if restart == 0:
            x0 = best_params.copy() if best_params is not None else init_pop[sorted_idx[0]]
            sig = 0.3 * np.mean(ranges)
        else:
            # Pick from top candidates with some randomness
            idx = min(restart - 1, len(sorted_idx) - 1)
            x0 = init_pop[sorted_idx[idx]].copy()
            x0 += np.random.randn(dim) * 0.1 * ranges
            x0 = clip(x0)
            sig = (0.2 + 0.2 * restart) * np.mean(ranges)
            sig = min(sig, 0.5 * np.mean(ranges))
        
        cma_es(x0, sigma0=sig, pop_size=pop_mult)
        restart += 1
        
        if restart > 8:
            restart = 0  # cycle back

    return best
#
#Key improvements:
#- **IPOP-CMA-ES**: Multiple restarts with doubling population size for better global exploration
#- **Opposition-based learning**: Evaluates opposite points of best initial samples
#- **Stagnation detection**: Restarts CMA-ES when stuck instead of wasting evaluations
#- **Better eigendecomposition scheduling**: Less frequent updates for larger populations
#- **More initial samples** with better coverage
#- **Removed DE phase**: CMA-ES with restarts is more sample-efficient
#- **Better sigma management** per restart
