#Looking at the previous algorithms, I see several issues to fix:
#
#1. The SHADE archive indexing is buggy (r2 can't index archive properly)
#2. Need better exploration-exploitation balance
#3. CMA-ES style local search would be more effective than Nelder-Mead for higher dimensions
#4. Should use multiple restarts of local search from diverse good solutions
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_x
        v = func(x)
        if v < best:
            best = v
            best_x = x.copy()
        return v

    # --- SHADE Differential Evolution ---
    pop_size = min(max(8 * dim, 40), 120)
    H = 50
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.8)
    k = 0
    
    # LHS init
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    pop = lower + pop * ranges
    pop = clip(pop)
    
    fit = np.array([ev(pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fit) < pop_size:
        fit = np.append(fit, [float('inf')] * (pop_size - len(fit)))
    
    archive = []
    gen = 0
    
    while elapsed() < max_time * 0.60:
        S_F, S_CR, S_delta = [], [], []
        gen += 1
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.60:
                break
            
            ri = np.random.randint(H)
            mu_F, mu_CR = memory_F[ri], memory_CR[ri]
            Fi = np.clip(mu_F + 0.1 * np.random.standard_cauchy(), 0.01, 1.5)
            CRi = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
            
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(fit)[:p])
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            combined_size = pop_size + len(archive)
            r2_pool = [j for j in range(combined_size) if j != i and j != r1]
            r2 = np.random.choice(r2_pool)
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            
            mask = np.random.random(dim) < CRi
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            trial = clip(np.where(mask, mutant, pop[i]))
            
            trial_fit = ev(trial)
            
            if trial_fit <= fit[i]:
                delta = fit[i] - trial_fit
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                if len(archive) < pop_size:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial
                fit[i] = trial_fit
        
        if S_F:
            w = np.array(S_delta); w = w / (w.sum() + 1e-30)
            memory_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            memory_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H

    # --- CMA-ES-like local search around top solutions ---
    top_k = min(5, pop_size)
    top_indices = np.argsort(fit)[:top_k]
    
    for ti in range(top_k):
        if elapsed() >= max_time * 0.96:
            break
        sigma = 0.05
        mean = pop[top_indices[ti]].copy()
        C = np.eye(dim)
        lam = max(4 + int(3 * np.log(dim)), 8)
        mu_cma = lam // 2
        weights = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        ps = np.zeros(dim); pc = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        for _ in range(10000):
            if elapsed() >= max_time * 0.96:
                break
            try:
                sqrtC = np.linalg.cholesky(C)
            except:
                C = np.eye(dim); sqrtC = np.eye(dim)
            zs = np.random.randn(lam, dim)
            xs = np.array([clip(mean + sigma * sqrtC @ z) for z in zs])
            fs = np.array([ev(x) for x in xs])
            idx = np.argsort(fs)
            zmean = np.sum(weights[:, None] * zs[idx[:mu_cma]], axis=0)
            mean_new = clip(mean + sigma * sqrtC @ zmean)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * zmean
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(_+1))) < (1.4 + 2/(dim+1)) * chiN)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (sqrtC @ zmean)
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for j in range(mu_cma):
                C += cmu * weights[j] * np.outer(sqrtC @ zs[idx[j]], sqrtC @ zs[idx[j]])
            C = (C + C.T) / 2 + 1e-10 * np.eye(dim)
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, np.max(ranges))
            mean = mean_new

    while elapsed() < max_time * 0.998:
        candidate = clip(best_x + np.random.randn(dim) * ranges * 0.002)
        ev(candidate)
    
    return best
