#Looking at the results, the best algorithm (8.08) used SHADE + CMA-ES. I'll analyze the issues and improve:
#
#1. **Better CMA-ES implementation** - use eigendecomposition instead of Cholesky (more numerically stable)
#2. **Smarter time allocation** - give CMA-ES more time with proper restarts
#3. **L-SHADE with linear population reduction** for better DE convergence
#4. **Multiple CMA-ES restarts with varying sigma** to escape local optima
#5. **Coordinate descent refinement** at the end
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

    # --- L-SHADE ---
    pop_size_init = min(max(10 * dim, 60), 180)
    pop_size = pop_size_init
    H = 80
    memory_F = np.full(H, 0.5)
    memory_CR = np.full(H, 0.8)
    k = 0
    min_pop = max(4, dim // 2)
    
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, d] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([ev(pop[i]) for i in range(pop_size)])
    archive = []
    nfe = pop_size
    max_nfe = pop_size_init * 350
    de_time = max_time * 0.45
    
    while elapsed() < de_time and nfe < max_nfe:
        S_F, S_CR, S_delta = [], [], []
        sorted_idx = np.argsort(fit)
        
        for i in range(pop_size):
            if elapsed() >= de_time:
                break
            ri = np.random.randint(H)
            Fi = -1
            while Fi <= 0:
                Fi = memory_F[ri] + 0.1 * np.random.standard_cauchy()
            Fi = min(Fi, 1.5)
            CRi = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = idxs[np.random.randint(len(idxs))]
            
            combined_size = pop_size + len(archive)
            while True:
                r2 = np.random.randint(combined_size)
                if r2 != i and r2 != r1:
                    break
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
            for d2 in range(dim):
                if mutant[d2] < lower[d2]:
                    mutant[d2] = (lower[d2] + pop[i][d2]) / 2
                elif mutant[d2] > upper[d2]:
                    mutant[d2] = (upper[d2] + pop[i][d2]) / 2
            
            jrand = np.random.randint(dim)
            trial = pop[i].copy()
            for d2 in range(dim):
                if np.random.random() < CRi or d2 == jrand:
                    trial[d2] = mutant[d2]
            
            tf = ev(trial)
            nfe += 1
            if tf <= fit[i]:
                delta = fit[i] - tf
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                if len(archive) < pop_size_init:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial; fit[i] = tf
        
        if S_F:
            w = np.array(S_delta); w /= w.sum() + 1e-30
            memory_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            memory_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        ratio = min(1.0, nfe / max_nfe)
        new_size = max(min_pop, int(round(pop_size_init + (min_pop - pop_size_init) * ratio)))
        if new_size < pop_size:
            idx_keep = np.argsort(fit)[:new_size]
            pop = pop[idx_keep]; fit = fit[idx_keep]; pop_size = new_size

    # --- CMA-ES restarts from diverse good solutions ---
    cands = []
    if best_x is not None:
        cands.append((best_x.copy(), 0.02))
    top_idx = np.argsort(fit)[:min(4, pop_size)]
    for ti in top_idx:
        cands.append((pop[ti].copy(), 0.05))
    # Add a wider restart
    if best_x is not None:
        cands.append((best_x.copy(), 0.15))
    
    for ci, (cand, sig_scale) in enumerate(cands):
        if elapsed() >= max_time * 0.93:
            break
        tleft = max_time * 0.93 - elapsed()
        deadline = elapsed() + tleft / max(1, len(cands) - ci)
        
        n = dim
        m = cand.copy()
        sigma = sig_scale * np.max(ranges)
        lam = max(4 + int(3 * np.log(n)), 8)
        mu_c = lam // 2
        w = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu_v = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        ps = np.zeros(n); pc = np.zeros(n); C = np.eye(n)
        D = np.ones(n); B = np.eye(n)
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        eg = -1
        
        for g in range(100000):
            if elapsed() >= deadline:
                break
            if g - eg > max(1, 1/(c1 + cmu_v)/n/10) or g == 0:
                C = (C + C.T) / 2
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-20))
                eg = g
            BD = B * D
            zs = np.random.randn(lam, n)
            ys = (BD @ zs.T).T
            xs = np.array([clip(m + sigma * y) for y in ys])
            fs = np.array([ev(x) for x in xs])
            idx = np.argsort(fs)
            yw = w @ ys[idx[:mu_c]]
            zw = w @ zs[idx[:mu_c]]
            m = clip(m + sigma * yw)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ zw)
            hn = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(g+1))) < (1.4 + 2/(n+1)) * chiN)
            pc = (1 - cc) * pc + hn * np.sqrt(cc * (2 - cc) * mu_eff) * yw
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hn) * cc * (2 - cc) * C)
            for j in range(mu_c):
                C += cmu_v * w[j] * np.outer(ys[idx[j]], ys[idx[j]])
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-16, np.max(ranges))
            if sigma < 1e-14:
                break

    # Final local perturbation
    while elapsed() < max_time * 0.999:
        step = ranges * 0.001 * np.random.randn(dim)
        ev(clip(best_x + step))
    
    return best
