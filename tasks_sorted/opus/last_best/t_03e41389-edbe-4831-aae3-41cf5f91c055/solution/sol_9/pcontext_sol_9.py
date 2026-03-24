#Looking at the previous implementations, I can see the CMA-ES has issues with the Cholesky-based covariance update and the overall time allocation could be improved. Let me build a more robust optimizer with proper SHADE-style DE, eigendecomposition-based CMA-ES with multiple restarts, and better local refinement.
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
    
    def evaluate(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: LHS sampling
    n_init = min(max(60, 20 * dim), 500)
    pop = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, i] = (perm + np.random.rand(n_init)) / n_init
    pop = lower + pop * ranges
    
    init_fits = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.08:
            break
        f = evaluate(pop[i])
        init_fits.append((f, pop[i].copy()))
    init_fits.sort(key=lambda x: x[0])

    # Phase 2: SHADE DE
    NP = min(max(40, 8 * dim), 150)
    de_pop = np.array([x[1] for x in init_fits[:min(NP, len(init_fits))]])
    while len(de_pop) < NP:
        de_pop = np.vstack([de_pop, lower + np.random.rand(dim) * ranges])
    de_fit = np.array([init_fits[i][0] if i < len(init_fits) else evaluate(de_pop[i]) for i in range(NP)])
    
    archive = []
    H = 6
    mem_F = np.full(H, 0.5)
    mem_CR = np.full(H, 0.8)
    k_idx = 0
    
    de_deadline = max_time * 0.35
    gen = 0
    while elapsed() < de_deadline:
        gen += 1
        S_F, S_CR, S_df = [], [], []
        p_best_rate = max(2/NP, 0.05 + 0.1 * np.random.rand())
        
        trial_pop = de_pop.copy()
        trial_fit = de_fit.copy()
        
        for i in range(NP):
            if elapsed() >= de_deadline:
                break
            ri = np.random.randint(H)
            Fi = np.clip(mem_F[ri] + 0.1 * np.random.standard_cauchy(), 0.05, 1.0)
            CRi = np.clip(mem_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(p_best_rate * NP))
            p_idx = np.argsort(de_fit)[:p]
            xpbest = de_pop[np.random.choice(p_idx)]
            
            candidates = list(range(NP))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            candidates.remove(r1)
            
            ext_pool = list(candidates)
            for ai in range(len(archive)):
                ext_pool.append(NP + ai)
            r2 = np.random.choice(ext_pool) if ext_pool else r1
            xr2 = de_pop[r2] if r2 < NP else archive[r2 - NP]
            
            v = de_pop[i] + Fi * (xpbest - de_pop[i]) + Fi * (de_pop[r1] - xr2)
            
            # Binomial crossover
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, v, de_pop[i])
            
            # Bounce-back
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (de_pop[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - de_pop[i][d])
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial < de_fit[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(de_fit[i] - f_trial)
                archive.append(de_pop[i].copy())
                if len(archive) > NP:
                    archive.pop(np.random.randint(len(archive)))
                trial_pop[i] = trial
                trial_fit[i] = f_trial
            elif f_trial == de_fit[i]:
                trial_pop[i] = trial
                trial_fit[i] = f_trial
        
        de_pop = trial_pop
        de_fit = trial_fit
        
        if S_F:
            w = np.array(S_df)
            w = w / (w.sum() + 1e-30)
            mem_F[k_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            mem_CR[k_idx] = np.sum(w * np.array(S_CR))
            k_idx = (k_idx + 1) % H

    # Phase 3: CMA-ES with restarts
    def run_cmaes(x0, sigma0, deadline):
        n = dim
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 8)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        ds = 1 + 2 * max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n*n))
        
        m = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        eigeneval = 0
        g = 0
        
        while elapsed() < min(deadline, max_time * 0.94):
            g += 1
            arz = np.random.randn(lam, n)
            arx = m + sigma * (arz @ np.diag(D) @ B.T)
            arx = np.clip(arx, lower, upper)
            
            fits = np.array([evaluate(arx[j]) for j in range(lam)])
            if elapsed() >= deadline:
                break
            
            idx = np.argsort(fits)
            xmean_old = m.copy()
            m = w @ arx[idx[:mu]]
            
            zmean = w @ arz[idx[:mu]]
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (m - xmean_old) / sigma)
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*g)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((m - xmean_old) / sigma)
            
            artmp = (arx[idx[:mu]] - xmean_old) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(w) @ artmp)
            
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 1e6)
            
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu + 1e-30) / n / 10:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-20))
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            if sigma < 1e-18 or sigma > 1e8:
                break
            if g > 2 and np.max(D) / np.min(D) > 1e7:
                break

    for r in range(10):
        rem = max_time * 0.94 - elapsed()
        if rem < 0.2:
            break
        bud = rem / max(1, 5 - min(r, 4))
        if r == 0:
            x0 = best_x.copy()
            sig = 0.15 * np.mean(ranges)
        elif r == 1:
            x0 = best_x + 0.01 * ranges * (np.random.rand(dim) - 0.5)
            x0 = np.clip(x0, lower, upper)
            sig = 0.03 * np.mean(ranges)
        elif r % 3 == 0:
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.3 * np.mean(ranges)
        else:
            x0 = best_x + 0.05 * ranges * (np.random.rand(dim) - 0.5)
            x0 = np.clip(x0, lower, upper)
            sig = 0.08 * np.mean(ranges)
        run_cmaes(x0, sig, elapsed() + bud)

    # Phase 4: Coordinate-wise local search
    if best_x is not None:
        sc = 0.005 * ranges
        for iteration in range(100):
            if elapsed() >= max_time * 0.998:
                break
            improved = False
            order = np.random.permutation(dim)
            for d in order:
                if elapsed() >= max_time * 0.998:
                    break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[d] += direction * sc[d]
                    trial = np.clip(trial, lower, upper)
                    f_trial = evaluate(trial)
                    if f_trial < best:
                        improved = True
                        # Accelerate in this direction
                        for _ in range(10):
                            trial2 = best_x.copy()
                            trial2[d] += direction * sc[d] * 2
                            trial2 = np.clip(trial2, lower, upper)
                            if evaluate(trial2) < best:
                                sc[d] *= 1.5
                            else:
                                break
                        break
            if not improved:
                sc *= 0.4
                if np.max(sc / ranges) < 1e-15:
                    break
    
    return best
