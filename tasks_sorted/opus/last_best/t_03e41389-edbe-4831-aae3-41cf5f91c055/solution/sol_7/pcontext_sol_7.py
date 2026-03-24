#Looking at the code, I can identify several issues with the CMA-ES implementation and overall strategy. Let me fix the CMA-ES (proper covariance update using the original space vectors), improve the DE phase, add better restart logic, and refine local search.
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

    # Phase 1: LHS initial sampling
    n_init = min(max(50, 20 * dim), 500)
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

    # Phase 2: DE/current-to-pbest with archive
    de_size = min(max(30, 6 * dim), 100)
    de_pop = np.array([x[1] for x in init_fits[:min(de_size, len(init_fits))]])
    while len(de_pop) < de_size:
        de_pop = np.vstack([de_pop, lower + np.random.rand(dim) * ranges])
    de_fit = np.array([init_fits[i][0] if i < len(init_fits) else evaluate(de_pop[i]) for i in range(de_size)])
    
    archive = []
    
    # Adaptive parameters (SHADE-like)
    mem_size = 5
    mem_F = np.full(mem_size, 0.5)
    mem_CR = np.full(mem_size, 0.8)
    mem_idx = 0
    
    de_deadline = max_time * 0.25
    gen = 0
    while elapsed() < de_deadline:
        gen += 1
        S_F, S_CR, S_df = [], [], []
        p_best = max(2, int(0.1 * de_size))
        
        for i in range(de_size):
            if elapsed() >= de_deadline:
                break
            ri = np.random.randint(mem_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + mem_F[ri], 0.1, 1.0)
            CRi = np.clip(np.random.randn() * 0.1 + mem_CR[ri], 0.0, 1.0)
            
            # p-best
            p_idx = np.argsort(de_fit)[:p_best]
            xpbest = de_pop[np.random.choice(p_idx)]
            
            idxs = list(range(de_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            union = list(range(de_size)) + list(range(len(archive)))
            r2_idx = np.random.choice(len(union))
            if r2_idx < de_size:
                xr2 = de_pop[r2_idx] if r2_idx != i else de_pop[r1]
            else:
                xr2 = archive[r2_idx - de_size]
            
            mutant = de_pop[i] + Fi * (xpbest - de_pop[i]) + Fi * (de_pop[r1] - xr2)
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, de_pop[i])
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial < de_fit[i]:
                S_F.append(Fi); S_CR.append(CRi); S_df.append(de_fit[i] - f_trial)
                archive.append(de_pop[i].copy())
                if len(archive) > de_size: archive.pop(np.random.randint(len(archive)))
                de_pop[i] = trial; de_fit[i] = f_trial
            elif f_trial == de_fit[i]:
                de_pop[i] = trial; de_fit[i] = f_trial
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            mem_F[mem_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            mem_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % mem_size

    # Phase 3: CMA-ES with IPOP restarts
    def run_cmaes(x0, sigma0, deadline, lam=None):
        n = dim
        if lam is None:
            lam = 4 + int(3 * np.log(n))
        lam = max(lam, 8)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        c1 = 2 / ((n+1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n)*(1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy(); sigma = sigma0
        C = np.eye(n); ps = np.zeros(n); pc = np.zeros(n)
        g = 0
        eigeneval = 0; counteval = 0
        B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
        
        while elapsed() < min(deadline, max_time * 0.94):
            g += 1
            arz = np.random.randn(lam, n)
            arx = mean[None, :] + sigma * (arz @ (B * D[None, :]) @ B.T)
            arx = np.clip(arx, lower, upper)
            fits = np.array([evaluate(arx[k]) for k in range(lam)])
            counteval += lam
            
            idx = np.argsort(fits)
            xold = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            zmean = weights @ arz[idx[:mu]]
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ (mean - xold)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*g))/chiN < 1.4 + 2/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff) * (mean - xold)/sigma
            
            artmp = (arx[idx[:mu]] - xold) / sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(weights) @ artmp)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            
            if counteval - eigeneval > lam/(c1+cmu)/n/10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1/D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            if sigma < 1e-16 or sigma > 1e8: break

    lam_base = 4 + int(3 * np.log(dim))
    lam_mult = 1
    for r in range(10):
        rem = max_time * 0.94 - elapsed()
        if rem < 0.2: break
        budget = rem / max(1, 5 - min(r, 4))
        
        if r == 0:
            x0 = best_x.copy(); sig = 0.2 * np.mean(ranges)
        elif r % 2 == 1:
            x0 = best_x + 0.05 * ranges * (np.random.rand(dim) - 0.5)
            x0 = np.clip(x0, lower, upper); sig = 0.08 * np.mean(ranges)
        else:
            x0 = lower + np.random.rand(dim) * ranges; sig = 0.3 * np.mean(ranges)
            lam_mult *= 2
        
        run_cmaes(x0, sig, elapsed() + budget, int(lam_base * (lam_mult if r % 2 == 0 else 1)))

    # Phase 4: Coordinate descent with golden section
    if best_x is not None:
        scale = 0.005 * ranges
        for _ in range(40):
            if elapsed() >= max_time * 0.995: break
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if elapsed() >= max_time * 0.995: break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[d] += direction * scale[d]
                    if evaluate(trial) < best:
                        improved = True
                        # Keep going in this direction
                        step = scale[d] * direction * 2
                        for _ in range(5):
                            trial2 = best_x.copy()
                            trial2[d] += step
                            if evaluate(trial2) < best:
                                step *= 2
                            else:
                                break
            if not improved:
                scale *= 0.4
                if np.max(scale) < 1e-16: break
    
    return best
#
#Key improvements:
#1. **Fixed CMA-ES**: Proper eigen-decomposition based sampling and update (B, D decomposition), correct `ps` update using `invsqrtC`, and lazy eigenvalue recomputation.
#2. **SHADE-style adaptive DE**: Uses success-history based parameter adaptation for F and CR, p-best mutation strategy, and archive.
#3. **Better IPOP restarts**: Doubles population size on exploration restarts while keeping small populations for exploitation restarts.
#4. **Improved local search**: Accelerated coordinate descent that continues stepping in successful directions.
#5. **Better time allocation**: More time for DE exploration phase, smarter CMA-ES budget splitting.
