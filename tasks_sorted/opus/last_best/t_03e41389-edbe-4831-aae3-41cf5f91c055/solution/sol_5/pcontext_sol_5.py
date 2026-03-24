#Looking at the best algorithm (score 12.6), it uses full CMA-ES with covariance matrix, which works well. The last attempt (score 42.0) used diagonal CMA-ES which is less effective. Let me improve the best algorithm by:
#
#1. Better initial sampling and DE phase for global exploration
#2. More robust CMA-ES with eigendecomposition (avoiding Cholesky issues)
#3. IPOP-CMA-ES restart strategy with increasing population
#4. Better local search at the end
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
    n_init = min(max(40, 15 * dim), 400)
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

    # Phase 2: Short DE phase for global exploration
    de_size = min(max(20, 5*dim), 80)
    de_pop = np.array([x[1] for x in init_fits[:min(de_size, len(init_fits))]])
    while len(de_pop) < de_size:
        de_pop = np.vstack([de_pop, [lower + np.random.rand(dim) * ranges]])
    de_fit = np.array([init_fits[i][0] if i < len(init_fits) else evaluate(de_pop[i]) for i in range(de_size)])
    
    while elapsed() < max_time * 0.20:
        for i in range(de_size):
            if elapsed() >= max_time * 0.20:
                break
            idxs = list(range(de_size))
            idxs.remove(i)
            a, b, c = de_pop[np.random.choice(idxs, 3, replace=False)]
            F = 0.5 + 0.3 * np.random.rand()
            mutant = a + F * (b - c)
            # current-to-best
            mutant = de_pop[i] + F * (best_x - de_pop[i]) + F * (a - b)
            CR = 0.9
            mask = np.random.rand(dim) < CR
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, de_pop[i])
            trial = np.clip(trial, lower, upper)
            f_trial = evaluate(trial)
            if f_trial <= de_fit[i]:
                de_pop[i] = trial
                de_fit[i] = f_trial

    # Phase 3: IPOP-CMA-ES restarts
    def run_cmaes(x0, sigma0, deadline, lam_mult=1):
        n = dim
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cs = (mueff+2)/(n+mueff+5); cc = (4+mueff/n)/(n+4+2*mueff/n)
        c1 = 2/((n+1.3)**2+mueff); cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        mean = x0.copy(); sigma = sigma0; C = np.eye(n); ps = np.zeros(n); pc = np.zeros(n); gen = 0
        eigvals = np.ones(n); eigvecs = np.eye(n); update_gap = 0
        while elapsed() < min(deadline, max_time*0.93):
            gen += 1
            if gen > update_gap:
                try: eigvals, eigvecs = np.linalg.eigh(C); eigvals = np.maximum(eigvals, 1e-20)
                except: C = np.eye(n); eigvals = np.ones(n); eigvecs = np.eye(n)
                update_gap = gen + max(1, int(1.0/(c1+cmu)/n/10))
            D = np.sqrt(eigvals); arz = np.random.randn(lam, n)
            arx = mean + sigma * (arz * D) @ eigvecs.T; arx = np.clip(arx, lower, upper)
            fits = np.array([evaluate(arx[k]) for k in range(lam)]); idx = np.argsort(fits)
            xold = mean.copy(); mean = weights @ arx[idx[:mu]]
            zmean = weights @ arz[idx[:mu]]
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (eigvecs * D) @ zmean  # wrong, fix:
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * eigvecs @ (D * (eigvecs.T @ ((mean-xold)/sigma)))
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/chiN < 1.4+2/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(mean-xold)/sigma
            artmp = (arx[idx[:mu]]-xold)/sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(weights) @ artmp)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            if sigma < 1e-16 or sigma > 1e8*np.mean(ranges): break

    pop_mult = 1.0
    for r in range(10):
        rem = max_time*0.93 - elapsed()
        if rem < 0.2: break
        budget = rem / max(1, 4 - min(r, 3))
        x0 = best_x + 0.05*ranges*(np.random.rand(dim)-0.5) if r % 3 != 2 else lower+np.random.rand(dim)*ranges
        run_cmaes(np.clip(x0, lower, upper), 0.25*np.mean(ranges), elapsed()+budget, pop_mult)
        pop_mult *= 1.5

    # Phase 4: coordinate-wise local search
    if best_x is not None:
        scale = 0.002 * ranges
        for _ in range(30):
            if elapsed() >= max_time*0.99: break
            imp = False
            for d in range(dim):
                if elapsed() >= max_time*0.99: break
                for dr in [1, -1]:
                    t = best_x.copy(); t[d] += dr*scale[d]
                    if evaluate(t) < best: imp = True
            if not imp: scale *= 0.5
            if np.max(scale) < 1e-16: break
    return best
