#Looking at the best algorithm (score 12.6), I can see it uses CMA-ES with full covariance matrix, which is expensive for higher dimensions. The last attempt (score 28.2) uses sep-CMA which is faster but has issues with restart strategy. Let me improve by:
#
#1. Using a diagonal CMA-ES that's more efficient per iteration
#2. Better restart strategy (IPOP with adaptive restarts from diverse points)
#3. Adding a DE phase for global exploration before CMA-ES refinement
#4. More sophisticated local search at the end
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: LHS initial sampling
    n_init = min(max(50, 12 * dim), 500)
    pop = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, i] = (perm + np.random.rand(n_init)) / n_init
    pop = lower + pop * ranges
    
    init_fits = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.10:
            break
        init_fits.append((evaluate(pop[i]), pop[i].copy()))
    
    init_fits.sort(key=lambda x: x[0])
    
    # Phase 2: Differential Evolution for global search
    de_pop_size = min(max(20, 5 * dim), 100)
    de_pop = np.array([x[1] for x in init_fits[:min(de_pop_size, len(init_fits))]])
    while len(de_pop) < de_pop_size:
        de_pop = np.vstack([de_pop, lower + np.random.rand(dim) * ranges])
    de_fit = np.array([evaluate(de_pop[i]) if i >= len(init_fits) else init_fits[i][0] if i < len(init_fits) else evaluate(de_pop[i]) for i in range(de_pop_size)])
    
    F = 0.8
    CR = 0.9
    while elapsed() < max_time * 0.35:
        for i in range(de_pop_size):
            if elapsed() >= max_time * 0.35:
                break
            idxs = np.random.choice(de_pop_size, 3, replace=False)
            while i in idxs:
                idxs = np.random.choice(de_pop_size, 3, replace=False)
            a, b, c = de_pop[idxs[0]], de_pop[idxs[1]], de_pop[idxs[2]]
            # current-to-best
            mutant = de_pop[i] + F * (best_x - de_pop[i]) + F * (a - b)
            mask = np.random.rand(dim) < CR
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, de_pop[i])
            trial = np.clip(trial, lower, upper)
            f_trial = evaluate(trial)
            if f_trial <= de_fit[i]:
                de_pop[i] = trial
                de_fit[i] = f_trial

    # Phase 3: CMA-ES restarts from best
    def run_cmaes(x0, sigma0, deadline, pop_mult=1):
        n = dim
        lam = int((4 + int(3 * np.log(n))) * pop_mult)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n); cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff); cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        mean = x0.copy(); sigma = sigma0; diagC = np.ones(n); ps = np.zeros(n); pc = np.zeros(n); gen = 0
        while elapsed() < min(deadline, max_time*0.93):
            gen += 1; sqrtD = np.sqrt(diagC); arz = np.random.randn(lam, n)
            arx = np.clip(mean + sigma*arz*sqrtD, lower, upper)
            fs = np.array([evaluate(arx[k]) for k in range(lam)]); idx = np.argsort(fs)
            xold = mean.copy(); mean = weights @ arx[idx[:mu]]
            diff = (mean-xold)/(sigma*sqrtD+1e-30); ps = (1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*diff
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/chiN < 1.4+2/(n+1)
            pc = (1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*(mean-xold)/sigma
            artmp = (arx[idx[:mu]]-xold)/sigma; diagC = (1-c1-cmu)*diagC+c1*(pc**2+(1-hsig)*cc*(2-cc)*diagC)+cmu*(weights@artmp**2)
            diagC = np.clip(diagC, 1e-20, 1e20); sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            if sigma < 1e-16: break

    pm = 1.0
    for r in range(8):
        rem = max_time*0.93 - elapsed()
        if rem < 0.1: break
        budget = rem / max(1, 3-r) if r < 2 else rem
        x0 = best_x + 0.05*ranges*(np.random.rand(dim)-0.5) if r%3!=2 else lower+np.random.rand(dim)*ranges
        run_cmaes(np.clip(x0, lower, upper), 0.2*np.mean(ranges)/pm, elapsed()+budget, pm)
        pm *= 1.3

    # Phase 4: coordinate descent
    if best_x is not None:
        scale = 0.001*ranges
        for _ in range(50):
            if elapsed() >= max_time*0.99: break
            imp = False
            for d in range(dim):
                if elapsed() >= max_time*0.99: break
                for dr in [1,-1]:
                    t = best_x.copy(); t[d] += dr*scale[d]
                    if evaluate(t) < best: imp = True
            if not imp: scale *= 0.5
            if np.max(scale) < 1e-16: break
    return best
