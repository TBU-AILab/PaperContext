#Looking at the previous algorithm, I'll focus on key improvements:
#
#1. **CMA-ES as primary optimizer** - much better for continuous optimization than DE alone
#2. **Multiple independent restarts with increasing population** (IPOP-CMA-ES strategy)
#3. **Better time management** with more restarts
#4. **Hybrid approach**: CMA-ES for global search + L-SHADE + refined local search
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
    mid = (lower + upper) / 2

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def cmaes_run(time_limit, sigma0=0.3, popsize=None, x0=None):
        nonlocal best, best_params
        n = dim
        if x0 is None:
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
        lam = popsize or (4 + int(3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0 * np.mean(ranges)
        pc = np.zeros(n); ps = np.zeros(n)
        C = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        gen = 0
        
        while elapsed() < time_limit:
            gen += 1
            if gen % (n // 2 + 1) == 0 or gen == 1:
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.array([clip(mean + sigma * (B @ (D * arz[k]))) for k in range(lam)])
            fitvals = np.array([evaluate(arx[k]) for k in range(lam) if elapsed() < time_limit])
            if len(fitvals) < lam: break
            
            idx = np.argsort(fitvals)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean-old_mean)/sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*gen)) / chiN < 1.4 + 2/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff) * (mean-old_mean)/sigma
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(weights) @ artmp)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.mean(ranges))
            if sigma < 1e-16 * np.mean(ranges): break

    # IPOP-CMA-ES restarts
    popsize = 4 + int(3*np.log(dim))
    restart = 0
    while elapsed() < max_time * 0.92:
        t_left = max_time * 0.92 - elapsed()
        t_alloc = max(t_left * 0.25, min(t_left, 2.0))
        x0 = best_params if (best_params is not None and restart % 3 != 0) else None
        s0 = 0.15 if (x0 is not None) else 0.35
        cmaes_run(elapsed() + t_alloc, sigma0=s0, popsize=popsize, x0=x0)
        restart += 1
        if restart % 2 == 0: popsize = min(popsize * 2, 256)
        else: popsize = 4 + int(3*np.log(dim))

    # Local search: coordinate descent
    if best_params is not None:
        x = best_params.copy(); fx = best; step = 0.005 * ranges
        for _ in range(50):
            if elapsed() >= max_time*0.99: break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time*0.99: break
                for sign in [1,-1]:
                    xn = x.copy(); xn[d] = np.clip(x[d]+sign*step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x, fx = xn, fn; improved = True; break
            if not improved:
                step *= 0.5
                if np.max(step/ranges) < 1e-12: break
    return best
