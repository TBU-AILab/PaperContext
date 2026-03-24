#Looking at the previous algorithms, the best one (score 12.6) uses CMA-ES with restarts and local search. The last attempt (score 141.6) has a bug where `run_cmaes` is defined but the calling code is outside the function scope. Let me fix and improve the approach:
#
#1. Fix the CMA-ES implementation with proper sep-CMA for high dimensions
#2. Better restart strategy with IPOP (increasing population)
#3. Differential evolution as alternative for diverse search
#4. More robust local search
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
    n_init = min(max(40, 10 * dim), 400)
    pop = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, i] = (perm + np.random.rand(n_init)) / n_init
    pop = lower + pop * ranges
    
    fits = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.12:
            break
        fits.append(evaluate(pop[i]))
    
    # Phase 2: CMA-ES with IPOP restarts
    def run_sep_cmaes(x0, sigma0, time_budget, pop_mult=1):
        nonlocal best, best_x
        deadline = elapsed() + time_budget
        n = dim
        lam = int((4 + int(3 * np.log(n))) * pop_mult)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + cs
        chiN = np.sqrt(n)*(1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        diagC = np.ones(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        gen = 0
        stag = 0
        prev_best = best
        
        while elapsed() < min(deadline, max_time * 0.93):
            gen += 1
            sqrtD = np.sqrt(diagC)
            arz = np.random.randn(lam, n)
            arx = mean[None,:] + sigma * arz * sqrtD[None,:]
            arx = np.clip(arx, lower, upper)
            
            fs = np.array([evaluate(arx[k]) for k in range(lam)])
            idx = np.argsort(fs)
            
            xold = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            diff = (mean - xold) / (sigma * sqrtD)
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * diff
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/chiN < 1.4+2/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(mean-xold)/sigma
            
            artmp = (arx[idx[:mu]] - xold) / sigma
            diagC = (1-c1-cmu)*diagC + c1*(pc**2 + (1-hsig)*cc*(2-cc)*diagC) + cmu*(weights @ artmp**2)
            diagC = np.clip(diagC, 1e-20, 1e20)
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges)*2)
            
            if best < prev_best: stag = 0; prev_best = best
            else: stag += 1
            if stag > 10 + 30*n/lam or sigma < 1e-16: break

    pop_mult = 1.0
    for r in range(10):
        rem = max_time * 0.93 - elapsed()
        if rem < 0.1: break
        budget = rem / max(1, 4 - r) if r < 3 else rem
        x0 = best_x + 0.1*(r>0)*ranges*(np.random.rand(dim)-0.5) if r % 2 == 0 else lower + np.random.rand(dim)*ranges
        x0 = np.clip(x0, lower, upper)
        run_sep_cmaes(x0, 0.25*np.mean(ranges), budget, pop_mult)
        pop_mult *= 1.5

    # Phase 3: Coordinate-wise local search
    if best_x is not None:
        scale = 0.001 * ranges
        for _ in range(30):
            if elapsed() >= max_time*0.99: break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time*0.99: break
                for dr in [1, -1]:
                    trial = best_x.copy(); trial[d] += dr*scale[d]
                    if evaluate(trial) < best: improved = True
            if not improved:
                scale *= 0.5
                if np.max(scale) < 1e-16: break

    return best
