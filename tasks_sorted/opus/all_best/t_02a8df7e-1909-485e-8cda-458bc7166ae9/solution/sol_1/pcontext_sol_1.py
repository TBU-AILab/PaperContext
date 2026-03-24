#Looking at the previous algorithm, I see several areas for improvement:
#
#1. **CMA-ES implementation has issues** - the restart strategy and time budget management are suboptimal
#2. **Need better exploration-exploitation balance** - more aggressive restarts with varying sigma
#3. **Diagonal CMA-ES** would be faster for high dimensions
#4. **Better local search** - pattern search with adaptive step sizes
#
#Let me create an improved version with a cleaner CMA-ES, better restart strategy (IPOP-CMA-ES style), and improved local refinement:
#
import numpy as np
from datetime import datetime, timedelta

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
        return max_time * 0.98 - elapsed()
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Stratified random sampling
    n_init = min(max(30 * dim, 200), 1000)
    
    pop = []
    fits = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        x = lower + np.random.rand(dim) * ranges
        f = evaluate(x)
        pop.append(x)
        fits.append(f)
    
    pop = np.array(pop)
    fits = np.array(fits)
    
    # Phase 2: CMA-ES with IPOP restarts
    def cma_es(x0, sigma0, max_evals_hint=None):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dim
        use_sep = (n > 40)
        if use_sep:
            diagC = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
        
        eigeneval = 0
        counteval = 0
        gen = 0
        stag_count = 0
        best_local = float('inf')
        
        while time_left() > 0:
            if use_sep:
                sqrtC = np.sqrt(diagC)
                arz = np.random.randn(lam, n)
                arx = mean[None, :] + sigma * (arz * sqrtC[None, :])
            else:
                if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        eigvals, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(eigvals, 1e-20))
                    except:
                        C = np.eye(n); D = np.ones(n); B = np.eye(n)
                
                arz = np.random.randn(lam, n)
                arx = mean[None, :] + sigma * (arz @ np.diag(D) @ B.T)
            
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.array([evaluate(arx[k]) for k in range(lam)])
            counteval += lam
            
            if time_left() <= 0:
                return
            
            idx = np.argsort(arfitness)
            
            local_best = arfitness[idx[0]]
            if local_best < best_local - 1e-10:
                best_local = local_best
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            diff = (mean - old_mean) / sigma
            
            if use_sep:
                invsqrtC = 1.0 / sqrtC
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC * diff)
                hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff
                
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                diagC = (1-c1-cmu_val)*diagC + c1*(pc**2 + (1-hsig)*cc*(2-cc)*diagC) + cmu_val*(weights @ (artmp**2))
                diagC = np.maximum(diagC, 1e-20)
            else:
                invsqrt = B @ np.diag(1.0/D) @ B.T
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrt @ diff)
                hsig = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN < 1.4+2/(n+1))
                pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*diff
                
                artmp = (arx[idx[:mu]] - old_mean)/sigma
                C = (1-c1-cmu_val)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C) + cmu_val*(artmp.T@np.diag(weights)@artmp)
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2*np.max(ranges))
            
            gen += 1
            if sigma < 1e-12 or stag_count > 20 + 30*n/lam or gen > 300 + 200*n:
                return
    
    # IPOP restarts
    sorted_idx = np.argsort(fits)
    lam_mult = 1
    restart = 0
    
    while time_left() > 1.0:
        if restart < min(5, len(sorted_idx)):
            x0 = pop[sorted_idx[restart]].copy()
        elif best_params is not None and np.random.rand() < 0.6:
            x0 = best_params + 0.2 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
        else:
            x0 = lower + np.random.rand(dim) * ranges
        
        sigma0 = 0.25 * np.mean(ranges) / max(1, lam_mult**0.25)
        
        cma_es(x0, sigma0)
        
        restart += 1
        if restart > 3:
            lam_mult *= 1.5
    
    # Phase 3: Fine local search around best
    if best_params is not None:
        current = best_params.copy()
        current_fit = best
        for scale in [0.01, 0.001, 0.0001]:
            step = scale * ranges
            for _ in range(3):
                if time_left() <= 0:
                    return best
                for i in range(dim):
                    if time_left() <= 0:
                        return best
                    for d in [1, -1]:
                        trial = current.copy()
                        trial[i] = np.clip(trial[i]+d*step[i], lower[i], upper[i])
                        f = evaluate(trial)
                        if f < current_fit:
                            current = trial; current_fit = f
    
    return best
