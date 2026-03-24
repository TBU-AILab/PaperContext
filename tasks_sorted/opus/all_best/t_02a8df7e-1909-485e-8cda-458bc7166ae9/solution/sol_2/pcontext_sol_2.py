#Looking at the previous results, I need to improve the optimization algorithm. Key issues I see:
#
#1. **CMA-ES eigendecomposition is expensive** for high dimensions - should use separable CMA-ES
#2. **Time management** needs improvement - too much time on initialization, not enough restarts
#3. **Need better restart strategy** - BIPOP-style with both small and large populations
#4. **Local search** should be more aggressive with multiple step sizes
#
#Let me create a cleaner, more efficient implementation:
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
    
    def remaining():
        return max_time * 0.97 - elapsed()
    
    evals = 0
    
    def evaluate(x):
        nonlocal best, best_params, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Sobol-like initialization with opposition-based learning
    n_init = min(max(15 * dim, 100), 800)
    
    pop = np.zeros((n_init, dim))
    fits = np.zeros(n_init)
    for i in range(n_init):
        if remaining() <= 0:
            return best
        pop[i] = lower + np.random.rand(dim) * ranges
        fits[i] = evaluate(pop[i])
    
    # Opposition-based candidates
    n_opp = min(n_init // 2, 200)
    sorted_idx = np.argsort(fits)
    for i in range(n_opp):
        if remaining() <= 0:
            return best
        opp = lower + upper - pop[sorted_idx[i]]
        evaluate(opp)

    # Phase 2: Separable CMA-ES with restarts
    def sep_cma_es(x0, sigma0, max_gen=None):
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
        damps = 1 + 2 * max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        diagC = np.ones(n)
        
        best_gen_fit = float('inf')
        stag = 0
        gen = 0
        if max_gen is None:
            max_gen = 100 + 150 * n // lam
        
        while gen < max_gen and remaining() > 0.5:
            sqrtC = np.sqrt(diagC)
            arz = np.random.randn(lam, n)
            arx = mean[None, :] + sigma * arz * sqrtC[None, :]
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.array([evaluate(arx[k]) for k in range(lam)])
            if remaining() <= 0:
                return
            
            idx = np.argsort(arfitness)
            if arfitness[idx[0]] < best_gen_fit - 1e-10:
                best_gen_fit = arfitness[idx[0]]
                stag = 0
            else:
                stag += 1
            
            old_mean = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            diff = (mean - old_mean) / sigma
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * diff / sqrtC
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*diff
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            diagC = (1-c1-cmu_val)*diagC + c1*(pc**2 + (1-hsig)*cc*(2-cc)*diagC) + cmu_val*(weights @ (artmp**2))
            diagC = np.clip(diagC, 1e-20, 1e6)
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.max(ranges))
            
            gen += 1
            if sigma < 1e-12 or stag > 15 + 30*n//lam:
                return
    
    restart = 0
    sorted_idx = np.argsort(fits)
    
    while remaining() > 1.0:
        if restart < 3:
            x0 = pop[sorted_idx[restart]].copy()
            sigma0 = 0.25 * np.mean(ranges)
        elif best_params is not None and np.random.rand() < 0.5:
            scale = 0.05 * (1 + restart * 0.3)
            x0 = best_params + scale * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = scale * np.mean(ranges)
        else:
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.3 * np.mean(ranges)
        
        sep_cma_es(x0, sigma0)
        restart += 1

    # Phase 3: Multi-scale local search
    if best_params is not None:
        current = best_params.copy()
        current_fit = best
        for scale in [0.01, 0.002, 0.0005, 0.0001]:
            step = scale * ranges
            for _ in range(5):
                improved = False
                for i in range(dim):
                    if remaining() <= 0:
                        return best
                    for d in [1, -1]:
                        trial = current.copy()
                        trial[i] = np.clip(trial[i] + d*step[i], lower[i], upper[i])
                        f = evaluate(trial)
                        if f < current_fit:
                            current = trial
                            current_fit = f
                            improved = True
                if not improved:
                    break
    
    return best
