#Looking at the previous results, I need to improve on the best score of 22.09. Let me analyze the weaknesses:
#
#1. The SHADE-DE is good but could benefit from **multiple restarts with smaller populations**
#2. **CMA-ES** would be more effective for local search than coordinate descent
#3. Need better **exploration-exploitation balance** and **multi-strategy approach**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = [0]
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Simplified CMA-ES
    def cma_es(x0, sigma0, budget_frac):
        nonlocal best, best_x
        time_limit = elapsed() + max_time * budget_frac
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for efficiency
        C_diag = np.ones(n)
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen = 0
        while elapsed() < time_limit:
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (np.sqrt(C_diag) * arz[k])
                arx[k] = clip(arx[k])
            
            # Evaluate
            fit = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= time_limit:
                    return
                fit[k] = evaluate(arx[k])
            
            # Sort
            idx = np.argsort(fit)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            
            # Update evolution paths
            zm = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (zm / np.sqrt(C_diag))
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * zm
            
            # Update diagonal covariance
            C_diag = (1 - c1 - cmu) * C_diag + c1 * (pc**2 + (1-hsig)*cc*(2-cc)*C_diag)
            for k in range(mu):
                diff = (arx[idx[k]] - old_mean) / sigma
                C_diag += cmu * weights[k] * diff**2
            
            C_diag = np.maximum(C_diag, 1e-20)
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            gen += 1
            if sigma < 1e-16:
                break

    # Phase 1: Multiple SHADE-DE restarts
    n_restarts = 0
    while elapsed() < max_time * 0.45:
        pop_size = min(max(20, 6 * dim), 200)
        pop = lower + np.random.random((pop_size, dim)) * ranges
        fit = np.array([evaluate(pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.44] + [float('inf')] * max(0, pop_size))[:pop_size]
        
        H = 50
        M_F, M_CR = np.full(H, 0.5), np.full(H, 0.5)
        k = 0
        archive = []
        
        for gen in range(500):
            if elapsed() >= max_time * 0.45:
                break
            sorted_idx = np.argsort(fit)
            for i in range(pop_size):
                if elapsed() >= max_time * 0.44:
                    break
                ri = np.random.randint(H)
                F_i = np.clip(M_F[ri] + 0.1*np.random.standard_cauchy(), 0.01, 1.0)
                CR_i = np.clip(M_CR[ri] + 0.1*np.random.randn(), 0, 1)
                p = max(2, int(0.15*pop_size))
                pb = sorted_idx[np.random.randint(p)]
                cands = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(cands)
                r2 = np.random.choice([j for j in cands if j != r1]) if len(cands)>1 else r1
                mutant = pop[i] + F_i*(pop[pb]-pop[i]) + F_i*(pop[r1]-pop[r2])
                mask = np.random.random(dim)<CR_i; mask[np.random.randint(dim)]=True
                trial = np.where(mask, mutant, pop[i])
                trial = clip(trial)
                ft = evaluate(trial)
                if ft <= fit[i]:
                    if ft < fit[i]:
                        M_F[k%H] = F_i; M_CR[k%H] = CR_i; k+=1
                    pop[i] = trial; fit[i] = ft
        n_restarts += 1

    # Phase 2: CMA-ES from best found
    if best_x is not None:
        for restart in range(5):
            if elapsed() >= max_time * 0.92:
                break
            sigma0 = 0.1 * np.mean(ranges) / (restart + 1)
            x0 = best_x + 0.01 * ranges * np.random.randn(dim) * (1.0/(restart+1))
            x0 = clip(x0)
            budget = min(0.12, (max_time*0.93 - elapsed())/max_time)
            if budget > 0.01:
                cma_es(x0, sigma0, budget)

    # Phase 3: Fine coordinate descent
    if best_x is not None:
        x, fx = best_x.copy(), best
        step = 0.001 * ranges
        while elapsed() < max_time * 0.97:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.97: return best
                for sign in [1, -1]:
                    xt = x.copy(); xt[d] = np.clip(x[d]+sign*step[d], lower[d], upper[d])
                    ft = evaluate(xt)
                    if ft < fx:
                        x, fx = xt, ft; step[d]*=1.3; improved=True; break
                else:
                    step[d]*=0.5
            if not improved or np.max(step)<1e-16:
                step = 0.0005*ranges*np.random.random(dim)
    
    return best
