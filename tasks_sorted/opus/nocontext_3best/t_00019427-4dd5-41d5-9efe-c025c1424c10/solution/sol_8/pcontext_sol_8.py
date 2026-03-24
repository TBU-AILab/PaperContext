#Looking at the results, algorithm 1 (SHADE + multi-scale Nelder-Mead) performed best at 4.25. I'll build on its strengths while adding key improvements:
#
#1. **CMA-ES as the main optimizer** - much better than DE for continuous optimization
#2. **Better initial sampling** with opposition-based learning
#3. **Adaptive restart strategy** from diverse elite solutions
#4. **Refined local search** with coordinate descent + Nelder-Mead
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
    mid = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Initial sampling with LHS + opposition ---
    n_init = min(max(40, 10 * dim), 300)
    population = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        population[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.05:
            break
        fitness[i] = evaluate(population[i])
        # Opposition-based point
        opp = lower + upper - population[i]
        evaluate(opp)

    # --- Phase 2: CMA-ES ---
    def run_cmaes(x0, sigma0, time_budget_frac):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        
        gen = 0
        while elapsed() < max_time * time_budget_frac:
            gen += 1
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fit = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= max_time * time_budget_frac:
                    return
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
                fit[k] = evaluate(arx[k])
            
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*gen)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff) * (B @ (D * zmean))
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1-c1-cmu_val)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if eval_count[0] - eigeneval > lam/(c1+cmu_val)/n/10:
                eigeneval = eval_count[0]
                C = np.triu(C) + np.triu(C,1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1/D) @ B.T
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                break

    # Run CMA-ES from best found point
    run_cmaes(best_params.copy(), 0.3 * np.max(ranges), 0.55)
    
    # Restart CMA-ES with smaller sigma
    if elapsed() < max_time * 0.7:
        run_cmaes(best_params.copy(), 0.05 * np.max(ranges), 0.75)
    
    # --- Phase 3: Nelder-Mead refinement ---
    def nelder_mead(start_point, scale_factor, time_lim):
        nonlocal best, best_params
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n+1, n))
        f_s = np.zeros(n+1)
        simplex[0] = start_point.copy()
        f_s[0] = evaluate(start_point)
        for i in range(n):
            if elapsed() >= time_lim: return
            simplex[i+1] = start_point.copy()
            simplex[i+1][i] += scale[i]
            f_s[i+1] = evaluate(simplex[i+1])
        for _ in range(10000):
            if elapsed() >= time_lim: return
            o = np.argsort(f_s); simplex=simplex[o]; f_s=f_s[o]
            c = np.mean(simplex[:-1],axis=0)
            xr=clip(c+1.0*(c-simplex[-1])); fr=evaluate(xr)
            if fr<f_s[0]:
                xe=clip(c+2.0*(xr-c)); fe=evaluate(xe)
                simplex[-1],f_s[-1]=(xe,fe) if fe<fr else (xr,fr)
            elif fr<f_s[-2]: simplex[-1],f_s[-1]=xr,fr
            else:
                xc=clip(c+0.5*((xr if fr<f_s[-1] else simplex[-1])-c)); fc=evaluate(xc)
                if fc<min(fr,f_s[-1]): simplex[-1],f_s[-1]=xc,fc
                else:
                    for j in range(1,n+1):
                        simplex[j]=clip(simplex[0]+0.5*(simplex[j]-simplex[0])); f_s[j]=evaluate(simplex[j])
                        if elapsed()>=time_lim: return
            if np.std(f_s)<1e-16: break

    nelder_mead(best_params, 0.02, max_time*0.85)
    nelder_mead(best_params, 0.005, max_time*0.93)
    nelder_mead(best_params, 0.001, max_time*0.98)
    
    # Final fine perturbation
    while elapsed() < max_time * 0.998:
        s = np.random.exponential(0.001)
        evaluate(best_params + np.random.randn(dim)*s*ranges)
    
    return best
#
#Key improvements:
#1. **CMA-ES** replaces SHADE - adapts a full covariance matrix for much better scaling on correlated landscapes
#2. **Opposition-based learning** in initialization doubles coverage
#3. **Two CMA-ES runs** - first broad exploration, then focused refinement
#4. **Progressive Nelder-Mead** at 3 decreasing scales for final polishing
#5. **Proper CMA-ES parameter tuning** with step-size control, cumulation paths, and eigendecomposition
