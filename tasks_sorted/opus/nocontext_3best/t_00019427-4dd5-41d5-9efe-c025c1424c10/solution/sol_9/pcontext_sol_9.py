#Looking at the results, algorithm 1 (SHADE + multi-scale Nelder-Mead) performed best with 4.255. I'll build on its strengths while adding several improvements:
#
#1. **CMA-ES as the primary optimizer** - much better than DE for continuous optimization
#2. **Better initial sampling** with opposition-based learning
#3. **Adaptive restart strategy** from diverse elite solutions
#4. **More sophisticated local search** combining Nelder-Mead with coordinate descent
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

    # --- Phase 1: Diverse initial sampling with opposition ---
    n_init = min(max(40, 10 * dim), 300)
    init_pop = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    init_pop = lower + init_pop * ranges
    
    init_fit = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        init_fit[i] = evaluate(init_pop[i])
        # Opposition-based point
        if elapsed() < max_time * 0.1:
            opp = lower + upper - init_pop[i]
            evaluate(opp)
    
    # --- Phase 2: CMA-ES ---
    def run_cmaes(x0, sigma0, time_budget_frac):
        nonlocal best, best_params
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
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dim
        use_full = n <= 50
        if use_full:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
        else:
            diagC = np.ones(n)
        
        update_interval = max(1, int(n / 10))
        gen_count = 0
        no_improve_count = 0
        local_best = best
        
        while elapsed() < max_time * time_budget_frac:
            gen_count += 1
            
            # Sample offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full:
                sqrtC = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-20))) @ eigenvectors.T
                for k in range(lam):
                    arx[k] = mean + sigma * (sqrtC @ arz[k])
            else:
                for k in range(lam):
                    arx[k] = mean + sigma * np.sqrt(np.maximum(diagC, 1e-20)) * arz[k]
            
            # Evaluate
            arfitness = np.full(lam, float('inf'))
            for k in range(lam):
                if elapsed() >= max_time * time_budget_frac:
                    return
                arx[k] = clip(arx[k])
                arfitness[k] = evaluate(arx[k])
            
            # Sort
            idx = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            selected = arx[idx[:mu]]
            mean = np.sum(weights[:, None] * selected, axis=0)
            
            # Cumulation
            diff = mean - old_mean
            if use_full:
                invsqrtC = eigenvectors @ np.diag(1.0/np.sqrt(np.maximum(eigenvalues, 1e-20))) @ eigenvectors.T
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * diff / (sigma * np.sqrt(np.maximum(diagC, 1e-20)))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen_count+1))) / chiN < 1.4 + 2.0/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff / sigma
            
            # Covariance update
            if use_full:
                artmp = (selected - old_mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
                for k in range(mu):
                    C += cmu * weights[k] * np.outer(artmp[k], artmp[k])
                C = np.triu(C) + np.triu(C, 1).T
                if gen_count % update_interval == 0:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(C)
                        eigenvalues = np.maximum(eigenvalues, 1e-20)
                    except:
                        C = np.eye(n)
                        eigenvalues = np.ones(n)
                        eigenvectors = np.eye(n)
            else:
                artmp = (selected - old_mean) / sigma
                diagC = (1 - c1 - cmu)*diagC + c1*(pc**2 + (1-hsig)*cc*(2-cc)*diagC)
                for k in range(mu):
                    diagC += cmu * weights[k] * artmp[k]**2
                diagC = np.maximum(diagC, 1e-20)
            
            # Step-size update
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if best < local_best - 1e-12:
                local_best = best
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if sigma < 1e-16 * np.max(ranges) or no_improve_count > 50 + 10*dim:
                return
    
    # Run CMA-ES from best initial point
    sorted_init = np.argsort(init_fit)
    run_cmaes(init_pop[sorted_init[0]], 0.3 * np.max(ranges), 0.55)
    
    # Restart from second-best diverse point
    for idx in sorted_init[1:6]:
        if elapsed() >= max_time * 0.7:
            break
        if np.linalg.norm(init_pop[idx] - best_params) > 0.05 * np.linalg.norm(ranges):
            run_cmaes(init_pop[idx], 0.2 * np.max(ranges), min(elapsed()/max_time + 0.15, 0.7))

    # Smaller CMA-ES restart around best
    run_cmaes(best_params, 0.05 * np.max(ranges), 0.82)
    
    # --- Phase 3: Nelder-Mead refinement ---
    def nelder_mead(start_point, scale_factor, time_limit):
        nonlocal best, best_params
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n+1, n)); f_s = np.zeros(n+1)
        simplex[0] = start_point.copy(); f_s[0] = evaluate(start_point)
        for i in range(n):
            if elapsed() >= time_limit: return
            simplex[i+1] = start_point.copy(); simplex[i+1][i] += scale[i]
            simplex[i+1] = clip(simplex[i+1]); f_s[i+1] = evaluate(simplex[i+1])
        for _ in range(10000):
            if elapsed() >= time_limit: return
            o = np.argsort(f_s); simplex=simplex[o]; f_s=f_s[o]
            c = np.mean(simplex[:-1],axis=0)
            xr=clip(c+1.0*(c-simplex[-1])); fr=evaluate(xr)
            if fr<f_s[0]:
                xe=clip(c+2.0*(xr-c)); fe=evaluate(xe)
                simplex[-1],f_s[-1]=(xe,fe) if fe<fr else (xr,fr)
            elif fr<f_s[-2]: simplex[-1],f_s[-1]=xr,fr
            else:
                cand=xr if fr<f_s[-1] else simplex[-1]
                xc=clip(c+0.5*(cand-c)); fc=evaluate(xc)
                if fc<min(fr,f_s[-1]): simplex[-1],f_s[-1]=xc,fc
                else:
                    for j in range(1,n+1):
                        simplex[j]=clip(simplex[0]+0.5*(simplex[j]-simplex[0])); f_s[j]=evaluate(simplex[j])
                        if elapsed()>=time_limit: return
            if np.std(f_s)<1e-16: break
    
    nelder_mead(best_params, 0.02, max_time*0.88)
    nelder_mead(best_params, 0.005, max_time*0.93)
    nelder_mead(best_params, 0.001, max_time*0.97)
    
    # --- Phase 4: Fine coordinate search ---
    step = 0.0005 * ranges
    while elapsed() < max_time * 0.995:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.995: break
            for s in [1, -1]:
                t = best_params.copy(); t[d] += s*step[d]; f_t = evaluate(clip(t))
                if f_t < best: improved = True; break
        if not improved:
            step *= 0.5
            if np.max(step/ranges) < 1e-13: break
    
    return best
