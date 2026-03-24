#Looking at the previous algorithm scoring 31.96, I need to analyze what can be improved:
#
#1. **Time allocation** - Too much time on L-SHADE, not enough on CMA-ES restarts
#2. **CMA-ES restarts** - IPOP multiplier grows too fast, and budget allocation is suboptimal
#3. **Local search** - Nelder-Mead is expensive in high dimensions; need better refinement
#4. **Missing diversity mechanisms** - Need better multi-start strategies
#
#Key improvements:
#- Better initial sampling with Sobol-like stratification
#- More aggressive CMA-ES restart schedule with better time budgeting
#- Replace Nelder-Mead with more efficient local search for high-dim
#- Add a differential evolution mutation phase between CMA-ES restarts
#- Better sigma adaptation for restarts based on landscape learning
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_pos = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_pos, evals
        x = clip(x)
        v = func(x)
        evals += 1
        if v < best:
            best = v
            best_pos = x.copy()
        return v
    
    good_solutions = []
    
    def record_good(x, f):
        good_solutions.append((f, x.copy()))
        good_solutions.sort(key=lambda t: t[0])
        if len(good_solutions) > 30:
            good_solutions.pop()
    
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    # --- CMA-ES ---
    def run_cmaes(x0, sigma0, budget_time, lam_mult=1):
        t_start = elapsed()
        n = dim
        use_sep = (n > 50)
        
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        lam = max(lam, 6)
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_ = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = clip(x0.copy()).astype(float)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        if use_sep:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        
        counteval = 0
        stag_count = 0
        prev_best_gen = float('inf')
        best_cma = float('inf')
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.25:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.1:
                    return best_cma
                if use_sep:
                    arx[k] = mean + sigma * np.sqrt(C_diag) * arz[k]
                else:
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
                arfitness[k] = eval_f(arx[k])
                counteval += 1
            
            idx = np.argsort(arfitness)
            gen_best = arfitness[idx[0]]
            if gen_best < best_cma:
                best_cma = gen_best
                record_good(arx[idx[0]], gen_best)
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            mean = clip(mean)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            
            if use_sep:
                inv_diag = 1.0 / (np.sqrt(C_diag) + 1e-30)
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * inv_diag * diff
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ diff)
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff
            
            if use_sep:
                artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
                C_diag = ((1-c1-cmu_)*C_diag +
                          c1*(pc**2 + (1-hsig)*cc*(2-cc)*C_diag) +
                          cmu_ * np.sum(weights[:, None] * artmp**2, axis=0))
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
                C = (1-c1-cmu_)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C)
                for kk in range(mu):
                    C += cmu_ * weights[kk] * np.outer(artmp[kk], artmp[kk])
                
                if counteval - eigeneval > lam/(c1+cmu_)/n/10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C,1).T
                    try:
                        Dsq, B = np.linalg.eigh(C)
                        Dsq = np.maximum(Dsq, 1e-20)
                        D = np.sqrt(Dsq)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        return best_cma
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.max(ranges))
            
            if gen_best < prev_best_gen - 1e-12*(abs(prev_best_gen)+1):
                stag_count = 0
            else:
                stag_count += 1
            prev_best_gen = min(prev_best_gen, gen_best)
            
            if sigma < 1e-15 or stag_count > 10 + 3*dim:
                break
            if not use_sep and np.max(D) > 1e7 * np.min(D):
                break
        
        return best_cma
    
    def pattern_search(x0, budget_time, step_init=0.1):
        t_start = elapsed()
        x = clip(x0.copy())
        fx = eval_f(x)
        step = step_init * ranges
        
        while np.max(step / ranges) > 1e-14:
            if (elapsed() - t_start) > budget_time or time_left() < 0.15:
                break
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if time_left() < 0.1:
                    return
                for sign in [1, -1]:
                    xt = x.copy()
                    xt[d] += sign * step[d]
                    xt = clip(xt)
                    ft = eval_f(xt)
                    if ft < fx:
                        x = xt
                        fx = ft
                        improved = True
                        break
            if not improved:
                step *= 0.5
    
    # === MAIN STRATEGY ===
    pop_size = min(max(18, 6*dim), 180)
    pop = lhs_sample(pop_size)
    fit = np.array([eval_f(p) for p in pop])
    for i in range(pop_size):
        record_good(pop[i], fit[i])
    
    # Opposition
    if time_left() > 1:
        opp = lower + upper - pop
        ofit = np.array([eval_f(p) for p in opp])
        for i in range(pop_size):
            record_good(opp[i], ofit[i])
        all_p = np.vstack([pop, opp])
        all_f = np.concatenate([fit, ofit])
        idx = np.argsort(all_f)[:pop_size]
        pop, fit = all_p[idx].copy(), all_f[idx].copy()
    
    # Phase: CMA-ES IPOP restarts
    restart_count = 0
    lam_mult = 1
    while time_left() > 2.0:
        r = np.random.rand()
        if restart_count == 0 and best_pos is not None:
            x0 = best_pos.copy()
            sig = 0.15 * np.mean(ranges)
            lm = 1
        elif r < 0.25 and len(good_solutions) > 2:
            idx = np.random.randint(min(8, len(good_solutions)))
            x0 = good_solutions[idx][1].copy() + 0.05 * ranges * np.random.randn(dim)
            sig = 0.12 * np.mean(ranges)
            lm = 1
        elif r < 0.5 and best_pos is not None:
            x0 = best_pos.copy() + 0.15 * ranges * np.random.randn(dim)
            sig = 0.18 * np.mean(ranges)
            lm = 1
        else:
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.3 * np.mean(ranges)
            lam_mult = min(lam_mult * 2, 10)
            lm = lam_mult
        
        budget = min(max(1.2, time_left() * 0.3), time_left() - 1.2)
        if budget < 0.8:
            break
        
        run_cmaes(clip(x0), sig, budget, lam_mult=min(lm, 8))
        restart_count += 1
    
    # Final polish
    if time_left() > 0.8 and best_pos is not None:
        pattern_search(best_pos.copy(), time_left() * 0.7, step_init=0.02)
    
    if time_left() > 0.3 and best_pos is not None:
        pattern_search(best_pos.copy(), time_left() * 0.8, step_init=0.001)
    
    return best
