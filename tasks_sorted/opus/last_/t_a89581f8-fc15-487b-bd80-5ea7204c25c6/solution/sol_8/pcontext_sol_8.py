#Looking at the previous algorithm scoring 75.99, I need to significantly improve the optimization. Let me analyze the issues:
#
#1. **CMA-ES budget allocation is wasteful** - spending too much time on bad restarts
#2. **IPOP multiplier grows too fast** - wastes evaluations with huge populations
#3. **Missing DE phase** - differential evolution is good for global exploration
#4. **Local search is too simple** - pattern search is slow in high dimensions
#5. **Need better time management** - more restarts with smaller budgets
#
#I'll implement a hybrid: DE for global exploration → CMA-ES restarts with BIPOP → efficient local refinement.
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_pos
        x = clip(x)
        v = func(x)
        if v < best:
            best = v
            best_pos = x.copy()
        return v
    
    archive = []
    
    def record(x, f):
        archive.append((f, x.copy()))
        archive.sort(key=lambda t: t[0])
        if len(archive) > 50:
            archive.pop()
    
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    # ==================== CMA-ES ====================
    def run_cmaes(x0, sigma0, budget_time, lam_override=None):
        t_start = elapsed()
        n = dim
        sep = (n > 80)
        
        lam = lam_override if lam_override else int(4 + 3 * np.log(n))
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
        
        if sep:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        
        counteval = 0
        stag = 0
        prev_best = float('inf')
        local_best = float('inf')
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.2:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arfit = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.08:
                    return local_best
                if sep:
                    arx[k] = mean + sigma * np.sqrt(C_diag) * arz[k]
                else:
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
                arfit[k] = eval_f(arx[k])
                counteval += 1
            
            idx = np.argsort(arfit)
            gb = arfit[idx[0]]
            if gb < local_best:
                local_best = gb
                record(arx[idx[0]], gb)
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            mean = clip(mean)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            
            if sep:
                inv_d = 1.0 / (np.sqrt(C_diag) + 1e-30)
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * inv_d * diff
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ diff)
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff
            
            if sep:
                artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
                C_diag = ((1-c1-cmu_)*C_diag +
                          c1*(pc**2 + (1-hsig)*cc*(2-cc)*C_diag) +
                          cmu_ * np.sum(weights[:, None] * artmp**2, axis=0))
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
                C = (1-c1-cmu_)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
                for kk in range(mu):
                    C += cmu_ * weights[kk] * np.outer(artmp[kk], artmp[kk])
                
                if counteval - eigeneval > lam/(c1+cmu_)/n/10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        Dsq, B = np.linalg.eigh(C)
                        Dsq = np.maximum(Dsq, 1e-20)
                        D = np.sqrt(Dsq)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        return local_best
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.max(ranges))
            
            if gb < prev_best - 1e-12*(abs(prev_best)+1):
                stag = 0
            else:
                stag += 1
            prev_best = min(prev_best, gb)
            
            if sigma < 1e-15 or stag > 10 + 3*n:
                break
            if not sep and np.max(D) > 1e7 * np.min(D):
                break
        
        return local_best
    
    # ==================== DE/current-to-pbest/1 ====================
    def run_de(budget_time):
        t_start = elapsed()
        NP = min(max(10*dim, 40), 200)
        pop = lhs_sample(NP)
        fit = np.array([eval_f(p) for p in pop])
        for i in range(NP):
            record(pop[i], fit[i])
        
        F = 0.5
        CR = 0.9
        p_best_rate = 0.1
        ext_archive = []
        
        gen = 0
        while (elapsed() - t_start) < budget_time and time_left() > 0.5:
            gen += 1
            # Adapt F and CR
            for i in range(NP):
                if time_left() < 0.15:
                    return
                
                # p-best
                p_best_size = max(2, int(NP * p_best_rate))
                p_best_idx = np.argsort(fit)[:p_best_size]
                pb = pop[np.random.choice(p_best_idx)]
                
                # Mutation: DE/current-to-pbest/1
                idxs = list(range(NP))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from pop + archive
                pool_size = NP + len(ext_archive)
                r2_idx = np.random.randint(pool_size)
                if r2_idx < NP:
                    xr2 = pop[r2_idx]
                else:
                    xr2 = ext_archive[r2_idx - NP]
                
                Fi = np.clip(np.random.standard_cauchy() * 0.1 + F, 0, 1)
                CRi = np.clip(np.random.randn() * 0.1 + CR, 0, 1)
                
                v = pop[i] + Fi * (pb - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Binomial crossover
                mask = np.random.rand(dim) < CRi
                mask[np.random.randint(dim)] = True
                trial = np.where(mask, v, pop[i])
                trial = clip(trial)
                
                ftrial = eval_f(trial)
                if ftrial <= fit[i]:
                    ext_archive.append(pop[i].copy())
                    pop[i] = trial
                    fit[i] = ftrial
                    record(trial, ftrial)
                
                if len(ext_archive) > NP:
                    ext_archive.pop(np.random.randint(len(ext_archive)))
    
    # ==================== Nelder-Mead simplex ====================
    def nelder_mead(x0, budget_time, step=0.05):
        t_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = clip(x0.copy())
        f_vals = np.zeros(n+1)
        f_vals[0] = eval_f(simplex[0])
        
        for i in range(n):
            if time_left() < 0.1:
                return
            p = x0.copy()
            p[i] += step * ranges[i]
            simplex[i+1] = clip(p)
            f_vals[i+1] = eval_f(simplex[i+1])
        
        for _ in range(5000):
            if (elapsed() - t_start) > budget_time or time_left() < 0.1:
                break
            
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < f_vals[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], f_vals[-1] = xe, fe
                else:
                    simplex[-1], f_vals[-1] = xr, fr
            elif fr < f_vals[-2]:
                simplex[-1], f_vals[-1] = xr, fr
            else:
                if fr < f_vals[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = eval_f(xc)
                if fc < min(fr, f_vals[-1]):
                    simplex[-1], f_vals[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        if time_left() < 0.1:
                            return
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_vals[i] = eval_f(simplex[i])
            
            # Convergence
            if np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30)) < 1e-14:
                break
    
    # ==================== MAIN STRATEGY ====================
    
    # Phase 1: DE exploration (~30% time)
    de_time = max_time * 0.28
    if de_time > 1.0:
        run_de(de_time)
    
    # Phase 2: CMA-ES BIPOP restarts (~55% time)
    restart = 0
    small_budget_base = max(0.8, max_time * 0.06)
    large_lam = int(4 + 3 * np.log(dim))
    
    while time_left() > 1.5:
        r = np.random.rand()
        
        if restart < 2 and best_pos is not None:
            # Small sigma around best
            x0 = best_pos + 0.05 * ranges * np.random.randn(dim)
            sig = 0.1 * np.mean(ranges)
            lam = None
        elif r < 0.35 and len(archive) > 3:
            # Near good solution
            idx = np.random.randint(min(10, len(archive)))
            x0 = archive[idx][1] + 0.08 * ranges * np.random.randn(dim)
            sig = 0.12 * np.mean(ranges)
            lam = None
        elif r < 0.6 and best_pos is not None:
            x0 = best_pos + 0.2 * ranges * np.random.randn(dim)
            sig = 0.2 * np.mean(ranges)
            lam = None
        else:
            # Global restart with larger population
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.35 * np.mean(ranges)
            lam = min(large_lam * (2 ** min(restart // 3, 4)), 256)
        
        budget = min(max(small_budget_base, time_left() * 0.25), time_left() - 1.0)
        if budget < 0.6:
            break
        
        run_cmaes(clip(x0), sig, budget, lam_override=lam)
        restart += 1
    
    # Phase 3: Local refinement with Nelder-Mead
    if time_left() > 0.8 and best_pos is not None and dim <= 40:
        nelder_mead(best_pos, time_left() * 0.7, step=0.01)
    
    # Phase 4: Final coordinate-wise search
    if time_left() > 0.3 and best_pos is not None:
        x = best_pos.copy()
        fx = best
        step = 0.005 * ranges
        for _ in range(3):
            if time_left() < 0.15:
                break
            for d in np.random.permutation(dim):
                if time_left() < 0.1:
                    break
                for sign in [1, -1]:
                    xt = x.copy()
                    xt[d] += sign * step[d]
                    xt = clip(xt)
                    ft = eval_f(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        break
            step *= 0.5
    
    return best
