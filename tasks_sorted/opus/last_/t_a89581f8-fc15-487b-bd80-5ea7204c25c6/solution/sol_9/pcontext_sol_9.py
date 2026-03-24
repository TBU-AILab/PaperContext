#Looking at the previous algorithm, I need to improve the optimization strategy. Key issues I see:
#
#1. **DE phase is too slow** - spending 28% of time on DE may be wasteful
#2. **CMA-ES restarts aren't diverse enough** - need better restart strategies
#3. **Missing multi-start local search** - for lower dimensions, local methods are very effective
#4. **No adaptive sigma** - sigma initialization should depend on the problem
#
#Let me implement a more sophisticated approach with better time allocation, SHADE (Success-History based Adaptive DE), improved CMA-ES with better restart logic, and a more aggressive local search phase.
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
    
    archive = []
    
    def record(x, f):
        archive.append((f, x.copy()))
        archive.sort(key=lambda t: t[0])
        if len(archive) > 80:
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
        
        lam = lam_override if lam_override else max(int(4 + 3 * np.log(n)), 6)
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2
        weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights_raw / weights_raw.sum()
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
        prev_median = float('inf')
        local_best = float('inf')
        local_best_x = None
        
        gen = 0
        while (elapsed() - t_start) < budget_time and time_left() > 0.15:
            gen += 1
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arfit = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.05:
                    return local_best, local_best_x
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
                local_best_x = arx[idx[0]].copy()
                record(arx[idx[0]], gb)
            
            med = np.median(arfit)
            
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
                        return local_best, local_best_x
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.max(ranges))
            
            if med < prev_median - 1e-12*(abs(prev_median)+1):
                stag = 0
            else:
                stag += 1
            prev_median = med
            
            if sigma < 1e-15:
                break
            if stag > 10 + 3*n:
                break
            if not sep and np.max(D) > 1e7 * np.min(D):
                break
        
        return local_best, local_best_x
    
    # ==================== SHADE ====================
    def run_shade(budget_time):
        t_start = elapsed()
        NP = min(max(8*dim, 30), 150)
        pop = lhs_sample(NP)
        fit = np.array([eval_f(p) for p in pop])
        for i in range(NP):
            record(pop[i], fit[i])
        
        H = 100
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        k = 0
        p_min = max(2/NP, 0.05)
        p_max = 0.2
        ext_archive = []
        
        gen = 0
        while (elapsed() - t_start) < budget_time and time_left() > 0.5:
            gen += 1
            SF = []
            SCR = []
            delta_f = []
            
            for i in range(NP):
                if time_left() < 0.1:
                    return
                
                ri = np.random.randint(H)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.randn() * 0.1 + MCR[ri], 0, 1)
                
                p = np.random.uniform(p_min, p_max)
                p_best_size = max(2, int(NP * p))
                p_best_idx = np.argsort(fit)[:p_best_size]
                pb = pop[np.random.choice(p_best_idx)]
                
                idxs = list(range(NP))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                pool_size = NP + len(ext_archive)
                r2_idx = np.random.randint(pool_size)
                while r2_idx == i or r2_idx == r1:
                    r2_idx = np.random.randint(pool_size)
                if r2_idx < NP:
                    xr2 = pop[r2_idx]
                else:
                    xr2 = ext_archive[r2_idx - NP]
                
                v = pop[i] + Fi * (pb - pop[i]) + Fi * (pop[r1] - xr2)
                
                mask = np.random.rand(dim) < CRi
                mask[np.random.randint(dim)] = True
                trial = np.where(mask, v, pop[i])
                
                # Bounce-back
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                trial = clip(trial)
                
                ftrial = eval_f(trial)
                if ftrial < fit[i]:
                    SF.append(Fi)
                    SCR.append(CRi)
                    delta_f.append(fit[i] - ftrial)
                    ext_archive.append(pop[i].copy())
                    pop[i] = trial
                    fit[i] = ftrial
                    record(trial, ftrial)
                elif ftrial == fit[i]:
                    pop[i] = trial
                    fit[i] = ftrial
            
            if len(ext_archive) > NP:
                np.random.shuffle(ext_archive)
                ext_archive = ext_archive[:NP]
            
            if len(SF) > 0:
                w = np.array(delta_f)
                w = w / (w.sum() + 1e-30)
                MF[k] = np.sum(w * np.array(SF)**2) / (np.sum(w * np.array(SF)) + 1e-30)
                MCR[k] = np.sum(w * np.array(SCR))
                k = (k + 1) % H
    
    # ==================== Nelder-Mead ====================
    def nelder_mead(x0, budget_time, initial_step=0.05):
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
            p[i] += initial_step * ranges[i] * (1 if np.random.rand() > 0.5 else -1)
            simplex[i+1] = clip(p)
            f_vals[i+1] = eval_f(simplex[i+1])
        
        for _ in range(10000):
            if (elapsed() - t_start) > budget_time or time_left() < 0.1:
                break
            
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
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
            
            if np.max(np.abs(f_vals[-1] - f_vals[0])) < 1e-15 * (abs(f_vals[0]) + 1):
                break
            spread = np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30))
            if spread < 1e-14:
                break
    
    # ==================== Powell-like coordinate descent ====================
    def coordinate_search(x0, budget_time):
        t_start = elapsed()
        x = clip(x0.copy())
        fx = eval_f(x)
        
        step = 0.1 * ranges.copy()
        
        for iteration in range(50):
            if (elapsed() - t_start) > budget_time or time_left() < 0.1:
                break
            improved = False
            for d in np.random.permutation(dim):
                if time_left() < 0.05:
                    return
                
                # Try golden-section-like exploration
                for sign in [1, -1]:
                    xt = x.copy()
                    xt[d] += sign * step[d]
                    xt = clip(xt)
                    ft = eval_f(xt)
                    if ft < fx:
                        # Accelerate in this direction
                        x, fx = xt, ft
                        improved = True
                        # Try doubling
                        for _ in range(5):
                            if time_left() < 0.05:
                                return
                            xt2 = x.copy()
                            xt2[d] += sign * step[d]
                            xt2 = clip(xt2)
                            ft2 = eval_f(xt2)
                            if ft2 < fx:
                                x, fx = xt2, ft2
                            else:
                                break
                        break
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-13:
                    break
    
    # ==================== MAIN STRATEGY ====================
    
    # Phase 1: Quick LHS sampling for initial exploration
    n_init = min(max(20, 5*dim), 200)
    init_pop = lhs_sample(n_init)
    for p in init_pop:
        if time_left() < max_time * 0.85:
            break
        f = eval_f(p)
        record(p, f)
    
    # Phase 2: SHADE (~25% time)
    shade_time = max_time * 0.22
    if shade_time > 1.0 and dim >= 5:
        run_shade(shade_time)
    elif shade_time > 0.5:
        run_shade(shade_time)
    
    # Phase 3: CMA-ES with BIPOP restarts (~45% time)
    restart = 0
    base_lam = max(int(4 + 3 * np.log(dim)), 6)
    
    while time_left() > 2.0:
        r = np.random.rand()
        
        if restart == 0 and best_pos is not None:
            x0 = best_pos + 0.01 * ranges * np.random.randn(dim)
            sig = 0.05 * np.mean(ranges)
            lam = base_lam
        elif restart < 3 and best_pos is not None:
            x0 = best_pos + 0.1 * ranges * np.random.randn(dim)
            sig = 0.15 * np.mean(ranges)
            lam = base_lam
        elif r < 0.3 and len(archive) > 3:
            idx = np.random.randint(min(10, len(archive)))
            x0 = archive[idx][1] + 0.1 * ranges * np.random.randn(dim)
            sig = 0.12 * np.mean(ranges)
            lam = base_lam
        elif r < 0.55 and best_pos is not None:
            x0 = best_pos + 0.3 * ranges * np.random.randn(dim)
            sig = 0.25 * np.mean(ranges)
            lam = base_lam
        elif r < 0.75:
            # Large population global
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.4 * np.mean(ranges)
            lam = min(base_lam * (2 ** min(restart // 2, 5)), 512)
        else:
            # Random restart
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.3 * np.mean(ranges)
            lam = base_lam * 2
        
        budget = min(max(1.0, time_left() * 0.22), time_left() - 1.5)
        if budget < 0.8:
            break
        
        _, _ = run_cmaes(clip(x0), sig, budget, lam_override=lam)
        restart += 1
    
    # Phase 4: Local refinement
    if time_left() > 1.0 and best_pos is not None:
        if dim <= 50:
            nelder_mead(best_pos, time_left() * 0.5, initial_step=0.01)
        else:
            coordinate_search(best_pos, time_left() * 0.5)
    
    # Phase 5: Very fine coordinate search
    if time_left() > 0.3 and best_pos is not None:
        coordinate_search(best_pos, time_left() * 0.8)
    
    # Phase 6: Final tiny perturbation search
    if time_left() > 0.15 and best_pos is not None:
        x = best_pos.copy()
        fx = best
        for scale in [0.001, 0.0001]:
            if time_left() < 0.08:
                break
            for _ in range(200):
                if time_left() < 0.05:
                    break
                pert = x + scale * ranges * np.random.randn(dim)
                pert = clip(pert)
                fp = eval_f(pert)
                if fp < fx:
                    x, fx = pert, fp
    
    return best
