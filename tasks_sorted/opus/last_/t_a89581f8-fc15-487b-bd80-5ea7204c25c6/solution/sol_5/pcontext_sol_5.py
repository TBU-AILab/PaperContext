#Looking at the previous algorithm, I need to analyze potential weaknesses:
#
#1. **Time allocation** could be better - DE might consume too much time
#2. **CMA-ES restarts** need better strategy (BIPOP-like with small/large population alternation)
#3. **Local search** could use coordinate-wise refinement
#4. **Population sizing** and adaptation could be improved
#5. **Need better exploitation** near the best found solution
#
#Key changes:
#- Better time allocation with adaptive phases
#- Add coordinate descent as local search
#- Improve CMA-ES with better restart strategy
#- Add more aggressive local refinement near end
#- Better DE with linear population size reduction (L-SHADE style)
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_pos = None
    evals = 0
    
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
        nonlocal best, best_pos, evals
        x = clip(x)
        v = func(x)
        evals += 1
        if v < best:
            best = v
            best_pos = x.copy()
        return v
    
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    # --- L-SHADE DE ---
    def run_lshade(pop, fit, budget_time):
        t_start = elapsed()
        pop_size = len(pop)
        N_init = pop_size
        N_min = max(4, dim)
        H = 6
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        max_evals_est = int(budget_time * 500)  # rough estimate
        total_evals = 0
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            S_F, S_CR, S_df = [], [], []
            sorted_idx = np.argsort(fit[:pop_size])
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            
            for i in range(pop_size):
                if time_left() < 0.2:
                    return new_pop, new_fit
                
                ri = np.random.randint(H)
                if M_F[ri] <= 0:
                    Fi = 0.01
                else:
                    Fi = -1
                    while Fi <= 0:
                        Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    Fi = min(Fi, 1.0)
                
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                p = max(2, int(np.clip(0.05 + 0.15 * np.random.rand(), 0.05, 0.2) * pop_size))
                pbest_idx = sorted_idx[np.random.randint(p)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                pool_size = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                trial = clip(trial)
                
                f_trial = eval_f(trial)
                total_evals += 1
                
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_df.append(abs(fit[i] - f_trial))
                        if len(archive) < N_init:
                            archive.append(pop[i].copy())
                        elif N_init > 0:
                            archive[np.random.randint(len(archive))] = pop[i].copy()
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop[:pop_size] = new_pop
            fit[:pop_size] = new_fit
            
            if len(S_F) > 0:
                S_df_arr = np.array(S_df)
                w = S_df_arr / (S_df_arr.sum() + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)
                M_F[k] = np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
                if np.max(S_CR_arr) == 0:
                    M_CR[k] = -1
                else:
                    M_CR[k] = np.sum(w * S_CR_arr)
                k = (k + 1) % H
            
            # Linear pop reduction
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * total_evals / max(1, max_evals_est))))
            if new_size < pop_size:
                sidx = np.argsort(fit[:pop_size])
                pop[:new_size] = pop[sidx[:new_size]]
                fit[:new_size] = fit[sidx[:new_size]]
                pop_size = new_size
                while len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
        
        return pop[:pop_size], fit[:pop_size]
    
    # --- CMA-ES ---
    def run_cmaes(x0, sigma0, budget_time, lam_mult=1):
        t_start = elapsed()
        n = dim
        use_sep = (n > 80)
        
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        lam = max(lam, 6)
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
        
        mean = clip(x0.copy())
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
        flat_count = 0
        prev_median = float('inf')
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.2:
                    return
                if use_sep:
                    arx[k] = mean + sigma * np.sqrt(C_diag) * arz[k]
                else:
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
                arfitness[k] = eval_f(arx[k])
                counteval += 1
            
            idx = np.argsort(arfitness)
            
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
                        return
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.max(ranges))
            
            med_f = np.median(arfitness)
            if abs(med_f - prev_median) < 1e-12 * (abs(prev_median) + 1e-30):
                flat_count += 1
            else:
                flat_count = 0
            prev_median = med_f
            
            if sigma < 1e-15 or flat_count > 20 + 5*dim:
                break
            if not use_sep and np.max(D) > 1e7 * np.min(D):
                break
    
    # --- Coordinate descent with golden section ---
    def coord_descent(x0, budget_time, scale=0.1):
        t_start = elapsed()
        x = clip(x0.copy())
        fx = eval_f(x)
        gr = 0.381966011250105  # 2 - golden_ratio
        
        for iteration in range(50):
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                    return
                
                a = max(lower[d], x[d] - scale * ranges[d])
                b = min(upper[d], x[d] + scale * ranges[d])
                
                if b - a < 1e-15 * ranges[d]:
                    continue
                
                c = a + gr * (b - a)
                dd_ = b - gr * (b - a)
                
                xc = x.copy(); xc[d] = c; fc = eval_f(xc)
                xd = x.copy(); xd[d] = dd_; fd = eval_f(xd)
                
                for _ in range(12):
                    if b - a < 1e-10 * ranges[d]:
                        break
                    if time_left() < 0.2:
                        return
                    if fc < fd:
                        b = dd_
                        dd_ = c
                        fd = fc
                        c = a + gr * (b - a)
                        xc = x.copy(); xc[d] = c; fc = eval_f(xc)
                    else:
                        a = c
                        c = dd_
                        fc = fd
                        dd_ = b - gr * (b - a)
                        xd = x.copy(); xd[d] = dd_; fd = eval_f(xd)
                
                best_d = (a + b) / 2
                xt = x.copy(); xt[d] = best_d; ft = eval_f(xt)
                if ft < fx:
                    x = xt; fx = ft; improved = True
                if fc < fx:
                    x[d] = c; fx = fc; improved = True
                if fd < fx:
                    x[d] = dd_; fx = fd; improved = True
            
            if not improved:
                scale *= 0.5
                if scale < 1e-12:
                    break
    
    # --- Nelder-Mead ---
    def nelder_mead(x0, budget_time, scale=0.05):
        t_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = clip(x0.copy())
        for i in range(n):
            simplex[i+1] = x0.copy()
            step = scale * ranges[i]
            simplex[i+1][i] += step if np.random.rand() > 0.5 else -step
            simplex[i+1] = clip(simplex[i+1])
        
        f_vals = np.array([eval_f(s) for s in simplex])
        
        for _ in range(10000):
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + alpha*(centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < f_vals[0]:
                xe = clip(centroid + gamma*(xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], f_vals[-1] = xe, fe
                else:
                    simplex[-1], f_vals[-1] = xr, fr
            elif fr < f_vals[-2]:
                simplex[-1], f_vals[-1] = xr, fr
            else:
                if fr < f_vals[-1]:
                    xc = clip(centroid + rho*(xr - centroid))
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1], f_vals[-1] = xc, fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
                else:
                    xc = clip(centroid + rho*(simplex[-1] - centroid))
                    fc = eval_f(xc)
                    if fc < f_vals[-1]:
                        simplex[-1], f_vals[-1] = xc, fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30)) < 1e-14:
                break
    
    # === MAIN STRATEGY ===
    pop_size = min(max(30, 10*dim), 300)
    
    # Phase 1: LHS sampling + opposition
    pop = lhs_sample(pop_size)
    fit = np.array([eval_f(p) for p in pop])
    
    if time_left() > 1:
        opp = lower + upper - pop
        ofit = np.array([eval_f(p) for p in opp])
        all_p = np.vstack([pop, opp])
        all_f = np.concatenate([fit, ofit])
        idx = np.argsort(all_f)[:pop_size]
        pop, fit = all_p[idx].copy(), all_f[idx].copy()
    
    # Phase 2: L-SHADE
    if time_left() > 2:
        pop, fit = run_lshade(pop, fit, time_left() * 0.30)
    
    # Phase 3: CMA-ES with BIPOP-like restarts
    restart_count = 0
    large_budget_used = 0
    while time_left() > 2.0:
        if restart_count == 0 and best_pos is not None:
            x0 = best_pos.copy() + 0.001 * ranges * np.random.randn(dim)
            sig = 0.15 * np.mean(ranges)
            lam_m = 1
        elif restart_count % 3 == 0:
            # Large population restart
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.4 * np.mean(ranges)
            lam_m = 2 + restart_count // 3
        else:
            # Small restart near best or random
            if best_pos is not None and np.random.rand() < 0.5:
                x0 = best_pos + 0.3 * ranges * np.random.randn(dim)
                sig = 0.1 * np.mean(ranges)
            else:
                x0 = lower + np.random.rand(dim) * ranges
                sig = 0.3 * np.mean(ranges)
            lam_m = 1
        
        budget = min(time_left() * 0.35, max(2.0, time_left() - 1.5))
        run_cmaes(clip(x0), sig, budget, lam_mult=min(lam_m, 4))
        restart_count += 1
    
    # Phase 4: Local refinement
    if time_left() > 1.0 and best_pos is not None:
        coord_descent(best_pos.copy(), time_left() * 0.45, scale=0.05)
    
    if time_left() > 0.5 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.8, scale=0.005)
    
    # Final coordinate polish
    if time_left() > 0.3 and best_pos is not None:
        coord_descent(best_pos.copy(), time_left() * 0.8, scale=0.001)
    
    return best
