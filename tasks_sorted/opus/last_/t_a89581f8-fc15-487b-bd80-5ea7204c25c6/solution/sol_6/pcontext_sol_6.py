#Looking at the previous algorithm, I need to identify weaknesses and improve. The score of 192.16 suggests the algorithm isn't converging well enough. Key issues I see:
#
#1. **Time allocation is suboptimal** - too much time on initial L-SHADE, not enough on CMA-ES restarts
#2. **CMA-ES restart strategy** needs improvement - should use IPOP with increasing populations
#3. **Need multi-start diversity** - better exploration of the search space
#4. **Local search could be more efficient** - use Powell-like pattern search
#5. **Missing: adaptive restart from diverse good solutions**
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
    
    # Keep track of good solutions for restarts
    good_solutions = []
    
    def record_good(x, f):
        good_solutions.append((f, x.copy()))
        good_solutions.sort(key=lambda t: t[0])
        if len(good_solutions) > 20:
            good_solutions.pop()
    
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    # --- CMA-ES with sep-CMA for high dim ---
    def run_cmaes(x0, sigma0, budget_time, lam_mult=1):
        t_start = elapsed()
        n = dim
        use_sep = (n > 60)
        
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
        flat_count = 0
        prev_best_gen = float('inf')
        stag_count = 0
        best_cma = float('inf')
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.15:
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
            
            # Stagnation detection
            if gen_best < prev_best_gen - 1e-12*(abs(prev_best_gen)+1):
                stag_count = 0
            else:
                stag_count += 1
            prev_best_gen = min(prev_best_gen, gen_best)
            
            if sigma < 1e-15 or stag_count > 15 + 3*dim:
                break
            if not use_sep and np.max(D) > 1e7 * np.min(D):
                break
        
        return best_cma
    
    # --- L-SHADE ---
    def run_lshade(pop, fit, budget_time):
        t_start = elapsed()
        pop_size = len(pop)
        N_init = pop_size
        N_min = max(4, dim // 2)
        H = 6
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        gen = 0
        max_gen_est = max(1, int(budget_time * 300 / pop_size))
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            sorted_idx = np.argsort(fit[:pop_size])
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            
            for i in range(pop_size):
                if time_left() < 0.2:
                    return new_pop[:pop_size], new_fit[:pop_size]
                
                ri = np.random.randint(H)
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                p = max(2, int(0.11 * pop_size))
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
                    record_good(trial, f_trial)
            
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
            
            new_size = max(N_min, int(round(N_init - (N_init - N_min) * gen / max(1, max_gen_est))))
            if new_size < pop_size:
                sidx = np.argsort(fit[:pop_size])
                pop[:new_size] = pop[sidx[:new_size]]
                fit[:new_size] = fit[sidx[:new_size]]
                pop_size = new_size
                while len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
        
        return pop[:pop_size], fit[:pop_size]
    
    # --- Pattern search (Hooke-Jeeves) ---
    def pattern_search(x0, budget_time, step_init=0.1):
        t_start = elapsed()
        x = clip(x0.copy())
        fx = eval_f(x)
        step = step_init * ranges
        
        while np.max(step / ranges) > 1e-13:
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if time_left() < 0.15:
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
        return x, fx
    
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
            simplex[i+1][i] += step if x0[i] < (lower[i]+upper[i])/2 else -step
            simplex[i+1] = clip(simplex[i+1])
        
        f_vals = np.array([eval_f(s) for s in simplex])
        
        for _ in range(20000):
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            if np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30)) < 1e-14:
                break
            
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
                    xc = clip(centroid - rho*(centroid - simplex[-1]))
                    fc = eval_f(xc)
                    if fc < f_vals[-1]:
                        simplex[-1], f_vals[-1] = xc, fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
    
    # === MAIN STRATEGY ===
    # Phase 1: Initial sampling with LHS + opposition
    pop_size = min(max(20, 8*dim), 200)
    pop = lhs_sample(pop_size)
    fit = np.array([eval_f(p) for p in pop])
    for i in range(pop_size):
        record_good(pop[i], fit[i])
    
    if time_left() > 1:
        opp = lower + upper - pop
        ofit = np.array([eval_f(p) for p in opp])
        for i in range(pop_size):
            record_good(opp[i], ofit[i])
        all_p = np.vstack([pop, opp])
        all_f = np.concatenate([fit, ofit])
        idx = np.argsort(all_f)[:pop_size]
        pop, fit = all_p[idx].copy(), all_f[idx].copy()
    
    # Phase 2: L-SHADE (shorter)
    if time_left() > 2:
        pop, fit = run_lshade(pop, fit, time_left() * 0.20)
    
    # Phase 3: CMA-ES IPOP restarts - main workhorse
    restart_count = 0
    lam_mult_base = 1
    while time_left() > 2.5:
        # Choose start point
        r = np.random.rand()
        if restart_count == 0 and best_pos is not None:
            x0 = best_pos.copy()
            sig = 0.2 * np.mean(ranges)
            lam_m = 1
        elif r < 0.3 and len(good_solutions) > 1:
            # Start from a good solution with perturbation
            idx = np.random.randint(min(5, len(good_solutions)))
            x0 = good_solutions[idx][1].copy() + 0.1 * ranges * np.random.randn(dim)
            sig = 0.15 * np.mean(ranges)
            lam_m = 1
        elif r < 0.5 and best_pos is not None:
            # Near best with medium sigma
            x0 = best_pos.copy() + 0.2 * ranges * np.random.randn(dim)
            sig = 0.2 * np.mean(ranges)
            lam_m = 1
        else:
            # Random restart with increasing population (IPOP)
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.35 * np.mean(ranges)
            lam_mult_base = min(lam_mult_base * 2, 8)
            lam_m = lam_mult_base
        
        # Allocate time: more for larger populations
        base_budget = max(1.5, time_left() * 0.25)
        budget = min(base_budget, time_left() - 1.5)
        if budget < 1.0:
            break
        
        run_cmaes(clip(x0), sig, budget, lam_mult=min(lam_m, 6))
        restart_count += 1
    
    # Phase 4: Local refinement with pattern search
    if time_left() > 1.5 and best_pos is not None:
        pattern_search(best_pos.copy(), time_left() * 0.35, step_init=0.05)
    
    # Phase 5: Nelder-Mead fine tuning
    if time_left() > 1.0 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.5, scale=0.01)
    
    # Phase 6: Final coordinate-wise golden section polish
    if time_left() > 0.5 and best_pos is not None:
        pattern_search(best_pos.copy(), time_left() * 0.8, step_init=0.002)
    
    return best
