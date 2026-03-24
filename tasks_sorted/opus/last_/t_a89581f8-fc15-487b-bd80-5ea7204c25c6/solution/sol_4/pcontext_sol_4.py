#Looking at the previous algorithm, I need to analyze what could be improved. The score of 37.65 suggests we're getting stuck in local optima. Key improvements I'll make:
#
#1. **Better exploration with multiple restarts and diverse initialization strategies**
#2. **Improved L-SHADE with better parameter adaptation**
#3. **Smarter time allocation based on dimension**
#4. **Add a proper restart CMA-ES (BIPOP-style)**
#5. **Better local search with Powell-like direction set method**
#6. **Improved boundary handling**
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
    mid = (lower + upper) / 2
    
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
    
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    # --- Adaptive DE (JADE-like with current-to-pbest/1) ---
    def run_de(pop, fit, budget_time):
        t_start = elapsed()
        pop_size = len(pop)
        mu_F = 0.5
        mu_CR = 0.5
        archive = []
        c = 0.1
        
        gen = 0
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            gen += 1
            S_F, S_CR = [], []
            
            sorted_idx = np.argsort(fit)
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if time_left() < 0.2:
                    return pop, fit
                
                # Generate F and CR
                Fi = -1
                while Fi <= 0:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)
                
                # p-best
                p = max(2, int(max(0.05, 0.25 - 0.20 * gen / max(1, budget_time * 30)) * pop_size))
                pbest = sorted_idx[np.random.randint(p)]
                
                # Select r1 != i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                # Select r2 from pop + archive, != i, r1
                pool = list(range(pop_size + len(archive)))
                pool = [x for x in pool if x != i and x != r1]
                if len(pool) == 0:
                    continue
                r2 = pool[np.random.randint(len(pool))]
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                # Mutation
                mutant = pop[i] + Fi * (pop[pbest] - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Crossover
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                
                # Bounce-back boundary
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = lower[d] + np.random.rand() * (pop[i][d] - lower[d])
                    elif trial[d] > upper[d]:
                        trial[d] = upper[d] - np.random.rand() * (upper[d] - pop[i][d])
                trial = clip(trial)
                
                f_trial = eval_f(trial)
                
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        elif pop_size > 0:
                            archive[np.random.randint(len(archive))] = pop[i].copy()
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            
            if len(S_F) > 0:
                mu_F = (1-c)*mu_F + c * np.sum(np.array(S_F)**2) / (np.sum(S_F) + 1e-30)
                mu_CR = (1-c)*mu_CR + c * np.mean(S_CR)
        
        return pop, fit
    
    # --- CMA-ES with sep option ---
    def run_cmaes(x0, sigma0, budget_time):
        t_start = elapsed()
        n = dim
        use_sep = (n > 50)
        
        lam = 4 + int(3 * np.log(n))
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
        stag_count = 0
        prev_best_local = best
        
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
                for k in range(mu):
                    C += cmu_ * weights[k] * np.outer(artmp[k], artmp[k])
                
                if counteval - eigeneval > lam/(c1+cmu_)/n/10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C,1).T
                    try:
                        Dsq, B = np.linalg.eigh(C)
                        Dsq = np.maximum(Dsq, 1e-20)
                        D = np.sqrt(Dsq)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-16, 2*np.max(ranges))
            
            if best < prev_best_local - 1e-10:
                stag_count = 0
                prev_best_local = best
            else:
                stag_count += 1
            
            if sigma < 1e-15 or stag_count > 30 + 10*dim:
                break
            if not use_sep and np.max(D) > 1e7 * np.min(D):
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
            simplex[i+1][i] += scale * ranges[i] * (1 if np.random.rand() > 0.5 else -1)
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
            
            if np.max(np.std(simplex, axis=0)/ranges) < 1e-14:
                break
    
    # === MAIN ===
    pop_size = min(max(40, 15*dim), 400)
    
    # Phase 1: LHS + opposition
    pop = lhs_sample(pop_size)
    fit = np.array([eval_f(p) for p in pop])
    
    if time_left() > 1:
        opp = lower + upper - pop
        ofit = np.array([eval_f(p) for p in opp])
        all_p = np.vstack([pop, opp])
        all_f = np.concatenate([fit, ofit])
        idx = np.argsort(all_f)[:pop_size]
        pop, fit = all_p[idx], all_f[idx]
    
    # Phase 2: DE
    if time_left() > 2:
        pop, fit = run_de(pop, fit, time_left() * 0.35)
    
    # Phase 3: CMA-ES restarts
    n_restarts = 0
    while time_left() > 2.0:
        if n_restarts == 0 and best_pos is not None:
            x0 = best_pos.copy()
            sig = 0.2 * np.mean(ranges)
        else:
            x0 = lower + np.random.rand(dim) * ranges
            sig = 0.4 * np.mean(ranges)
        run_cmaes(x0, sig, min(time_left() * 0.4, max(2.0, time_left() - 1.5)))
        n_restarts += 1
    
    # Phase 4: Nelder-Mead polish
    if time_left() > 0.5 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.8, scale=0.01)
    
    return best
