#Looking at the current algorithm, I see it scores 13.64. Let me analyze potential issues:
#
#1. The CMA-ES implementation may have bugs or inefficiencies
#2. Time allocation between phases may not be optimal
#3. The DE population might be too large or too small
#4. Need better restart diversity and adaptive mechanisms
#
#Let me redesign with a cleaner, more robust approach:
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
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Differential Evolution with SHADE + Linear Pop Reduction (L-SHADE) ---
    def run_lshade(time_budget):
        nonlocal best, best_params
        t_start = elapsed()
        
        N_init = max(18 * dim, 80)
        N_min = 4
        pop_size = N_init
        
        # LHS initialization
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            pop[:, d] = lower[d] + (perm + np.random.random(pop_size)) / pop_size * ranges[d]
        
        fit = np.array([evaluate(pop[i]) for i in range(pop_size)])
        if remaining() <= 0:
            return
        
        H = 100
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        k = 0
        archive = []
        
        total_evals_est = 0
        max_evals_est = int(time_budget * pop_size * 2)  # rough estimate
        
        gen = 0
        while True:
            gen += 1
            if elapsed() - t_start > time_budget * 0.95 or remaining() < 0.05:
                break
            
            SF, SCR, S_delta = [], [], []
            
            # Generate F and CR
            r_idx = np.random.randint(0, H, pop_size)
            F_arr = np.zeros(pop_size)
            CR_arr = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    fi = np.random.standard_cauchy() * 0.1 + MF[r_idx[i]]
                    if fi > 0:
                        break
                F_arr[i] = min(fi, 1.0)
                cri = np.random.normal(MCR[r_idx[i]], 0.1)
                CR_arr[i] = np.clip(cri, 0.0, 1.0)
            
            p_best_rate = max(2.0/pop_size, 0.05 + 0.15 * (1.0 - (elapsed()-t_start)/time_budget))
            
            trials = np.empty((pop_size, dim))
            trial_from = np.zeros(pop_size, dtype=bool)
            
            sorted_idx = np.argsort(fit)
            
            for i in range(pop_size):
                Fi = F_arr[i]
                CRi = CR_arr[i]
                
                p_num = max(2, int(np.ceil(p_best_rate * pop_size)))
                pb = sorted_idx[np.random.randint(0, p_num)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(0, union_size - 1)
                if r2 >= i:
                    r2 += 1
                if r2 == r1:
                    r2 = np.random.randint(0, union_size)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(0, union_size)
                
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Bounce back
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i][d]) / 2.0
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i][d]) / 2.0
                
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[jrand] = True
                trial[mask] = mutant[mask]
                trials[i] = trial
            
            # Evaluate all trials
            for i in range(pop_size):
                if elapsed() - t_start > time_budget * 0.95 or remaining() < 0.05:
                    return
                f_trial = evaluate(trials[i])
                
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        SF.append(F_arr[i])
                        SCR.append(CR_arr[i])
                        S_delta.append(abs(fit[i] - f_trial))
                        archive.append(pop[i].copy())
                    pop[i] = trials[i]
                    fit[i] = f_trial
            
            # Trim archive
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            
            # Update memory
            if SF:
                w = np.array(S_delta)
                ws = w / (np.sum(w) + 1e-30)
                sf = np.array(SF)
                scr = np.array(SCR)
                MF[k] = np.sum(ws * sf * sf) / (np.sum(ws * sf) + 1e-30)
                MCR[k] = np.sum(ws * scr)
                k = (k + 1) % H
            
            # Linear population size reduction
            total_evals_est += pop_size
            ratio = min(1.0, (elapsed() - t_start) / time_budget)
            new_pop_size = max(N_min, int(np.round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                sidx = np.argsort(fit)
                pop = pop[sidx[:new_pop_size]]
                fit = fit[sidx[:new_pop_size]]
                pop_size = new_pop_size
                while len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))

    # --- CMA-ES ---
    def run_cmaes(x0, sigma0, time_budget):
        nonlocal best, best_params
        t_start = elapsed()
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        while elapsed() - t_start < time_budget and remaining() > 0.05:
            # Eigen decomposition
            if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = counteval
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D2, B = np.linalg.eigh(C)
                    D2 = np.maximum(D2, 1e-20)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    B = np.eye(n)
                    D = np.ones(n)
                    C = np.eye(n)
                    invsqrtC = np.eye(n)
            
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.empty((lam, n))
            for k in range(lam):
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
            
            # Evaluate
            arfitness = np.empty(lam)
            for k in range(lam):
                if remaining() < 0.05:
                    return
                arfitness[k] = evaluate(arx[k])
                counteval += 1
            
            # Sort
            arindex = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for i in range(mu):
                mean += weights[i] * arx[arindex[i]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2.0/(n+1))
            
            # CMA
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
            
            artmp = np.empty((mu, n))
            for i in range(mu):
                artmp[i] = (arx[arindex[i]] - old_mean) / sigma
            
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + \
                cmu * sum(weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
            
            # Sigma adaptation
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.max(ranges) * 2)
            
            if sigma < 1e-16:
                break
            if np.max(D) > 1e7 * np.min(D):
                break
            
            # Check for stagnation
            if arfitness[arindex[0]] == arfitness[arindex[min(lam-1, mu)]]:
                sigma *= np.exp(0.2 + cs/damps)

    # --- Nelder-Mead ---
    def run_nm(x0, time_budget, step_scale=0.02):
        nonlocal best, best_params
        t_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.empty((n+1, n))
        simplex[0] = x0.copy()
        step = ranges * step_scale
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += step[i] if x0[i] + step[i] <= upper[i] else -step[i]
        
        fs = np.array([evaluate(clip(simplex[i])) for i in range(n+1)])
        
        while elapsed() - t_start < time_budget and remaining() > 0.02:
            order = np.argsort(fs)
            simplex = simplex[order]; fs = fs[order]
            c = np.mean(simplex[:-1], axis=0)
            
            xr = clip(c + alpha*(c - simplex[-1]))
            fr = evaluate(xr)
            if remaining() < 0.02: break
            
            if fr < fs[0]:
                xe = clip(c + gamma*(xr - c))
                fe = evaluate(xe)
                if fe < fr: simplex[-1], fs[-1] = xe, fe
                else: simplex[-1], fs[-1] = xr, fr
            elif fr < fs[-2]:
                simplex[-1], fs[-1] = xr, fr
            else:
                if fr < fs[-1]:
                    xc = clip(c + rho*(xr - c)); fc = evaluate(xc)
                    if fc <= fr: simplex[-1], fs[-1] = xc, fc
                    else:
                        for i in range(1,n+1):
                            simplex[i] = clip(simplex[0]+sigma_nm*(simplex[i]-simplex[0]))
                            fs[i] = evaluate(simplex[i])
                else:
                    xc = clip(c - rho*(xr - c)); fc = evaluate(xc)
                    if fc < fs[-1]: simplex[-1], fs[-1] = xc, fc
                    else:
                        for i in range(1,n+1):
                            simplex[i] = clip(simplex[0]+sigma_nm*(simplex[i]-simplex[0]))
                            fs[i] = evaluate(simplex[i])
            
            if np.max(np.abs(simplex[-1]-simplex[0])) < 1e-14:
                break

    # === Main orchestration ===
    # Phase 1: L-SHADE (55% of time)
    run_lshade(max_time * 0.55)
    
    # Phase 2: CMA-ES restarts (30% of time)
    cma_end = max_time * 0.85
    restart = 0
    while remaining() > max_time * 0.15:
        budget = min(remaining() - max_time*0.15, max_time * 0.12)
        if budget < 0.1: break
        if restart == 0 and best_params is not None:
            x0 = best_params.copy()
            s0 = np.max(ranges) * 0.15
        else:
            x0 = lower + np.random.random(dim) * ranges
            if best_params is not None and np.random.random() < 0.5:
                x0 = best_params + np.random.randn(dim) * ranges * 0.3
                x0 = clip(x0)
            s0 = np.max(ranges) * 0.25
        run_cmaes(x0, s0, budget)
        restart += 1
    
    # Phase 3: Nelder-Mead polish
    if best_params is not None and remaining() > 0.1:
        run_nm(best_params.copy(), remaining() * 0.7, step_scale=0.01)
    if best_params is not None and remaining() > 0.05:
        run_nm(best_params.copy(), remaining(), step_scale=0.001)
    
    return best
