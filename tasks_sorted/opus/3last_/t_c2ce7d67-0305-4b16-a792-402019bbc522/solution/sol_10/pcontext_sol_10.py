#Looking at the results, algorithm 1 performed best (3.39) with SHADE DE + CMA-ES restarts + multi-scale coordinate descent. Let me analyze what worked and improve:
#
#1. The full multi-phase approach (SHADE → CMA-ES → coordinate descent) was most effective
#2. CMA-ES restarts helped escape local optima
#3. Need better time allocation - more time for CMA-ES local search
#4. Add Nelder-Mead simplex for final polishing
#5. Better initialization with opposition-based learning
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Latin Hypercube + Opposition-based sampling
    n_init = min(800, max(120, dim * 15))
    perms = np.zeros((n_init, dim))
    for d in range(dim):
        perms[:, d] = np.random.permutation(n_init)
    init_samples = (perms + np.random.rand(n_init, dim)) / n_init
    
    top_k = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.035:
            break
        x = lower + init_samples[i] * ranges
        f = eval_f(x)
        top_k.append((f, x.copy()))
        # Opposition-based learning
        if i < n_init // 3 and elapsed() < max_time * 0.03:
            opp = lower + upper - x
            eval_f(opp)
    
    top_k.sort(key=lambda t: t[0])
    top_k = top_k[:max(10, dim)]

    # Phase 2: SHADE DE with linear population size reduction
    def run_shade(time_frac):
        nonlocal best, best_x
        pop_size_init = min(max(8 * dim, 60), 250)
        pop_size_min = max(4, dim // 2)
        pop_size = pop_size_init
        pop = lower + np.random.rand(pop_size, dim) * ranges
        fit = np.zeros(pop_size)
        
        if best_x is not None:
            pop[0] = best_x.copy()
            idx_seed = 1
            for tk_f, tk_x in top_k[:min(pop_size // 4, 15)]:
                if idx_seed >= pop_size:
                    break
                pop[idx_seed] = tk_x.copy()
                idx_seed += 1
            for j in range(idx_seed, min(idx_seed + 8, pop_size)):
                scale = 0.02 + 0.04 * j
                pop[j] = np.clip(best_x + scale * ranges * np.random.randn(dim), lower, upper)
        
        for i in range(pop_size):
            if remaining() < max_time * 0.55:
                pop = pop[:max(i, 1)]
                fit = fit[:max(i, 1)]
                pop_size = max(i, 1)
                break
            fit[i] = eval_f(pop[i])
        
        if pop_size < 4:
            return
            
        H = 8
        memory_F = np.full(H, 0.5)
        memory_CR = np.full(H, 0.85)
        mem_idx = 0
        archive = []
        
        end_time = elapsed() + max_time * time_frac
        stagnation = 0
        local_best = best
        gen = 0
        max_gen_est = max(1, int(max_time * time_frac * pop_size / (dim * 0.001 + 0.01)))
        
        while elapsed() < end_time and remaining() > max_time * 0.12:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            sort_idx = np.argsort(fit[:pop_size])
            
            for i in range(pop_size):
                if elapsed() >= end_time or remaining() < max_time * 0.12:
                    break
                
                ri = np.random.randint(H)
                while True:
                    F_i = np.random.standard_cauchy() * 0.1 + memory_F[ri]
                    if F_i > 0:
                        break
                F_i = min(F_i, 1.0)
                CR_i = np.clip(np.random.randn() * 0.1 + memory_CR[ri], 0.0, 1.0)
                
                p = max(2, int(0.1 * pop_size))
                pb = sort_idx[np.random.randint(p)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                combined = pop_size + len(archive)
                r2 = np.random.randint(combined)
                att = 0
                while (r2 == i or r2 == r1) and att < 25:
                    r2 = np.random.randint(combined)
                    att += 1
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + F_i * (pop[pb] - pop[i]) + F_i * (pop[r1] - xr2)
                
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i][d]) / 2.0
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i][d]) / 2.0
                
                cross_points = np.random.rand(dim) < CR_i
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = eval_f(trial)
                if f_trial <= fit[i]:
                    df = fit[i] - f_trial
                    if df > 0:
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        S_df.append(df)
                    archive.append(pop[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial
                    fit[i] = f_trial
            
            if S_F:
                w = np.array(S_df)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                memory_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                memory_CR[mem_idx] = np.sum(w * scr)
                mem_idx = (mem_idx + 1) % H
            
            if best < local_best - 1e-12:
                local_best = best
                stagnation = 0
            else:
                stagnation += 1
            
            # L-SHADE population reduction
            new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * (elapsed() - (end_time - max_time * time_frac)) / (max_time * time_frac + 1e-30))))
            if new_pop_size < pop_size:
                sidx = np.argsort(fit[:pop_size])
                pop = pop[sidx[:new_pop_size]].copy()
                fit = fit[sidx[:new_pop_size]].copy()
                pop_size = new_pop_size
            
            if stagnation > 20:
                half = pop_size // 2
                for j in range(half, pop_size):
                    if best_x is not None:
                        sc = 0.06 * (1 + stagnation * 0.02)
                        pop[j] = np.clip(best_x + sc * ranges * np.random.randn(dim), lower, upper)
                    else:
                        pop[j] = lower + np.random.rand(dim) * ranges
                    if remaining() < max_time * 0.12:
                        break
                    fit[j] = eval_f(pop[j])
                stagnation = 0

    run_shade(0.45)

    # Phase 3: CMA-ES restarts
    def run_cmaes(x0, sigma0, lam, budget_time):
        nonlocal best, best_x
        cma_start = elapsed()
        n = dim
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n))
        cc = (4.0 + mu_eff/n) / (n + 4.0 + 2.0*mu_eff/n)
        c1 = 2.0 / ((n+1.3)**2 + mu_eff)
        cmu_v = min(1.0 - c1, 2.0*(mu_eff - 2.0 + 1.0/mu_eff)/((n+2.0)**2 + mu_eff))
        mean = x0.copy()
        sigma = sigma0
        ps = np.zeros(n)
        pc = np.zeros(n)
        use_full = n <= 80
        if use_full:
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
            B = np.eye(n)
            D = np.ones(n)
        else:
            diagC = np.ones(n)
        gen = 0
        stag = 0
        local_best = best
        while (elapsed() - cma_start) < budget_time and remaining() > max_time * 0.06:
            gen += 1
            arz = np.random.randn(lam, n)
            if use_full:
                if gen == 1 or (gen - eigeneval) > lam / (c1 + cmu_v) / n / 10:
                    eigeneval = gen
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        Dv, B = np.linalg.eigh(C)
                        Dv = np.maximum(Dv, 1e-20)
                        D = np.sqrt(Dv)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        C = np.eye(n)
                        D = np.ones(n)
                        B = np.eye(n)
                        invsqrtC = np.eye(n)
                arx = mean + sigma * (arz @ (B * D).T)
            else:
                sqrtD = np.sqrt(diagC)
                arx = mean + sigma * arz * sqrtD
            arx = np.clip(arx, lower, upper)
            fvals = []
            for k in range(lam):
                if remaining() < max_time * 0.04:
                    break
                fvals.append(eval_f(arx[k]))
            if len(fvals) < lam:
                break
            fvals = np.array(fvals)
            idx = np.argsort(fvals)
            if fvals[idx[0]] < local_best - 1e-14:
                local_best = fvals[idx[0]]
                stag = 0
            else:
                stag += 1
            old_mean = mean.copy()
            mean = np.clip(weights @ arx[idx[:mu]], lower, upper)
            diff = mean - old_mean
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ diff) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff / (sigma * sqrtD)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*gen)) / E_norm < 1.4 + 2.0/(n+1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff / sigma
            if use_full:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * (artmp.T @ np.diag(weights) @ artmp)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                diagC = (1 - c1 - cmu_v) * diagC + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + cmu_v * np.sum(weights[:, None] * artmp**2, 0)
                diagC = np.maximum(diagC, 1e-20)
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / E_norm - 1))
            sigma = min(sigma, np.max(ranges) * 2)
            if sigma < 1e-16 or stag > 12 + 30 * n // lam:
                break

    for restart in range(40):
        if remaining() < max_time * 0.10:
            break
        lam = min(int((4 + 3 * np.log(dim)) * min(2**max(0, restart-1), 6)), 250)
        bt = remaining() * 0.30
        if best_x is not None and (restart < 3 or np.random.rand() < 0.65):
            scale = 0.02 * (1.4 ** restart)
            x0 = np.clip(best_x + scale * ranges * np.random.randn(dim), lower
