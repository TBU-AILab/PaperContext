#Looking at the results, algorithm 5 (score 36.62) and algorithm 7 (score 28.52) performed best. Algorithm 7 used DE+CMA-ES with LHS, while algorithm 5 used SHADE+CMA-ES+Nelder-Mead. Let me combine the best elements and add key improvements:
#
#1. **Better L-SHADE with Lehmer mean for both F and CR**
#2. **Separable CMA-ES for high dimensions**
#3. **More aggressive multi-restart with warm-starting from elite archive**
#4. **Randomized coordinate descent with adaptive step sizes**
#5. **Better time allocation favoring CMA-ES**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    eval_time = [0.0]
    
    top_k = min(50, max(10, 2*dim))
    top_solutions = []
    top_fitnesses = []
    
    def update_top(x, f):
        if len(top_solutions) < top_k:
            top_solutions.append(x.copy())
            top_fitnesses.append(f)
        elif f < max(top_fitnesses):
            idx = np.argmax(top_fitnesses)
            top_solutions[idx] = x.copy()
            top_fitnesses[idx] = f
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        t0 = elapsed()
        f = func(x)
        eval_time[0] += elapsed() - t0
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        update_top(x, f)
        return f
    
    def avg_eval():
        if eval_count[0] == 0:
            return 0.001
        return eval_time[0] / eval_count[0]
    
    # Phase 1: LHS + Opposition
    n_init = min(max(20 * dim, 200), 2000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    actual_init = 0
    for i in range(n_init):
        if elapsed() > max_time * 0.06:
            break
        init_fitness[i] = eval_func(init_pop[i])
        actual_init = i + 1
    
    if time_ok(0.07):
        eval_func((lower + upper) / 2.0)
    
    if actual_init > 10 and time_ok(0.09):
        si = np.argsort(init_fitness[:actual_init])
        n_opp = min(actual_init // 4, 30)
        for i in range(n_opp):
            if elapsed() > max_time * 0.09:
                break
            opp = lower + upper - init_pop[si[i]]
            eval_func(opp)
    
    sorted_idx = np.argsort(init_fitness[:actual_init])
    
    # L-SHADE
    def lshade_phase(time_budget):
        nonlocal best, best_params
        if time_budget < 0.1:
            return
        deadline = elapsed() + time_budget
        
        N_init = min(max(8 * dim, 60), 300)
        N_min = max(4, dim // 3)
        pop_size = N_init
        H = 100
        
        n_elite = min(pop_size // 3, actual_init)
        pop = np.zeros((N_init, dim))
        fit = np.full(N_init, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, N_init):
            pop[i] = lower + np.random.random(dim) * ranges
        
        for i in range(n_elite, N_init):
            if not time_ok(0.96) or elapsed() > deadline:
                return
            fit[i] = eval_func(pop[i])
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.8)
        k = 0
        archive = []
        max_archive = N_init
        
        et = avg_eval()
        max_evals_est = max(1, int(time_budget / max(et, 1e-6)))
        evals_used = 0
        gen = 0
        stag = 0
        prev_best = best
        
        while time_ok(0.96) and elapsed() < deadline:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            sort_fit = np.argsort(fit[:pop_size])
            
            for i in range(pop_size):
                if not time_ok(0.96) or elapsed() > deadline:
                    pop[:pop_size] = new_pop
                    fit[:pop_size] = new_fit
                    return
                
                ri = np.random.randint(0, H)
                for _ in range(10):
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi > 0:
                        break
                Fi = min(max(Fi, 0.01), 1.0)
                
                CRi = M_CR[ri]
                if CRi < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(CRi + 0.1 * np.random.randn(), 0, 1)
                
                progress = evals_used / max(1, max_evals_est)
                p_rate = max(2.0/pop_size, 0.05 + 0.20 * (1 - progress))
                p = max(2, int(pop_size * p_rate))
                pbest_idx = sort_fit[:p]
                xpbest = pop[pbest_idx[np.random.randint(p)]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = idxs[np.random.randint(len(idxs))]
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                att = 0
                while (r2 == i or r2 == r1) and att < 25:
                    r2 = np.random.randint(pool_size)
                    att += 1
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - xr2)
                for di in range(dim):
                    if mutant[di] < lower[di]:
                        mutant[di] = (lower[di] + pop[i, di]) / 2
                    elif mutant[di] > upper[di]:
                        mutant[di] = (upper[di] + pop[i, di]) / 2
                
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[jrand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = eval_func(trial)
                evals_used += 1
                
                if f_trial < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_df.append(fit[i] - f_trial)
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial; new_fit[i] = f_trial
                elif f_trial == fit[i]:
                    new_pop[i] = trial; new_fit[i] = f_trial
            
            pop[:pop_size] = new_pop; fit[:pop_size] = new_fit
            
            if len(S_F) > 0:
                sdf = np.array(S_df); w = sdf / (np.sum(sdf) + 1e-30)
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr**2) / (np.sum(w * scr) + 1e-30) if np.max(scr) > 0 else -1.0
                k = (k + 1) % H
            
            if best < prev_best - 1e-12:
                stag = 0; prev_best = best
            else:
                stag += 1
            
            new_ps = max(N_min, int(round(N_init - (N_init - N_min) * evals_used / max(1, max_evals_est))))
            if new_ps < pop_size:
                bi = np.argsort(fit[:pop_size])[:new_ps]
                pop[:new_ps] = pop[bi]; fit[:new_ps] = fit[bi]; pop_size = new_ps
            
            if stag > 20 and pop_size > N_min + 2:
                nr = max(1, pop_size // 5)
                wi = np.argsort(fit[:pop_size])[-nr:]
                for w_i in wi:
                    if len(top_solutions) > 2:
                        st = np.argsort(top_fitnesses)
                        pick = min(np.random.randint(min(5, len(top_solutions))), len(top_solutions)-1)
                        pop[w_i] = clip(top_solutions[st[pick]] + 0.1 * ranges * np.random.randn(dim))
                    else:
                        pop[w_i] = lower + np.random.random(dim) * ranges
                    fit[w_i] = eval_func(pop[w_i])
                stag = 0
    
    # CMA-ES with sep-CMA for high dim
    def cma_es_run(x0, sigma0, time_budget, pop_mult=1):
        nonlocal best, best_params
        if time_budget < 0.05:
            return
        deadline = elapsed() + time_budget
        n = dim
        use_sep = n > 40
        
        lam = max(4 + int(3 * np.log(n)), 6) * pop_mult
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        if use_sep:
            c1 *= (n + 2) / 3; cmu_val *= (n + 2) / 3
            c1 = min(c1, 0.9); cmu_val = min(cmu_val, 0.9 - c1)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
        
        pc = np.zeros(n); ps = np.zeros(n)
        mean = x0.copy(); sigma = sigma0
        
        if use_sep:
            diagC = np.ones(n)
        else:
            B = np.eye(n); D = np.ones(n); C = np.eye(n); invsqrtC = np.eye(n)
        
        gen = 0; no_improve = 0; prev_best = best
        eigfreq = max(1, int(1.0 / (c1 + cmu_val + 1e-20) / n / 10))
        
        while time_ok(0.96) and elapsed() < deadline:
            if use_sep:
                sqrtC = np.sqrt(diagC)
                arz = np.random.randn(lam, n)
                arx = mean[None, :] + sigma * sqrtC[None, :] * arz
                for ki in range(lam):
                    oob = (arx[ki] < lower) | (arx[ki] > upper)
                    if np.any(oob):
                        arx[ki] = np.where(oob, lower + np.random.random(n) * ranges, arx[ki])
                    arx[ki] = clip(arx[ki])
            else:
                arz = np.random.randn(lam, n)
                arx = np.zeros((lam, n))
                BD = B * D[None, :]
                for ki in range(lam):
                    arx[ki] = mean + sigma * (BD @ arz[ki])
                    oob = (arx[ki] < lower) | (arx[ki] > upper)
                    if np.any(oob):
                        arx[ki] = np.where(oob, mean + sigma * np.random.randn(n) * D, arx[ki])
                    arx[ki] = clip(arx[ki])
            
            fitnesses = np.zeros(lam)
            for ki in range(lam):
                if not time_ok(0.96) or elapsed() > deadline:
                    return
                fitnesses[ki] = eval_func(arx[ki])
            
            idx = np.argsort(fitnesses)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            diff = mean - old_mean
            
            if use_sep:
                invsqrtCvec = 1.0 / np.sqrt(diagC + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtCvec * diff) / max(sigma, 1e-30)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / max(sigma, 1e-30)
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / max(sigma, 1e-30)
            
            artmp = (arx[idx[:mu]] - old_mean) / max(sigma, 1e-30)
            if use_sep:
                diagC = ((1 - c1 - cmu_val) * diagC + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + cmu_val * np.sum(weights[:, None] * artmp**2, axis=0))
                diagC = np.maximum(diagC, 1e-20)
            else:
                C = ((1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * np.einsum('k,ki,kj->ij', weights, artmp, artmp))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            gen += 1
            
            if not use_sep and gen % eigfreq == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, Bn = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq); B = Bn
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            if best < prev_best - 1e-10:
                no_improve = 0; prev_best = best
            else:
                no_improve += 1
            
            cond_val = sigma * (np.max(np.sqrt(diagC)) if use_sep else np.max(D))
            if cond_val < 1e-14 * np.max(ranges) or no_improve > 25 + 8*n:
                break
    
    def nelder_mead(x0, scale_factor, time_budget):
        nonlocal best, best_params
        if time_budget < 0.05:
            return
        deadline = elapsed() + time_budget
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1, i] += scale[i] * (1 if np.random.random() < 0.5 else -1)
            simplex[i+1] = clip(simplex[i+1])
        f_s = np.zeros(n + 1)
        for i in range(n+1):
            if not time_ok(0.96) or elapsed() > deadline: return
            f_s[i] = eval_func(simplex[i])
        while time_ok(0.97) and elapsed() < deadline:
            order = np.argsort(f_s); simplex = simplex[order]; f_s = f_s[order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + (centroid - simplex[-1]))
            fr = eval_func(xr)
            if f_s[0] <= fr < f_s[-2]:
                simplex[-1] = xr; f_s[-1] = fr
            elif fr < f_s[0]:
                xe = clip(centroid + 2.0*(xr - centroid))
                if not time_ok(0.97) or elapsed() > deadline: break
                fe = eval_func(xe)
                if fe < fr: simplex[-1] = xe; f_s[-1] = fe
                else: simplex[-1] = xr; f_s[-1] = fr
            else:
                if fr < f_s[-1]: xc = clip(centroid + 0.5*(xr - centroid))
                else: xc = clip(centroid + 0.5*(simplex[-1] - centroid))
                if not time_ok(0.97) or elapsed() > deadline: break
                fc = eval_func(xc)
                if fc < min(fr, f_s[-1]): simplex[-1] = xc; f_s[-1] = fc
                else:
                    for i in range(1, n+1):
                        if not time_ok(0.97) or elapsed() > deadline: return
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i]); f_s[i] = eval_func(simplex[i])
            if np.max(np.abs(f_s - f_s[0])) < 1e-16: break
    
    def pattern_search(x0, time_budget):
        nonlocal best, best_params
        if time_budget < 0.05: return
        deadline = elapsed() + time_budget
        x = x0.copy(); fx = eval_func(x)
        step = 0.02 * ranges.copy()
        while time_ok(0.98) and elapsed() < deadline:
            improved = False
            for di in np.random.permutation(dim):
                if not time_ok(0.98) or elapsed() > deadline: return
                xn = x.copy(); xn[di] = min(x[di] + step[di], upper[di])
                fn = eval_func(xn)
                if fn < fx: x = xn; fx = fn; step[di] *= 1.3; improved = True; continue
                xn = x.copy(); xn[di] = max(x[di] - step[di], lower[di])
                if not time_ok(0.98) or elapsed() > deadline: return
                fn = eval_func(xn)
                if fn < fx: x = xn; fx = fn; step[di] *= 1.3; improved = True
                else: step[di] *= 0.5
            if not improved or np.max(step / ranges) < 1e-14: break
    
    rem = max_time - elapsed()
    lshade_phase(rem * 0.28)
    
    restart = 0
    while time_ok(0.78):
        rem_cma = max_time * 0.78 - elapsed()
        if rem_cma < 0.15: break
        n_left = max(1, min(8, int(rem_cma / max(0.2, rem_cma/6))))
        t_budget = rem_cma / n_left * 0.9
        if restart == 0:
            x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim)*ranges
            sig = 0.2 * np.max(ranges); pm = 1
        elif restart % 3 == 1:
            if len(top_solutions) > 1:
                st = np.argsort(top_fitnesses); pick = min(restart // 3, len(top_solutions) - 1)
                x0 = top_solutions[st[pick]].copy()
            else: x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim)*ranges
            x0 = clip(x0 + 0.04*ranges*np.random.randn(dim)); sig = 0.07 * np.max(ranges); pm = 1
        elif restart % 3 == 2:
            x0 = lower + np.random.random(dim)*ranges; sig = 0.4 * np.max(ranges); pm = 2
        else:
            x0 = clip(best_params + 0.1*ranges*np.random.randn(dim)) if best_params is not None else lower + np.random.random(dim)*ranges
            sig = 0.15 * np.max(ranges); pm = 1
        cma_es_run(x0, min(sig, np.max(ranges)), t_budget, pm)
        restart += 1
    
    if time_ok(0.80) and len(top_solutions) > 0:
        st = np.argsort(top_fitnesses)
        nm_runs = min(3, len(top_solutions))
        for r in range(nm_runs):
            if not time_ok(0.88): break
            rem_nm = max_time * 0.90 - elapsed()
            if rem_nm < 0.1: break
            nelder_mead(top_solutions[st[r]], 0.02 / (1 + r), rem_nm / (nm_runs - r) * 0.7)
    
    if time_ok(0.90) and best_params is not None:
        nelder_mead(best_params, 0.004, (max_time*0.95 - elapsed()) * 0.5)
    
    if time_ok(0.95) and best_params is not None:
        pattern_search(best_params, max_time*0.995 - elapsed())
    
    return best
