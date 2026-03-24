#Looking at the previous algorithm, I can identify several issues:
#
#1. **SHADE DE has a bug**: `pbest_idx` uses sorted index but population may not be sorted at mutation time
#2. **CMA-ES restarts get too little time** - the time allocation is suboptimal
#3. **Pattern search is too slow** for high dimensions
#4. **Missing Nelder-Mead** which can be very effective for local search
#5. **Population size may be too large** wasting evaluations
#
#Let me redesign with better components:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.995 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def bounce_back(mutant, parent):
        x = mutant.copy()
        for d in range(dim):
            if x[d] < lower[d]:
                x[d] = (lower[d] + parent[d]) / 2.0
            elif x[d] > upper[d]:
                x[d] = (upper[d] + parent[d]) / 2.0
        return x

    # ---- Nelder-Mead local search ----
    def nelder_mead(x0, max_nm_time, initial_scale=0.05):
        nonlocal best, best_params
        nm_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            step = initial_scale * ranges[i]
            if step < 1e-12:
                step = 1e-6
            p[i] += step
            p = clip(p)
            simplex[i + 1] = p
        
        f_simplex = np.array([evaluate(simplex[i]) for i in range(n + 1) if remaining() > 0])
        if len(f_simplex) < n + 1:
            return
        
        for _ in range(10000):
            if (elapsed() - nm_start) >= max_nm_time or remaining() <= 0:
                return
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                # Expand
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[n] = xc; f_simplex[n] = fc
                    else:
                        for i in range(1, n + 1):
                            if remaining() <= 0: return
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[n]))
                    fc = evaluate(xc)
                    if fc < f_simplex[n]:
                        simplex[n] = xc; f_simplex[n] = fc
                    else:
                        for i in range(1, n + 1):
                            if remaining() <= 0: return
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-16:
                return

    # ---- CMA-ES ----
    def cmaes_search(x0, sigma0, max_cma_time):
        nonlocal best, best_params
        cma_start = elapsed()
        n = dim
        mean = clip(x0.copy())
        sigma = sigma0
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        eigen_counter = 0
        gen = 0
        stag_count = 0
        best_cma = float('inf')
        
        while (elapsed() - cma_start) < max_cma_time and remaining() > 0:
            gen += 1
            
            # Eigendecomposition
            if eigen_counter >= max(1, int(1.0/(10*n*(c1+cmu_val)+1e-30))):
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-30)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                    eigen_counter = 0
                except:
                    return
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
            
            fit = np.zeros(lam)
            for k in range(lam):
                if remaining() <= 0:
                    return
                fit[k] = evaluate(arx[k])
            
            idx = np.argsort(fit)
            
            if fit[idx[0]] < best_cma - 1e-14:
                best_cma = fit[idx[0]]
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*gen)) / chiN < 1.4 + 2.0/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 3)
            eigen_counter += 1
            
            if sigma < 1e-18 or stag_count > 50 + 20*n:
                return
            if np.max(D) > 1e8 * np.min(D):
                return

    # ---- L-SHADE DE ----
    def lshade_de(pop, fit, max_de_time, min_pop=4):
        nonlocal best, best_params
        de_start = elapsed()
        pop_size = len(pop)
        init_pop_size = pop_size
        H = 100
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        hist_pos = 0
        archive = []
        archive_max = pop_size
        gen = 0
        
        while (elapsed() - de_start) < max_de_time and remaining() > 0 and pop_size >= min_pop:
            gen += 1
            SF, SCR, S_delta = [], [], []
            trial_pop = pop.copy()
            trial_fit = fit.copy()
            
            # Sort for current-to-pbest
            sort_idx = np.argsort(fit)
            
            for i in range(pop_size):
                if remaining() <= 0 or (elapsed() - de_start) >= max_de_time:
                    return pop[:pop_size], fit[:pop_size]
                
                ri = np.random.randint(0, H)
                while True:
                    Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
                    if Fi > 0:
                        break
                Fi = min(Fi, 1.0)
                CRi = np.clip(np.random.normal(MCR[ri], 0.1), 0.0, 1.0)
                
                p = max(2, int(np.random.uniform(0.05, 0.25) * pop_size))
                top_p = sort_idx[:p]
                pbest_idx = top_p[np.random.randint(len(top_p))]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                combined_size = pop_size + len(archive)
                r2 = np.random.randint(0, combined_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, combined_size)
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                mutant = bounce_back(mutant, pop[i])
                
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = evaluate(trial)
                if f_trial <= fit[i]:
                    if f_trial < fit[i]:
                        SF.append(Fi)
                        SCR.append(CRi)
                        S_delta.append(abs(fit[i] - f_trial))
                        archive.append(pop[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                    trial_pop[i] = trial
                    trial_fit[i] = f_trial
            
            pop = trial_pop
            fit = trial_fit
            
            if SF and sum(S_delta) > 0:
                w = np.array(S_delta)
                w = w / w.sum()
                sf_a = np.array(SF)
                scr_a = np.array(SCR)
                MF[hist_pos] = np.sum(w * sf_a**2) / (np.sum(w * sf_a) + 1e-30)
                MCR[hist_pos] = np.sum(w * scr_a)
                hist_pos = (hist_pos + 1) % H
            
            # Linear population size reduction (L-SHADE)
            new_pop_size = max(min_pop, int(round(init_pop_size + (min_pop - init_pop_size) * (elapsed() - de_start) / max(max_de_time, 1e-10))))
            if new_pop_size < pop_size:
                idx2 = np.argsort(fit)
                pop = pop[idx2[:new_pop_size]]
                fit = fit[idx2[:new_pop_size]]
                pop_size = new_pop_size
                archive_max = pop_size
                while len(archive) > archive_max:
                    archive.pop(np.random.randint(len(archive)))
        
        return pop[:pop_size], fit[:pop_size]

    # ==== MAIN FLOW ====
    
    # Phase 1: LHS initialization with opposition
    pop_size = min(max(20, 10 * dim), 300)
    pop = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    pop = lower + pop * ranges
    
    # Evaluate
    fit = np.zeros(pop_size)
    for i in range(pop_size):
        if remaining() <= 0:
            return best
        fit[i] = evaluate(pop[i])
    
    # Opposition-based
    opp_pop = lower + upper - pop
    opp_fit = np.zeros(pop_size)
    for i in range(pop_size):
        if remaining() <= 0:
            break
        opp_fit[i] = evaluate(opp_pop[i])
    
    combined = np.vstack([pop, opp_pop])
    combined_fit = np.concatenate([fit, opp_fit])
    idx = np.argsort(combined_fit)
    pop = combined[idx[:pop_size]].copy()
    fit = combined_fit[idx[:pop_size]].copy()
    
    # Phase 2: L-SHADE DE (use ~45% of time)
    de_budget = remaining() * 0.45
    if de_budget > 0.5 and remaining() > 1:
        pop, fit = lshade_de(pop, fit, de_budget)
    
    # Phase 3: CMA-ES restarts (use ~40% of remaining time)
    restart = 0
    while remaining() > max_time * 0.10:
        restart += 1
        cma_time = min(remaining() - max_time * 0.08, remaining() * 0.45)
        if cma_time < 0.3:
            break
        
        if restart == 1:
            x0 = best_params.copy()
            s0 = np.max(ranges) * 0.1
        elif restart == 2:
            x0 = best_params.copy()
            s0 = np.max(ranges) * 0.02
        elif restart <= 4:
            x0 = best_params + np.random.randn(dim) * ranges * 0.1
            x0 = clip(x0)
            s0 = np.max(ranges) * 0.15
        else:
            x0 = lower + np.random.random(dim) * ranges
            s0 = np.max(ranges) * 0.3
        
        cmaes_search(x0, s0, cma_time)
    
    # Phase 4: Nelder-Mead polish
    if best_params is not None and remaining() > 0.5:
        nelder_mead(best_params.copy(), remaining() * 0.6, initial_scale=0.02)
    
    if best_params is not None and remaining() > 0.3:
        nelder_mead(best_params.copy(), remaining() * 0.8, initial_scale=0.001)
    
    return best
#
#Key improvements:
#1. **L-SHADE** with linear population reduction - focuses search as time progresses
#2. **Fixed pbest selection** - properly selects from top-p sorted individuals
#3. **Better time allocation**: 45% DE, ~40% CMA-ES restarts, ~15% Nelder-Mead
#4. **Multiple CMA-ES restarts** with decreasing then increasing sigma for exploration/exploitation balance
#5. **Nelder-Mead** for final polishing instead of pattern search (more efficient in moderate dimensions)
#6. **Better eigendecomposition scheduling** with proper frequency control
#7. **Improved stagnation handling** in CMA-ES with tighter thresholds
