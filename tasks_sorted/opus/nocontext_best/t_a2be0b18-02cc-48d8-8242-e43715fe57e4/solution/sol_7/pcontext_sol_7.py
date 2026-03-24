#Looking at the previous algorithm, I can identify several issues:
#
#1. **The SHADE-DE `pbest_idx` selection is wrong** - it's sorting but using raw index, not sorted index
#2. **CMA-ES restarts don't get enough time** and the time allocation is suboptimal
#3. **Pattern search is too simple** - should use coordinate descent with golden section
#4. **Missing a proper Nelder-Mead local search**
#5. **Population reduction (LPSR) could help DE convergence**
#
#Let me redesign with fixes and improvements:
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
        return max_time * 0.99 - elapsed()
    
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

    def nelder_mead(x0, initial_step=0.05, max_nm_time=1.0):
        nonlocal best, best_params
        nm_start = elapsed()
        n = len(x0)
        # Build initial simplex
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += initial_step * ranges[i]
            if p[i] > upper[i]:
                p[i] = x0[i] - initial_step * ranges[i]
            simplex[i+1] = clip(p)
        
        f_simplex = np.array([evaluate(simplex[i]) for i in range(n+1)])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while (elapsed() - nm_start) < max_nm_time and remaining() > 0:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = evaluate(xr)
            if remaining() <= 0: return
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n-1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[n] = xc; f_simplex[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                            if remaining() <= 0: return
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc < f_simplex[n]:
                        simplex[n] = xc; f_simplex[n] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                            if remaining() <= 0: return
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-15:
                break
            diam = np.max([np.linalg.norm(simplex[i]-simplex[0]) for i in range(1,n+1)])
            if diam < 1e-15:
                break

    def cmaes_search(x0, sigma0, max_cma_time):
        nonlocal best, best_params
        cma_start = elapsed()
        n = dim
        mean = x0.copy()
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
        eigen_update = 0
        gen = 0
        best_cma = float('inf')
        stag = 0
        
        while (elapsed() - cma_start) < max_cma_time and remaining() > 0:
            gen += 1
            
            if eigen_update >= max(1, int(1.0/(c1 + cmu_val)/n/10)):
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-30)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                    eigen_update = 0
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
                    sigma *= 0.5
            eigen_update += 1
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Bounce back
                for d in range(n):
                    if arx[k][d] < lower[d]:
                        arx[k][d] = lower[d] + abs(arx[k][d] - lower[d]) % (ranges[d]) if ranges[d] > 0 else lower[d]
                    if arx[k][d] > upper[d]:
                        arx[k][d] = upper[d] - abs(arx[k][d] - upper[d]) % (ranges[d]) if ranges[d] > 0 else upper[d]
                arx[k] = clip(arx[k])
            
            fit = np.zeros(lam)
            for k in range(lam):
                if remaining() <= 0: return
                fit[k] = evaluate(arx[k])
            
            idx = np.argsort(fit)
            
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
            
            if fit[idx[0]] < best_cma - 1e-12:
                best_cma = fit[idx[0]]
                stag = 0
            else:
                stag += 1
            
            if sigma < 1e-16 or stag > 50 + 20*n:
                break
            if np.max(D) > 1e7 * np.min(D):
                break

    def shade_de(population, fitness, max_de_time, min_pop=None):
        nonlocal best, best_params
        de_start = elapsed()
        pop_size = len(population)
        if min_pop is None:
            min_pop = max(6, dim)
        H = 100
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.8)
        hist_pos = 0
        archive = []
        archive_max = pop_size
        init_pop_size = pop_size
        gen = 0
        nfe_start = evals
        max_nfe = None  # We rely on time
        
        while (elapsed() - de_start) < max_de_time and remaining() > 0:
            gen += 1
            SF, SCR, S_delta = [], [], []
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if remaining() <= 0 or (elapsed() - de_start) >= max_de_time:
                    idx2 = np.argsort(new_fit)
                    return new_pop[idx2], new_fit[idx2]
                
                ri = np.random.randint(0, H)
                while True:
                    Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
                    if Fi > 0: break
                Fi = min(Fi, 1.0)
                CRi = np.clip(np.random.normal(MCR[ri], 0.1), 0.0, 1.0)
                
                # current-to-pbest/1
                p = max(2, int(np.random.uniform(0.05, 0.25) * pop_size))
                sorted_idx = np.argsort(new_fit)
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                combined = pop_size + len(archive)
                r2 = np.random.randint(0, combined)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, combined)
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
                
                # Midpoint bounce-back
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + population[i][d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + population[i][d]) / 2
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = evaluate(trial)
                if f_trial <= new_fit[i]:
                    if f_trial < new_fit[i]:
                        SF.append(Fi)
                        SCR.append(CRi)
                        S_delta.append(abs(new_fit[i] - f_trial))
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            if SF and sum(S_delta) > 0:
                w = np.array(S_delta); w /= w.sum()
                sf_a = np.array(SF); scr_a = np.array(SCR)
                MF[hist_pos] = np.sum(w * sf_a**2) / (np.sum(w * sf_a) + 1e-30)
                MCR[hist_pos] = np.sum(w * scr_a)
                hist_pos = (hist_pos + 1) % H
            
            population = new_pop
            fitness = new_fit
            idx2 = np.argsort(fitness)
            population = population[idx2]
            fitness = fitness[idx2]
            
            # Linear population size reduction
            elapsed_frac = (elapsed() - de_start) / max(max_de_time, 1e-10)
            new_size = max(min_pop, int(init_pop_size - (init_pop_size - min_pop) * elapsed_frac))
            if new_size < pop_size:
                population = population[:new_size]
                fitness = fitness[:new_size]
                pop_size = new_size
                archive_max = pop_size
        
        idx2 = np.argsort(fitness)
        return population[idx2], fitness[idx2]

    # ==== MAIN FLOW ====
    
    # Phase 1: Latin Hypercube Sampling + opposition
    pop_size = min(250, max(40, 18 * dim))
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    opp = lower + upper - pop
    all_pts = np.vstack([pop, opp])
    
    fit_all = []
    for i in range(len(all_pts)):
        if remaining() <= 0: break
        fit_all.append(evaluate(all_pts[i]))
    fit_all = np.array(fit_all)
    all_pts = all_pts[:len(fit_all)]
    
    idx = np.argsort(fit_all)
    pop = all_pts[idx[:pop_size]]
    fit = fit_all[idx[:pop_size]]
    
    # Phase 2: SHADE-DE with LPSR
    if remaining() > max_time * 0.45:
        de_time = remaining() - max_time * 0.45
        pop, fit = shade_de(pop, fit, de_time, min_pop=max(6, dim))
    
    # Phase 3: CMA-ES restarts
    restart_count = 0
    while remaining() > max_time * 0.10:
        restart_count += 1
        cma_time = min(remaining() - max_time * 0.08, remaining() * 0.45)
        if cma_time < 0.3: break
        
        if restart_count == 1:
            x0 = best_params.copy()
            s0 = np.max(ranges) * 0.1
        elif restart_count == 2:
            x0 = best_params + np.random.randn(dim) * ranges * 0.02
            x0 = clip(x0)
            s0 = np.max(ranges) * 0.03
        elif restart_count <= 4:
            x0 = best_params + np.random.randn(dim) * ranges * 0.1
            x0 = clip(x0)
            s0 = np.max(ranges) * 0.1
        else:
            x0 = lower + np.random.random(dim) * ranges
            s0 = np.max(ranges) * 0.3
        
        cmaes_search(x0, s0, cma_time)
    
    # Phase 4: Nelder-Mead polish
    if best_params is not None and remaining() > 0.5:
        nelder_mead(best_params.copy(), initial_step=0.01, max_nm_time=remaining() * 0.5)
    
    if best_params is not None and remaining() > 0.3:
        nelder_mead(best_params.copy(), initial_step=0.001, max_nm_time=remaining())
    
    return best
#
#Key improvements over previous version:
#1. **Fixed pbest selection** - now properly uses sorted indices in SHADE-DE
#2. **Linear Population Size Reduction (L-SHADE)** - reduces population during DE, focusing search
#3. **Proper Nelder-Mead** with all operations (reflection, expansion, inside/outside contraction, shrink)
#4. **Better time allocation** - more time for CMA-ES and local search
#5. **Larger initial sampling** (250 vs 200) with LHS + opposition
#6. **Higher initial CR (0.8)** in SHADE for better convergence on correlated problems
#7. **Bounce-back boundary handling** in CMA-ES instead of simple clipping
#8. **Multiple CMA-ES restarts** with progressively different strategies
#9. **Better convergence detection** in CMA-ES (shorter stagnation threshold)
