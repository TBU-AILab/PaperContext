#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **The SHADE DE has bugs** - pbest_idx selection doesn't use sorted indices properly
#2. **CMA-ES restarts strategy is suboptimal** - needs IPOP-style increasing population
#3. **Time allocation could be better** - more aggressive local search at the end
#4. **Missing Nelder-Mead** which is often very effective for final polishing
#5. **Better diversity mechanisms needed**
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
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += initial_step * ranges[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.array([evaluate(simplex[i]) for i in range(n + 1)])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        while (elapsed() - nm_start) < max_nm_time and remaining() > 0:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if remaining() <= 0:
                return simplex[0], f_simplex[0]
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                            if remaining() <= 0:
                                order2 = np.argsort(f_simplex)
                                return simplex[order2[0]], f_simplex[order2[0]]
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[-1]))
                    fc = evaluate(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                            if remaining() <= 0:
                                order2 = np.argsort(f_simplex)
                                return simplex[order2[0]], f_simplex[order2[0]]
            
            # Convergence check
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-16:
                break
        
        order = np.argsort(f_simplex)
        return simplex[order[0]], f_simplex[order[0]]

    def cmaes_search(x0, sigma0, max_cma_time, lam_mult=1):
        nonlocal best, best_params
        cma_start = elapsed()
        n = dim
        mean = clip(x0.copy())
        sigma = sigma0
        lam = max(4 + int(3 * np.log(n)), (4 + int(3 * np.log(n))) * lam_mult)
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
        best_cma_x = mean.copy()
        stag = 0
        
        while (elapsed() - cma_start) < max_cma_time and remaining() > 0:
            gen += 1
            
            if eigen_update >= max(1, int(1.0/(c1 + cmu_val + 1e-30)/n/10)):
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
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Bounce-back
                for d in range(n):
                    while arx[k][d] < lower[d] or arx[k][d] > upper[d]:
                        if arx[k][d] < lower[d]:
                            arx[k][d] = 2*lower[d] - arx[k][d]
                        if arx[k][d] > upper[d]:
                            arx[k][d] = 2*upper[d] - arx[k][d]
                arx[k] = clip(arx[k])
            
            fit = np.zeros(lam)
            for k in range(lam):
                if remaining() <= 0:
                    return best_cma_x, best_cma
                fit[k] = evaluate(arx[k])
            
            idx = np.argsort(fit)
            if fit[idx[0]] < best_cma:
                best_cma = fit[idx[0]]
                best_cma_x = arx[idx[0]].copy()
                stag = 0
            else:
                stag += 1
            
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
            
            eigen_update += 1
            
            if sigma < 1e-17 or stag > 50 + 20*n:
                break
            if np.max(D) > 1e8 * np.min(D):
                break
        
        return best_cma_x, best_cma

    def shade_de(population, fitness, max_de_time):
        nonlocal best, best_params
        de_start = elapsed()
        pop_size = len(population)
        H = 100
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.9)
        hist_pos = 0
        archive = []
        archive_max = pop_size
        stag = 0
        prev_b = best
        gen = 0
        
        while (elapsed() - de_start) < max_de_time and remaining() > 0:
            gen += 1
            SF, SCR, S_delta = [], [], []
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            # Sort for pbest selection
            sorted_idx = np.argsort(fitness)
            
            for i in range(pop_size):
                if remaining() <= 0 or (elapsed() - de_start) >= max_de_time:
                    idx2 = np.argsort(new_fit)
                    return new_pop[idx2], new_fit[idx2]
                
                ri = np.random.randint(0, H)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
                    if Fi > 1: Fi = 1.0
                CRi = np.clip(np.random.normal(MCR[ri], 0.1), 0.0, 1.0)
                
                # pbest from sorted
                p = max(2, int(np.random.uniform(0.05, 0.20) * pop_size))
                pbest_rank = np.random.randint(0, p)
                pbest_idx = sorted_idx[pbest_rank]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                combined_size = pop_size + len(archive)
                r2 = np.random.randint(0, combined_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, combined_size)
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
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            if abs(best - prev_b) < 1e-14:
                stag += 1
            else:
                stag = 0; prev_b = best
            
            # Linear population size reduction
            min_pop = max(5, dim)
            new_size = int(round(pop_size - gen * (pop_size - min_pop) / max(1, (max_de_time * pop_size / 0.01))))
            new_size = max(min_pop, min(new_size, pop_size))
            if new_size < len(population):
                population = population[:new_size]
                fitness = fitness[:new_size]
                pop_size = new_size
            
            if stag > 20 + dim:
                n_replace = max(1, len(population) // 3)
                for k in range(n_replace):
                    if remaining() <= 0: break
                    ri2 = len(population) - 1 - k
                    if ri2 <= 0: break
                    if np.random.random() < 0.5:
                        population[ri2] = lower + np.random.random(dim) * ranges
                    else:
                        population[ri2] = best_params + np.random.randn(dim) * ranges * np.random.uniform(0.005, 0.15)
                    population[ri2] = clip(population[ri2])
                    fitness[ri2] = evaluate(population[ri2])
                idx2 = np.argsort(fitness)
                population = population[idx2]
                fitness = fitness[idx2]
                stag = 0
                archive.clear()
        
        idx2 = np.argsort(fitness)
        return population[idx2], fitness[idx2]

    # ==== MAIN FLOW ====
    
    # Phase 1: LHS initial sampling with opposition
    pop_size = min(200, max(20, 10 * dim))
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    opp = lower + upper - pop
    all_pop = np.vstack([pop, opp])
    
    all_fit = np.zeros(len(all_pop))
    for i in range(len(all_pop)):
        if remaining() <= 0:
            break
        all_fit[i] = evaluate(all_pop[i])
    
    valid = min(i + 1, len(all_pop))
    all_pop = all_pop[:valid]
    all_fit = all_fit[:valid]
    
    idx = np.argsort(all_fit)
    pop = all_pop[idx[:pop_size]]
    fit = all_fit[idx[:pop_size]]
    
    # Phase 2: SHADE DE - use ~45% of time
    if remaining() > max_time * 0.4:
        de_time = remaining() - max_time * 0.45
        de_time = max(de_time, remaining() * 0.4)
        pop, fit = shade_de(pop, fit, de_time)
    
    # Phase 3: CMA-ES with IPOP restarts - use ~40% of time
    restart_count = 0
    lam_mult = 1
    while remaining() > max_time * 0.10:
        restart_count += 1
        cma_time = min(remaining() - max_time * 0.08, remaining() * 0.45)
        if cma_time < 0.3: break
        
        if restart_count == 1:
            x0 = best_params.copy()
            s0 = np.max(ranges) * 0.15
        elif restart_count == 2:
            x0 = best_params + np.random.randn(dim) * ranges * 0.02
            x0 = clip(x0)
            s0 = np.max(ranges) * 0.05
        else:
            # IPOP: increase population
            lam_mult *= 2
            if np.random.random() < 0.3:
                x0 = lower + np.random.random(dim) * ranges
                s0 = np.max(ranges) * 0.3
            else:
                x0 = best_params + np.random.randn(dim) * ranges * 0.1
                x0 = clip(x0)
                s0 = np.max(ranges) * 0.1
        
        cmaes_search(x0, s0, cma_time, min(lam_mult, 8))
    
    # Phase 4: Nelder-Mead polish with multiple starts
    if best_params is not None and remaining() > 0.5:
        nm_time = remaining() * 0.6
        nelder_mead(best_params.copy(), 0.02, nm_time)
    
    if best_params is not None and remaining() > 0.3:
        nm_time = remaining() * 0.7
        nelder_mead(best_params.copy(), 0.002, nm_time)
    
    # Phase 5: Fine pattern search
    if best_params is not None and remaining() > 0.1:
        x = best_params.copy()
        fx = best
        step = 0.001
        while step > 1e-15 and remaining() > 0.05:
            improved = False
            for i in range(dim):
                if remaining() <= 0.05:
                    break
                for sign in [1.0, -1.0]:
                    trial = x.copy()
                    trial[i] += sign * step * ranges[i]
                    trial = clip(trial)
                    ft = evaluate(trial)
                    if ft < fx:
                        fx = ft
                        x = trial
                        improved = True
                        break
            if not improved:
                step *= 0.5
    
    return best
#
#Key improvements over previous version:
#1. **Fixed SHADE pbest selection** - now properly uses sorted indices
#2. **IPOP-CMA-ES** - increasing population on restarts for better global exploration
#3. **Bounce-back boundary handling** in CMA-ES instead of simple clipping
#4. **Linear population size reduction** in SHADE (L-SHADE style)
#5. **Better time allocation** - balanced between DE, CMA-ES, and polishing
#6. **Multi-scale Nelder-Mead** - two rounds with decreasing simplex sizes
#7. **Fine pattern search** as final polish
#8. **Proper convergence detection** and restart triggering
