#Looking at the current algorithm, I can identify several issues:
#
#1. **Time allocation is suboptimal** - too much time on SHADE which may not converge well
#2. **CMA-ES eigendecomposition frequency** is off
#3. **No multi-start diversity** - needs better exploration
#4. **Cache overhead** with rounding may cause issues
#5. **Missing adaptive population sizing** and better restart strategies
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
    center = (upper + lower) / 2.0
    
    n_evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_func(x):
        nonlocal best, best_params, n_evals
        x = np.clip(x, lower, upper)
        f = func(x)
        n_evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def mirror_bound(x):
        """Mirror boundary handling"""
        result = x.copy()
        for d in range(dim):
            while result[d] < lower[d] or result[d] > upper[d]:
                if result[d] < lower[d]:
                    result[d] = 2 * lower[d] - result[d]
                if result[d] > upper[d]:
                    result[d] = 2 * upper[d] - result[d]
        return np.clip(result, lower, upper)
    
    # === CMA-ES Implementation ===
    def cmaes(x0, sigma0, max_seconds, pop_scale=1.0):
        nonlocal best, best_params
        t0 = elapsed()
        n = dim
        
        lam = int((4 + int(3 * np.log(n))) * pop_scale)
        lam = max(lam, 6)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_sep = (n > 60)
        
        if use_sep:
            diagC = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_update = 0
        
        gen = 0
        best_gen_f = float('inf')
        stag = 0
        f_history = []
        
        while True:
            if elapsed() - t0 >= max_seconds or time_left() <= 0:
                return
            
            # Update eigensystem
            if not use_sep:
                needs_update = (gen == 0) or (gen - eigen_update > 1.0 / (c1 + cmu) / n / 5.0)
                if needs_update:
                    try:
                        C = np.triu(C) + np.triu(C, 1).T
                        evals, evecs = np.linalg.eigh(C)
                        evals = np.maximum(evals, 1e-20)
                        D = np.sqrt(evals)
                        B = evecs
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                        eigen_update = gen
                    except:
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
                        invsqrtC = np.eye(n)
            
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                if use_sep:
                    arx[k] = mean + sigma * np.sqrt(diagC) * arz[k]
                else:
                    arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], lower, upper)
            
            # Evaluate
            fit = np.empty(lam)
            for k in range(lam):
                if elapsed() - t0 >= max_seconds or time_left() <= 0:
                    return
                fit[k] = eval_func(arx[k])
            
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.dot(weights, arx[:mu])
            
            zmean = np.dot(weights, arz[:mu])
            
            if use_sep:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ((mean - old_mean) / sigma))
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((mean - old_mean) / sigma)
            
            if use_sep:
                artmp = (arx[:mu] - old_mean) / sigma
                diagC = ((1 - c1 - cmu) * diagC +
                         c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diagC) +
                         cmu * np.sum(weights[:, None] * artmp ** 2, axis=0))
                diagC = np.maximum(diagC, 1e-20)
            else:
                artmp = (arx[:mu] - old_mean) / sigma
                C = ((1 - c1 - cmu) * C +
                     c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                     cmu * (weights[:, None] * artmp).T @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            
            # Stagnation
            cur_best = fit[idx[0]]
            f_history.append(cur_best)
            if cur_best < best_gen_f - 1e-12 * abs(best_gen_f + 1e-30):
                best_gen_f = cur_best
                stag = 0
            else:
                stag += 1
            
            # Check convergence
            if sigma < 1e-16:
                return
            if stag > 30 + 10 * n:
                return
            if len(f_history) > 20:
                recent = f_history[-20:]
                if max(recent) - min(recent) < 1e-14 * (abs(min(recent)) + 1e-30):
                    return
    
    # === SHADE ===
    def shade(budget_seconds, pop=None, fit_pop=None):
        nonlocal best, best_params
        t0 = elapsed()
        
        pop_size = min(max(7 * dim, 50), 300)
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        if pop is None:
            pop = np.random.uniform(lower, upper, (pop_size, dim))
            fit_pop = np.full(pop_size, float('inf'))
            # Seed best solutions
            if best_params is not None:
                pop[0] = best_params.copy()
            for i in range(pop_size):
                if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                    return pop, fit_pop
                fit_pop[i] = eval_func(pop[i])
        else:
            pop_size = len(pop)
        
        archive = []
        
        gen = 0
        while True:
            if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                return pop, fit_pop
            
            S_F, S_CR, S_delta = [], [], []
            
            # Sort for pbest
            sort_idx = np.argsort(fit_pop)
            
            for i in range(pop_size):
                if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                    return pop, fit_pop
                
                ri = np.random.randint(H)
                
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    if Fi > 1: Fi = 1.0
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                # pbest
                p_count = max(2, int(max(0.05, 0.2 - 0.15 * gen / max(1, gen + 50)) * pop_size))
                pbest = pop[sort_idx[np.random.randint(p_count)]]
                
                candidates = [j for j in range(pop_size) if j != i]
                r1 = np.random.choice(candidates)
                
                # r2 from pop+archive
                all_pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                r2_cands = [j for j in all_pool if j != i and j != r1]
                if not r2_cands:
                    r2_cands = [j for j in range(pop_size) if j != i]
                r2 = np.random.choice(r2_cands)
                if r2 < pop_size:
                    xr2 = pop[r2]
                else:
                    xr2 = archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pbest - pop[i]) + Fi * (pop[r1] - xr2)
                
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Bounce-back
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                trial = np.clip(trial, lower, upper)
                
                f_trial = eval_func(trial)
                
                if f_trial < fit_pop[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(fit_pop[i] - f_trial)
                    archive.append(pop[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial
                    fit_pop[i] = f_trial
                elif f_trial == fit_pop[i]:
                    pop[i] = trial
                    fit_pop[i] = f_trial
            
            if S_F:
                S_delta = np.array(S_delta)
                w = S_delta / (np.sum(S_delta) + 1e-30)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
            gen += 1
        
        return pop, fit_pop
    
    # === Nelder-Mead ===
    def nelder_mead(x0, f0, budget_seconds, scale_factor=0.05):
        nonlocal best, best_params
        t0 = elapsed()
        n = dim
        
        simplex = np.zeros((n + 1, n))
        fvals = np.full(n + 1, float('inf'))
        simplex[0] = x0.copy()
        fvals[0] = f0
        
        step = ranges * scale_factor
        for i in range(n):
            if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                return
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] = np.clip(x0[i] + step[i], lower[i], upper[i])
            if abs(simplex[i + 1][i] - x0[i]) < 1e-15:
                simplex[i + 1][i] = np.clip(x0[i] - step[i], lower[i], upper[i])
            fvals[i + 1] = eval_func(simplex[i + 1])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(100000):
            if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                return
            
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Convergence check
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                return
            if fvals[-1] - fvals[0] < 1e-15:
                return
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
            elif fr < fvals[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                if elapsed() - t0 >= budget_seconds:
                    simplex[-1] = xr; fvals[-1] = fr; return
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc; fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            if elapsed() - t0 >= budget_seconds:
                                return
                            fvals[i] = eval_func(simplex[i])
                else:
                    xcc = np.clip(centroid - rho * (centroid - simplex[-1]), lower, upper)
                    fcc = eval_func(xcc)
                    if fcc < fvals[-1]:
                        simplex[-1] = xcc; fvals[-1] = fcc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            if elapsed() - t0 >= budget_seconds:
                                return
                            fvals[i] = eval_func(simplex[i])
    
    # === Coordinate Descent with Golden Section ===
    def golden_section_search(x0, budget_seconds):
        nonlocal best, best_params
        t0 = elapsed()
        gr = (np.sqrt(5) - 1) / 2
        
        x = x0.copy()
        fx = eval_func(x)
        
        for _round in range(10):
            improved = False
            for d in range(dim):
                if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                    return
                
                a, b = lower[d], upper[d]
                # Narrow the search around current point
                width = min(ranges[d] * 0.5 / (1 + _round), (b - a))
                a = max(lower[d], x[d] - width)
                b = min(upper[d], x[d] + width)
                
                if b - a < 1e-15:
                    continue
                
                c = b - gr * (b - a)
                dd = a + gr * (b - a)
                
                xc = x.copy(); xc[d] = c
                xd = x.copy(); xd[d] = dd
                fc = eval_func(xc)
                fd = eval_func(xd)
                
                for _ in range(20):
                    if elapsed() - t0 >= budget_seconds or time_left() <= 0:
                        break
                    if b - a < 1e-13:
                        break
                    
                    if fc < fd:
                        b = dd
                        dd = c; fd = fc
                        c = b - gr * (b - a)
                        xc = x.copy(); xc[d] = c
                        fc = eval_func(xc)
                    else:
                        a = c
                        c = dd; fc = fd
                        dd = a + gr * (b - a)
                        xd = x.copy(); xd[d] = dd
                        fd = eval_func(xd)
                
                best_d = c if fc < fd else dd
                best_fd = min(fc, fd)
                if best_fd < fx:
                    x[d] = best_d
                    fx = best_fd
                    improved = True
            
            if not improved:
                break
    
    # === LHS Initialization ===
    n_init = min(max(20 * dim, 100), 500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fit = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.1:
            n_init = i
            break
        init_fit[i] = eval_func(init_pop[i])
    
    if n_init == 0:
        return best
    
    # === Main Orchestration ===
    remaining = time_left()
    
    # Phase 1: SHADE (35% of time)
    if remaining > 1:
        shade(remaining * 0.35)
    
    # Phase 2: CMA-ES with IPOP restarts (35% of time)
    remaining = time_left()
    if remaining > 1:
        cma_budget = remaining * 0.50
        cma_t0 = elapsed()
        restart = 0
        pop_mult = 1.0
        while elapsed() - cma_t0 < cma_budget and time_left() > 0.3:
            sub_budget = min(cma_budget - (elapsed() - cma_t0), cma_budget / 3)
            if sub_budget < 0.2:
                break
            
            if restart == 0 and best_params is not None:
                sp = best_params.copy()
                sig = np.mean(ranges) * 0.25
            elif best_params is not None and np.random.random() < 0.6:
                sp = best_params + np.random.randn(dim) * ranges * (0.2 + 0.15 * restart)
                sp = np.clip(sp, lower, upper)
                sig = np.mean(ranges) * (0.3 + 0.1 * restart)
            else:
                sp = np.random.uniform(lower, upper)
                sig = np.mean(ranges) * 0.4
            
            cmaes(sp, sig, sub_budget, pop_scale=pop_mult)
            restart += 1
            pop_mult *= 1.5  # IPOP
    
    # Phase 3: Nelder-Mead
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        nelder_mead(best_params.copy(), best, remaining * 0.4, scale_factor=0.05)
    
    # Phase 4: Golden section coordinate descent
    remaining = time_left()
    if remaining > 0.5 and best_params is not None:
        golden_section_search(best_params.copy(), remaining * 0.5)
    
    # Phase 5: Fine Nelder-Mead
    remaining = time_left()
    if remaining > 0.3 and best_params is not None:
        nelder_mead(best_params.copy(), best, remaining * 0.7, scale_factor=0.005)
    
    # Phase 6: Very fine coordinate descent
    remaining = time_left()
    if remaining > 0.2 and best_params is not None:
        golden_section_search(best_params.copy(), remaining * 0.9)
    
    return best
