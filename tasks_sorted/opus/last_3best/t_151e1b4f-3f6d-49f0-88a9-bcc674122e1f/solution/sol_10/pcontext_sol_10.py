#Looking at the results, algorithm 1 (score 12.05) is the best. I'll improve it by:
#
#1. **Better time allocation** - more time for CMA-ES which is strong for local search
#2. **Multiple CMA-ES restarts** with different starting points (not just best)
#3. **Improved SHADE** with better archive handling
#4. **Adaptive restart strategy** using top-k solutions from DE phase
#5. **More efficient Nelder-Mead** with adaptive step sizes
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Store top solutions for multi-start
    top_solutions = []
    top_max = 10
    
    def record_top(x, f):
        nonlocal top_solutions
        x = clip(x)
        top_solutions.append((f, x.copy()))
        top_solutions.sort(key=lambda t: t[0])
        if len(top_solutions) > top_max:
            top_solutions = top_solutions[:top_max]

    # --- Phase 1: LHS initialization ---
    pop_size = min(max(30, 8 * dim), 200)
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.06:
            break
        fitness[i] = ev(population[i])
        record_top(population[i], fitness[i])
    
    # Opposition-based initialization
    for i in range(pop_size):
        if elapsed() >= max_time * 0.10:
            break
        opp = lower + upper - population[i]
        of = ev(opp)
        if of < fitness[i]:
            population[i] = clip(opp)
            fitness[i] = of
        record_top(population[i], fitness[i])

    # --- Phase 2: L-SHADE ---
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    init_pop_size = pop_size
    min_pop_size = max(4, dim)
    
    de_time_limit = max_time * 0.55
    gen = 0

    while elapsed() < de_time_limit:
        gen += 1
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.11 * pop_size))
        
        S_F = []
        S_CR = []
        delta_f = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= de_time_limit:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 30:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            pb = sorted_idx[np.random.randint(p_best_size)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = idxs[np.random.randint(len(idxs))]
            
            pool = pop_size + len(archive)
            r2 = np.random.randint(pool - 1)
            candidates = [j for j in range(pool) if j != i and j != r1]
            if candidates:
                r2v = candidates[np.random.randint(len(candidates))]
            else:
                r2v = r1
            x_r2 = population[r2v] if r2v < pop_size else archive[r2v - pop_size]
            
            mutant = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - x_r2)
            
            # Bounce-back boundary handling
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            for dd in range(dim):
                if trial[dd] < lower[dd]:
                    trial[dd] = (lower[dd] + population[i][dd]) / 2.0
                elif trial[dd] > upper[dd]:
                    trial[dd] = (upper[dd] + population[i][dd]) / 2.0
            
            trial_f = ev(trial)
            record_top(trial, trial_f)
            
            if trial_f < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                delta_f.append(fitness[i] - trial_f)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = trial_f
        
        population = new_pop
        fitness = new_fit
            
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / de_time_limit)))
        if new_pop_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_pop_size]]
            fitness = fitness[si[:new_pop_size]]
            pop_size = new_pop_size
            max_archive = pop_size
            if len(archive) > max_archive:
                archive = archive[:max_archive]
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 15:
            n_replace = max(1, pop_size // 3)
            si = np.argsort(fitness)
            for ii in range(n_replace):
                idx = si[-(ii + 1)]
                if np.random.random() < 0.5:
                    population[idx] = best_params + 0.15 * ranges * np.random.randn(dim)
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip(population[idx])
                if elapsed() >= de_time_limit:
                    break
                fitness[idx] = ev(population[idx])
                record_top(population[idx], fitness[idx])
            stagnation = 0

    # --- Phase 3: CMA-ES with multi-restart ---
    def run_cmaes(x_start, sigma_init, time_limit):
        n = dim
        lam = max(4 + int(3 * np.log(n)), 10)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        mean = x_start.copy()
        sigma = sigma_init
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        count_eval = 0
        
        while elapsed() < time_limit:
            if count_eval - eigeneval > lam / (c1 + cmu_v + 1e-30) / n / 10:
                eigeneval = count_eval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            
            for j in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                arx[j] = clip(mean + sigma * (B @ (D * z)))
                arf[j] = ev(arx[j])
                count_eval += 1
            
            idx_sort = np.argsort(arf)
            old_mean = mean.copy()
            
            mean = np.zeros(n)
            for j in range(mu):
                mean += weights[j] * arx[idx_sort[j]]
            mean = clip(mean)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * count_eval / lam + 1e-30)) < (1.4 + 2 / (n + 1)) * chiN else 0.0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for j in range(mu):
                dv = (arx[idx_sort[j]] - old_mean) / (sigma + 1e-30)
                C += cmu_v * weights[j] * np.outer(dv, dv)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(1e-15, min(sigma, max(ranges)))
            
            if sigma < 1e-13:
                break
            if np.max(D) > 1e7 * np.min(D):
                break
    
    # Multi-restart CMA-ES from top solutions
    cma_end = max_time * 0.90
    restart_idx = 0
    sigmas = [0.1, 0.05, 0.02, 0.15, 0.01]
    
    while elapsed() < cma_end and restart_idx < len(top_solutions) + 3:
        remaining = cma_end - elapsed()
        if remaining < 0.5:
            break
        
        if restart_idx < len(top_solutions):
            x_start = top_solutions[restart_idx][1].copy()
        else:
            x_start = best_params.copy()
        
        sig = sigmas[restart_idx % len(sigmas)] * np.mean(ranges)
        time_budget = min(remaining, max(remaining / max(1, len(top_solutions) - restart_idx), 1.0))
        run_cmaes(x_start, sig, elapsed() + time_budget)
        restart_idx += 1

    # --- Phase 4: Nelder-Mead from best ---
    if best_params is not None and elapsed() < max_time * 0.98:
        x0 = best_params.copy()
        n = dim
        scale = 0.005 * ranges
        simplex = np.vstack([x0] + [x0 + scale[j] * np.eye(n)[j] for j in range(n)])
        fs = np.array([ev(simplex[j]) for j in range(n + 1) if elapsed() < max_time * 0.98] + [float('inf')] * max(0, n + 1 - sum(1 for _ in range(n+1))))
        
        if len(fs) == n + 1:
            while elapsed() < max_time * 0.98:
                o = np.argsort(fs)
                simplex = simplex[o]
                fs = fs[o]
                c = np.mean(simplex[:-1], axis=0)
                
                xr = clip(c + (c - simplex[-1]))
                fr = ev(xr)
                if fs[0] <= fr < fs[-2]:
                    simplex[-1] = xr; fs[-1] = fr
                elif fr < fs[0]:
                    xe = clip(c + 2 * (xr - c))
                    fe = ev(xe)
                    simplex[-1] = xe if fe < fr else xr
                    fs[-1] = min(fe, fr)
                else:
                    xc = clip(c + 0.5 * (simplex[-1] -
