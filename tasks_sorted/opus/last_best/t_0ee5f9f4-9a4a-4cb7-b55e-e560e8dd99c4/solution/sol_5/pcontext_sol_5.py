#Looking at the previous algorithm, I'll focus on several improvements:
#
#1. **Better time allocation** - DE phase takes too long (55%), reduce it and give more to CMA-ES
#2. **Vectorize DE operations** - avoid Python loops for crossover/mutation
#3. **Improve CMA-ES restart strategy** - use IPOP with better sigma adaptation
#4. **Add multiple Nelder-Mead restarts from diverse good solutions**
#5. **Better stagnation detection and parameter adaptation**
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.98):
        return elapsed() < max_time * fraction
    
    def eval_func(x):
        nonlocal best, best_params, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def eval_batch(xs):
        fs = np.array([eval_func(x) for x in xs])
        return fs

    # Phase 1: Sobol-like quasi-random + LHS sampling
    n_init = min(max(15 * dim, 80), 400)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    actual_init = 0
    for i in range(n_init):
        if not time_ok(0.15):
            break
        init_fitness[i] = eval_func(init_pop[i])
        actual_init = i + 1
    
    sorted_idx = np.argsort(init_fitness[:actual_init])
    
    # Archive
    archive_size = min(50, actual_init)
    archive = init_pop[sorted_idx[:archive_size]].copy()
    archive_fit = init_fitness[sorted_idx[:archive_size]].copy()
    
    def update_archive(x, f):
        nonlocal archive, archive_fit
        worst_idx = np.argmax(archive_fit)
        if f < archive_fit[worst_idx]:
            archive[worst_idx] = x.copy()
            archive_fit[worst_idx] = f
    
    # Phase 2: SHADE Differential Evolution
    pop_size = min(max(6 * dim, 30), 80)
    if actual_init >= pop_size:
        pop = init_pop[sorted_idx[:pop_size]].copy()
        pop_fit = init_fitness[sorted_idx[:pop_size]].copy()
    else:
        pop = np.zeros((pop_size, dim))
        pop_fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if i < actual_init:
                pop[i] = init_pop[sorted_idx[i]].copy()
                pop_fit[i] = init_fitness[sorted_idx[i]]
            else:
                pop[i] = lower + np.random.rand(dim) * ranges
                if time_ok(0.20):
                    pop_fit[i] = eval_func(pop[i])
    
    mem_size = 20
    mem_F = np.full(mem_size, 0.5)
    mem_CR = np.full(mem_size, 0.5)
    mem_idx = 0
    
    de_gens = 0
    while time_ok(0.40):
        S_F, S_CR, S_df = [], [], []
        
        p_best_rate = max(0.05, 0.25 - 0.20 * de_gens / (de_gens + 50))
        p_best_size = max(2, int(pop_size * p_best_rate))
        sorted_pop_idx = np.argsort(pop_fit)
        
        # Generate all parameters at once
        ri_all = np.random.randint(mem_size, size=pop_size)
        Fi_all = np.clip(np.random.standard_cauchy(pop_size) * 0.1 + mem_F[ri_all], 0.01, 1.0)
        CRi_all = np.clip(np.random.randn(pop_size) * 0.1 + mem_CR[ri_all], 0, 1)
        
        for i in range(pop_size):
            if not time_ok(0.40):
                break
            
            Fi = Fi_all[i]
            CRi = CRi_all[i]
            
            p_best_idx = sorted_pop_idx[np.random.randint(p_best_size)]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size - 2)
            if r2 >= i: r2 += 1
            if r2 >= r1 + (1 if r1 < i else 0): r2 += 1
            r2 = r2 % combined_size
            
            if r2 < pop_size:
                x_r2 = pop[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            mutant = pop[i] + Fi * (pop[p_best_idx] - pop[i]) + Fi * (pop[r1] - x_r2)
            
            mask = np.random.rand(dim) < CRi
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            trial = np.where(mask, mutant, pop[i])
            
            # Bounce-back
            below = trial < lower
            above = trial > upper
            if np.any(below):
                trial[below] = lower[below] + np.random.rand(np.sum(below)) * (pop[i][below] - lower[below])
            if np.any(above):
                trial[above] = upper[above] - np.random.rand(np.sum(above)) * (upper[above] - pop[i][above])
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_func(trial)
            update_archive(trial, f_trial)
            
            if f_trial <= pop_fit[i]:
                df = pop_fit[i] - f_trial
                if df > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(df)
                pop[i] = trial
                pop_fit[i] = f_trial
        
        if len(S_F) > 0:
            w = np.array(S_df)
            w = w / (np.sum(w) + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            mem_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            mem_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % mem_size
        
        de_gens += 1
    
    # Phase 3: CMA-ES with IPOP restarts
    def cma_es(x0, sigma0, lam=None, time_limit=0.96):
        n = dim
        if lam is None:
            lam = 4 + int(3 * np.log(n))
        lam = max(lam, 6)
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
        
        sigma = sigma0
        xmean = x0.copy()
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        eigeneval = 0
        counteval = 0
        gen = 0
        stag_count = 0
        best_in_run = float('inf')
        flat_count = 0
        
        while time_ok(time_limit):
            arx = np.zeros((lam, n))
            arz = np.random.randn(lam, n)
            
            for k in range(lam):
                arx[k] = xmean + sigma * (B @ (D * arz[k]))
            
            # Repair boundaries
            for k in range(lam):
                for dd in range(n):
                    attempts = 0
                    while (arx[k, dd] < lower[dd] or arx[k, dd] > upper[dd]) and attempts < 10:
                        if arx[k, dd] < lower[dd]:
                            arx[k, dd] = 2 * lower[dd] - arx[k, dd]
                        if arx[k, dd] > upper[dd]:
                            arx[k, dd] = 2 * upper[dd] - arx[k, dd]
                        attempts += 1
                arx[k] = np.clip(arx[k], lower, upper)
            
            arfitness = np.full(lam, float('inf'))
            for k in range(lam):
                if not time_ok(time_limit):
                    return
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            for idx in arindex[:min(3, lam)]:
                update_archive(arx[idx], arfitness[idx])
            
            gen_best = arfitness[arindex[0]]
            if gen_best < best_in_run - 1e-10 * (abs(best_in_run) + 1e-10):
                best_in_run = gen_best
                stag_count = 0
                flat_count = 0
            else:
                stag_count += 1
            
            # Check flat fitness
            if arfitness[arindex[0]] == arfitness[arindex[min(lam-1, mu)]]:
                flat_count += 1
            else:
                flat_count = 0
            
            xold = xmean.copy()
            selected = arx[arindex[:mu]]
            xmean = np.sum(weights[:, None] * selected, axis=0)
            
            diff = (xmean - xold) / (sigma + 1e-30)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            
            norm_ps = np.linalg.norm(ps)
            hsig = float(norm_ps / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (selected - xold) / (sigma + 1e-30)
            
            C = ((1 - c1 - cmu_val) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu_val * (weights[:, None] * artmp).T @ artmp)
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                return
            if stag_count > 15 + 30 * n / lam:
                return
            if np.max(D) > 1e7 * np.min(D):
                return
            if flat_count > 5:
                return
    
    base_lam = 4 + int(3 * np.log(dim))
    restart = 0
    ipop_factor = 1
    
    all_candidates = np.vstack([pop, archive])
    all_fits = np.concatenate([pop_fit, archive_fit])
    cand_order = np.argsort(all_fits)
    n_top = min(5, len(cand_order))
    
    while time_ok(0.90):
        if restart < n_top:
            x0 = all_candidates[cand_order[restart]].copy()
            sigma0 = 0.15 * np.mean(ranges)
            lam_r = base_lam
        elif restart % 3 == 0:
            ipop_factor *= 2
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.35 * np.mean(ranges)
            lam_r = min(base_lam * ipop_factor, 256)
        elif restart % 3 == 1:
            x0 = best_params.copy() + 0.02 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.05 * np.mean(ranges)
            lam_r = base_lam
        else:
            idx_a = np.random.randint(min(10, len(archive)))
            order_a = np.argsort(archive_fit)
            x0 = archive[order_a[idx_a]].copy() + 0.05 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.1 * np.mean(ranges)
            lam_r = base_lam
        
        cma_es(x0, sigma0, lam_r, 0.92)
        restart += 1
    
    # Phase 4: Nelder-Mead polish from best
    def nelder_mead(x_start, step_scale, time_frac):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex_f = np.full(n + 1, float('inf'))
        simplex[0] = x_start.copy()
        simplex_f[0] = eval_func(x_start)
        
        step = step_scale * ranges
        for i in range(n):
            if not time_ok(time_frac):
                return
            simplex[i+1] = x_start.copy()
            simplex[i+1][i] = np.clip(simplex[i+1][i] + step[i], lower[i], upper[i])
            simplex_f[i+1] = eval_func(simplex[i+1])
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(100000):
            if not time_ok(time_frac):
                return
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if fr < simplex_f[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                if fr < simplex_f[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1], simplex_f[-1] = xc, fc
                        continue
                else:
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), lower, upper)
                    fc = eval_func(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = xc, fc
                        continue
                # Shrink
                for i in range(1, n+1):
                    if not time_ok(time_frac):
                        return
                    simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    simplex_f[i] = eval_func(simplex[i])
            
            diam = np.max(np.abs(simplex[-1] - simplex[0]))
            if diam < 1e-14:
                return
    
    if best_params is not None and time_ok(0.93):
        nelder_mead(best_params, 0.005, 0.97)
    
    # Second NM with smaller step
    if best_params is not None and time_ok(0.97):
        nelder_mead(best_params, 0.0005, 0.995)
    
    return best
