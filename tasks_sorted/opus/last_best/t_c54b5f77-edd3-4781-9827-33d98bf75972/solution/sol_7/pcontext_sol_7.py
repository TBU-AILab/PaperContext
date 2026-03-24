#Looking at the performance gap (best: 8.096 vs last: 26.98), I need to analyze what made the best algorithm work well and improve upon it. Key observations:
#
#1. The best algorithm had good L-SHADE + CMA-ES + Nelder-Mead pipeline
#2. I need better CMA-ES implementation (proper invsqrtC computation), better time allocation, and more aggressive local search
#3. Adding multiple diverse restarts and better stagnation handling
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    top_solutions = []
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        if len(top_solutions) < 20:
            top_solutions.append((f, x.copy()))
            top_solutions.sort(key=lambda t: t[0])
        elif f < top_solutions[-1][0]:
            top_solutions[-1] = (f, x.copy())
            top_solutions.sort(key=lambda t: t[0])
        return f

    # --- Phase 1: LHS Initialization + Opposition ---
    pop_size = min(max(30, 8 * dim), 200)
    N_init = pop_size
    N_min = 4
    
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.06:
            break
        fitness_vals[i] = eval_f(population[i])
    
    if elapsed() < max_time * 0.10:
        opp_pop = lower + upper - population
        opp_fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= max_time * 0.10:
                break
            if fitness_vals[i] < float('inf'):
                opp_fit[i] = eval_f(opp_pop[i])
        
        all_pop = np.vstack([population, opp_pop])
        all_fit = np.concatenate([fitness_vals, opp_fit])
        valid = all_fit < float('inf')
        if np.sum(valid) >= pop_size:
            idx = np.argsort(all_fit)
            population = all_pop[idx[:pop_size]]
            fitness_vals = all_fit[idx[:pop_size]]
    
    valid = fitness_vals < float('inf')
    if np.sum(valid) < N_min:
        while elapsed() < max_time * 0.95:
            x = lower + np.random.random(dim) * ranges
            eval_f(x)
        return best
    
    population = population[valid]
    fitness_vals = fitness_vals[valid]
    pop_size = len(population)
    
    # --- Phase 2: L-SHADE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k_idx = 0
    archive = []
    archive_max = N_init
    p_min = max(2.0 / pop_size, 0.05)
    p_max = 0.25
    generation = 0
    stagnation_count = 0
    prev_best = best
    shade_time_limit = max_time * 0.45
    shade_start = elapsed()
    
    while elapsed() < shade_time_limit and pop_size >= N_min:
        generation += 1
        sort_idx = np.argsort(fitness_vals)
        population = population[sort_idx]
        fitness_vals = fitness_vals[sort_idx]
        
        S_F, S_CR, S_delta = [], [], []
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= shade_time_limit:
                break
            
            r = np.random.randint(H)
            for _ in range(20):
                Fi = M_F[r] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[r] + 0.1 * np.random.randn(), 0, 1)
            
            p = p_min + np.random.random() * (p_max - p_min)
            p_count = max(1, int(p * pop_size))
            x_pbest = population[np.random.randint(p_count)]
            
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            total = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(total)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            if f_trial <= new_fit[i]:
                if len(archive) < archive_max:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                delta = fitness_vals[i] - f_trial
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_delta.append(delta)
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness_vals = new_fit
        
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[k_idx] = np.sum(weights * scr)
            k_idx = (k_idx + 1) % H
        
        t_ratio = min((elapsed() - shade_start) / (shade_time_limit - shade_start + 1e-30), 1.0)
        new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * t_ratio)))
        if new_pop_size < pop_size:
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx[:new_pop_size]]
            fitness_vals = fitness_vals[sort_idx[:new_pop_size]]
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(prev_best - best) < 1e-15:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 15 + dim:
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx]
            fitness_vals = fitness_vals[sort_idx]
            keep = max(2, pop_size // 5)
            for i in range(keep, pop_size):
                if elapsed() >= shade_time_limit:
                    break
                if np.random.random() < 0.5 and best_x is not None:
                    population[i] = clip(best_x + 0.05 * ranges * np.random.randn(dim))
                else:
                    population[i] = lower + np.random.random(dim) * ranges
                fitness_vals[i] = eval_f(population[i])
            stagnation_count = 0; archive = []

    # --- Phase 3: CMA-ES with IPOP restarts ---
    def run_cmaes(x_start, sigma0, time_limit, lam_override=None):
        nonlocal best, best_x
        n = dim
        x_mean = x_start.copy()
        sigma = sigma0
        base_lam = 4 + int(3 * np.log(n))
        lam = lam_override if lam_override else base_lam
        lam = max(lam, 6)
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w_raw / w_raw.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        pc = np.zeros(n); ps = np.zeros(n)
        use_full = (n <= 80)
        if use_full:
            C = np.eye(n); B = np.eye(n); D_diag = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_interval = max(1, int(1.0 / (10*n*(c1+cmu_v) + 1e-30)))
            eigen_countdown = 0
        else:
            diag_C = np.ones(n)
        
        cma_gen = 0; no_improve = 0; local_best = float('inf')
        
        while elapsed() < time_limit:
            cma_gen += 1
            
            if use_full and eigen_countdown <= 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, B = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D_diag = np.sqrt(eigvals)
                    invsqrtC = B @ np.diag(1.0 / D_diag) @ B.T
                    eigen_countdown = eigen_interval
                except:
                    return
            if use_full:
                eigen_countdown -= 1
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for ki in range(lam):
                if use_full:
                    arx[ki] = x_mean + sigma * (B @ (D_diag * arz[ki]))
                else:
                    arx[ki] = x_mean + sigma * np.sqrt(diag_C) * arz[ki]
                arx[ki] = clip(arx[ki])
            
            fit = np.full(lam, float('inf'))
            for ki in range(lam):
                if elapsed() >= time_limit:
                    return
                fit[ki] = eval_f(arx[ki])
            
            idx = np.argsort(fit)
            x_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * arx[idx[:mu]], axis=0)
            diff = (x_mean - x_old) / (sigma + 1e-30)
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / (np.sqrt(diag_C) + 1e-30)
            
            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1 - (1-cs)**(2*cma_gen)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            if use_full:
                artmp = (arx[idx[:mu]] - x_old) / (sigma + 1e-30)
                rank_mu = np.zeros((n, n))
                for mi in range(mu):
                    rank_mu += w[mi] * np.outer(artmp[mi], artmp[mi])
                C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * rank_mu
            else:
                artmp = (arx[idx[:mu]] - x_old) / (sigma + 1e-30)
                diag_C = (1 - c1 - cmu_v) * diag_C + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diag_C) + cmu_v * np.sum(w[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = min(sigma, np.mean(ranges) * 2)
            if sigma < 1e-16 * np.mean(ranges):
                return
            
            cur_f = fit[idx[0]]
            if cur_f < local_best - 1e-13:
                local_best = cur_f; no_improve = 0
            else:
                no_improve += 1
            if no_improve > 20 + 10 * n / lam:
                return
            if use_full and cma_gen % 20 == 0:
                if D_diag.max() / (D_diag.min() + 1e-30) > 1e14:
                    return

    cma_end = max_time * 0.88
    base_lam = 4 + int(3 * np.log(dim))
    
    sigmas = [0.001, 0.01, 0.05, 0.15, 0.3]
    for ri, sig_frac in enumerate(sigmas):
        if elapsed() >= cma_end:
            break
        remaining = cma_end - elapsed()
        tl = min(elapsed() + remaining / max(1, len(sigmas) - ri) * 1.3, cma_end)
        xs = best_x.copy() if best_x is not None else lower + 0.5 * ranges
        run_cmaes(xs, sig_frac * np.mean(ranges), tl)
    
    # IPOP restarts
    ipop_lam = base_lam
    for ipop_i in range(8):
        if elapsed() >= cma_end:
            break
        ipop_lam = min(ipop_lam * 2, 400)
        remaining = cma_end - elapsed()
        tl = min(elapsed() + remaining * 0.4, cma_end)
        if best_x is not None and np.random.random() < 0.4:
            xs = clip(best_x + 0.3 * ranges * np.random.randn(dim))
        elif top_solutions and np.random.random() < 0.3:
            xs = top_solutions[np.random.randint(min(5, len(top_solutions)))][1].copy()
        else:
            xs = lower + np.random.random(dim) * ranges
        run_cmaes(xs, 0.25 * np.mean(ranges), tl, ipop_lam)

    # --- Phase 4: Nelder-Mead ---
    if best_x is not None and elapsed() < max_time * 0.96:
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        simplex_f = np.zeros(n + 1)
        simplex_f[0] = best
        
        init_scale = 0.005 * ranges
        for i in range(n):
            if elapsed() >= max_time * 0.96:
                break
            simplex[i+1] = best_x.copy()
            simplex[i+1][i] += init_scale[i] * (1 if np.random.random() > 0.5 else -1)
            simplex[i+1] = clip(simplex[i+1])
            simplex_f[i+1] = eval_f(simplex[i+1])
        
        while elapsed() < max_time * 0.97:
            order = np.argsort(simplex_f)
            simplex = simplex[order]; simplex_f = simplex_f[order]
            if np.abs(simplex_f[-1] - simplex_f[0]) < 1e-17:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + (centroid - simplex[-1]))
            fr = eval_f(xr)
            if fr < simplex_f[0]:
                xe = clip(centroid + 2*(xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1]=xe; simplex_f[-1]=fe
                else:
                    simplex[-1]=xr; simplex_f[-1]=fr
            elif fr < simplex_f[-2]:
                simplex[-1]=xr; simplex_f[-1]=fr
            else:
                if fr < simplex_f[-1]:
                    xc = clip(centroid + 0.5*(xr - centroid))
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1]=xc; simplex_f[-1]=fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + 0.5*(simplex[i]-simplex[0]))
                            simplex_f[i] = eval_f(simplex[i])
                            if elapsed() >= max_time * 0.97: break
                else:
                    xc = clip(centroid - 0.5*(centroid - simplex[-1]))
                    fc = eval_f(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1]=xc; simplex_f[-1]=fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + 0.5*(simplex[i]-simplex[0]))
                            simplex_f[i] = eval_f(simplex[i])
                            if elapsed() >= max_time * 0.97: break

    # --- Phase 5: Coordinate descent with acceleration ---
    if best_x is not None and elapsed() < max_time * 0.998:
        x_c = best_x.copy(); f_c = best
        step = 0.001 * ranges
        while elapsed() < max_time * 0.998:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.998: break
                for sign in [1, -1]:
                    x_t = x_c.copy()
                    x_t[d] += sign * step[d]
                    x_t = clip(x_t)
                    f_t = eval_f(x_t)
                    if f_t < f_c:
                        x_c = x_t; f_c = f_t; improved = True
                        for _ in range(8):
                            x_t2 = x_c.copy()
                            x_t2[d] += sign * step[d]
                            x_t2 = clip(x_t2)
                            f_t2 = eval_f(x_t2)
                            if f_t2 < f_c:
                                x_c = x_t2; f_c = f_t2
                            else:
                                break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-16: break
    
    return best
