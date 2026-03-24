#Looking at the algorithms, I need to improve the optimization strategy. Key observations:
#
#1. The best algorithm (12.53) uses LHS + SHADE + CMA-ES + Nelder-Mead
#2. Time allocation and restart diversity matter significantly
#3. L-SHADE with population reduction helps
#4. I should add better local search and more sophisticated restart strategies
#
#Let me improve with:
#- Better SHADE with weighted recombination
#- Smarter CMA-ES restarts using clustering of good solutions
#- Powell-like coordinate descent for final refinement
#- Tighter time management
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
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    # Store top-k diverse solutions
    top_solutions = []
    top_k = 20
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        # Maintain diverse top solutions
        if len(top_solutions) < top_k:
            top_solutions.append((f, x.copy()))
            top_solutions.sort(key=lambda t: t[0])
        elif f < top_solutions[-1][0]:
            # Check diversity - don't add if too close to existing
            min_dist = min(np.linalg.norm(x - s[1]) for s in top_solutions)
            if min_dist > 0.01 * np.linalg.norm(ranges):
                top_solutions[-1] = (f, x.copy())
                top_solutions.sort(key=lambda t: t[0])
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(50 * dim, 500), 3000)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.87:
            break
        init_fitness[i] = evaluate(init_points[i])
    
    valid_count = int(np.sum(init_fitness < float('inf')))
    sorted_idx = np.argsort(init_fitness)

    # Opposition-based learning on top candidates
    n_opp = min(valid_count // 5, 150)
    for i in range(n_opp):
        if time_left() < max_time * 0.84:
            break
        idx = sorted_idx[i]
        opp = lower + upper - init_points[idx]
        evaluate(opp)

    # --- Phase 2: L-SHADE ---
    pop_size_init = min(max(10 * dim, 60), 200)
    pop_size_min = max(4, dim)
    pop_size = pop_size_init
    
    pop = np.zeros((pop_size, dim))
    pop_fit = np.full(pop_size, float('inf'))
    
    n_elite_init = min(pop_size, valid_count)
    for i in range(n_elite_init):
        idx = sorted_idx[i]
        pop[i] = init_points[idx].copy()
        pop_fit[i] = init_fitness[idx]
    for i in range(n_elite_init, pop_size):
        pop[i] = lower + np.random.random(dim) * ranges
        pop_fit[i] = evaluate(pop[i])
    
    H = 60
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k_idx = 0
    archive = []
    archive_max = pop_size_init
    
    nfe_start = eval_count[0]
    nfe_max = max(800, 25 * dim * pop_size_init)
    de_time_end = 0.52
    
    gen_count = 0
    while time_left() > max_time * (1.0 - de_time_end) and time_left() > 1.0:
        S_F, S_CR, S_df = [], [], []
        
        progress = min(1.0, (eval_count[0] - nfe_start) / max(1, nfe_max))
        new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * progress)))
        
        if new_pop_size < pop_size:
            best_idx = np.argsort(pop_fit)[:new_pop_size]
            pop = pop[best_idx]
            pop_fit = pop_fit[best_idx]
            pop_size = new_pop_size
        
        for i in range(pop_size):
            if time_left() < max_time * (1.0 - de_time_end) or time_left() < 0.5:
                break
            
            ri = np.random.randint(H)
            F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(0.11 * pop_size))
            p_best_idx = np.argsort(pop_fit)[:p]
            pbest = pop[p_best_idx[np.random.randint(p)]]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            union_size = pop_size + len(archive)
            r2_idx = np.random.randint(union_size)
            attempts = 0
            while (r2_idx == i or r2_idx == r1) and attempts < 20:
                r2_idx = np.random.randint(union_size)
                attempts += 1
            xr2 = pop[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = pop[i] + F_i * (pbest - pop[i]) + F_i * (pop[r1] - xr2)
            
            cross_points = np.random.random(dim) < CR_i
            cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + pop[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + pop[i][d]) / 2
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            if f_trial <= pop_fit[i]:
                if f_trial < pop_fit[i]:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_df.append(pop_fit[i] - f_trial)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                pop[i] = trial
                pop_fit[i] = f_trial
        
        if S_F:
            S_df_arr = np.array(S_df)
            w = S_df_arr / (np.sum(S_df_arr) + 1e-30)
            S_F_arr = np.array(S_F)
            S_CR_arr = np.array(S_CR)
            M_F[k_idx] = np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
            M_CR[k_idx] = np.sum(w * S_CR_arr)
            k_idx = (k_idx + 1) % H
        
        gen_count += 1

    # Collect diverse restart points
    restart_points = []
    ps = np.argsort(pop_fit)
    for i in ps[:min(8, pop_size)]:
        restart_points.append((pop_fit[i], pop[i].copy()))
    for f_val, x_val in top_solutions[:8]:
        restart_points.append((f_val, x_val.copy()))
    restart_points.sort(key=lambda t: t[0])
    # Remove near-duplicates
    filtered = [restart_points[0]]
    for f_val, x_val in restart_points[1:]:
        if all(np.linalg.norm(x_val - fp[1]) > 0.05 * np.linalg.norm(ranges) for fp in filtered):
            filtered.append((f_val, x_val))
    restart_points = filtered

    # --- Phase 3: CMA-ES with restarts ---
    def cmaes_run(x0, sigma0, time_budget, lam_mult=1):
        nonlocal best, best_params
        end_time = elapsed() + time_budget
        n = len(x0)
        
        lam = max(4 + int(3 * np.log(n)), (4 + int(3 * np.log(n))) * lam_mult)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        pc = np.zeros(n)
        ps_vec = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        sigma = sigma0
        mean = x0.copy()
        gen = 0
        stag = 0
        prev_best_f = float('inf')
        f_hist = []
        
        while elapsed() < end_time and elapsed() < max_time * 0.94:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for kk in range(lam):
                arx[kk] = clip(mean + sigma * (B @ (D * arz[kk])))
            
            fitnesses = np.zeros(lam)
            for kk in range(lam):
                if elapsed() >= end_time or elapsed() >= max_time * 0.94:
                    return
                fitnesses[kk] = evaluate(arx[kk])
            
            arindex = np.argsort(fitnesses)
            gb = fitnesses[arindex[0]]
            f_hist.append(gb)
            
            if gb >= prev_best_f - 1e-15:
                stag += 1
            else:
                stag = 0
            prev_best_f = min(prev_best_f, gb)
            
            if stag > 10 + 30 * n / lam:
                return
            if len(f_hist) > 25 and abs(f_hist[-1] - f_hist[-25]) < 1e-14:
                return
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            ps_vec = (1 - cs) * ps_vec + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            hsig = float(np.linalg.norm(ps_vec) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - old_mean) / (sigma + 1e-30)
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for kk in range(mu):
                C += cmu_val * weights[kk] * np.outer(artmp[kk], artmp[kk])
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps_vec) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu_val + 1e-20) / n / 10:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return
            
            if sigma * np.max(D) < 1e-13 * np.max(ranges):
                return
    
    sigma_base = 0.3 * np.max(ranges)
    n_restarts = 12
    
    for r in range(n_restarts):
        t = time_left()
        if t < 0.6:
            break
        budget = t / max(1, n_restarts - r) * 0.88
        
        if r == 0:
            x0 = best_params.copy()
            s0 = sigma_base * 0.15
            lm = 1
        elif r == 1:
            x0 = best_params.copy()
            s0 = sigma_base * 0.5
            lm = 1
        elif r == 2:
            x0 = best_params.copy()
            s0 = sigma_base
            lm = 2
        elif r - 3 < len(restart_points):
            x0 = restart_points[r - 3][1].copy()
            s0 = sigma_base * 0.3
            lm = 1
        elif r % 2 == 0:
            x0 = lower + np.random.random(dim) * ranges
            s0 = sigma_base * 0.9
            lm = 2
        else:
            x0 = best_params + np.random.randn(dim) * sigma_base * 0.4
            x0 = clip(x0)
            s0 = sigma_base * 0.25
            lm = 1
        
        cmaes_run(x0, s0, budget, lm)

    # --- Phase 4: Coordinate-wise golden section + Nelder-Mead ---
    # Golden section search along each coordinate
    if time_left() > 0.8 and best_params is not None:
        gr = (np.sqrt(5) - 1) / 2
        x_cur = best_params.copy()
        f_cur = best
        
        for d in range(dim):
            if time_left() < 0.4:
                break
            a = max(lower[d], x_cur[d] - 0.05 * ranges[d])
            b = min(upper[d], x_cur[d] + 0.05 * ranges[d])
            
            c = b - gr * (b - a)
            dd = a + gr * (b - a)
            
            xc = x_cur.copy(); xc[d] = c
            xd = x_cur.copy(); xd[d] = dd
            fc = evaluate(xc)
            fd = evaluate(xd)
            
            for _ in range(8):
                if time_left() < 0.3:
                    break
                if fc < fd:
                    b = dd
                    dd = c
                    fd = fc
                    c = b - gr * (b - a)
                    xc = x_cur.copy(); xc[d] = c
                    fc = evaluate(xc)
                else:
                    a = c
                    c = dd
                    fc = fd
                    dd = a + gr * (b - a)
                    xd = x_cur.copy(); xd[d] = dd
                    fd = evaluate(xd)
            
            best_d = (a + b) / 2
            x_cur[d] = best_d
            f_cur = evaluate(x_cur)

    # Nelder-Mead refinement
    if time_left() > 0.3 and best_params is not None:
        n = dim
        step = 0.002 * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([evaluate(simplex[i]) for i in range(n + 1)])
        
        while time_left() > 0.05:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + 1.0 * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + 2.0 * (xr - centroid))
                fe = evaluate(xe)
                simplex[-1], f_simplex[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                ref = simplex[-1] if fr >= f_simplex[-1] else xr
                f_ref = f_simplex[-1] if fr >= f_simplex[-1] else fr
                xc = clip(centroid + 0.5 * (ref - centroid))
                fc = evaluate(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                        f_simplex[i] = evaluate(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-16:
                break
    
    return best
