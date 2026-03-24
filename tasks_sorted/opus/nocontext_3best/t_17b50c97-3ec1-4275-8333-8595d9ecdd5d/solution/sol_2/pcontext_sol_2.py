#Looking at the two algorithms, the first one (score 15.08) is better. Let me analyze what could be improved:
#
#1. Better time allocation between phases
#2. More aggressive multi-start strategy with diverse initial points
#3. Improved DE with SHADE-like adaptive parameters
#4. Better CMA-ES restart strategy (BIPOP-style)
#5. More sophisticated local search
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
    
    eval_count = 0
    def evaluate(x):
        nonlocal best, best_params, eval_count
        x = clip(x)
        f = func(x)
        eval_count += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling initialization ---
    n_init = min(max(40 * dim, 300), 2000)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.85:
            break
        init_fitness[i] = evaluate(init_points[i])
    
    valid_mask = init_fitness < float('inf')
    valid_count = np.sum(valid_mask)
    sorted_idx = np.argsort(init_fitness)

    # --- Phase 2: SHADE-like Differential Evolution ---
    pop_size = min(max(8 * dim, 40), 150)
    n_elite_init = min(pop_size, valid_count)
    
    pop = np.zeros((pop_size, dim))
    pop_fit = np.full(pop_size, float('inf'))
    
    for i in range(n_elite_init):
        idx = sorted_idx[i]
        pop[i] = init_points[idx].copy()
        pop_fit[i] = init_fitness[idx]
    for i in range(n_elite_init, pop_size):
        pop[i] = lower + np.random.random(dim) * ranges
        pop_fit[i] = evaluate(pop[i])
    
    # SHADE memory
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    
    # Archive
    archive = []
    archive_max = pop_size
    
    de_end_fraction = 0.55  # use up to 55% of time for DE
    
    while time_left() > max_time * (1.0 - de_end_fraction) and time_left() > 1.0:
        S_F = []
        S_CR = []
        S_df = []
        
        for i in range(pop_size):
            if time_left() < max_time * (1.0 - de_end_fraction) or time_left() < 0.5:
                break
            
            ri = np.random.randint(H)
            F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            p_best_idx = np.argsort(pop_fit)[:p]
            pbest = pop[p_best_idx[np.random.randint(p)]]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # r2 from pop + archive
            union_size = pop_size + len(archive)
            r2_idx = np.random.randint(union_size)
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(union_size)
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
            S_df = np.array(S_df)
            w = S_df / (np.sum(S_df) + 1e-30)
            S_F = np.array(S_F)
            S_CR = np.array(S_CR)
            M_F[k] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
            M_CR[k] = np.sum(w * S_CR)
            k = (k + 1) % H

    # --- Phase 3: CMA-ES with IPOP restarts ---
    def cmaes_run(x0, sigma0, time_budget):
        nonlocal best, best_params
        end_time = elapsed() + time_budget
        n = len(x0)
        
        lam = 4 + int(3 * np.log(n))
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
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        sigma = sigma0
        mean = x0.copy()
        gen = 0
        stag = 0
        prev_best = float('inf')
        
        while elapsed() < end_time and elapsed() < max_time * 0.92:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k_ in range(lam):
                arx[k_] = clip(mean + sigma * (B @ (D * arz[k_])))
            
            fitnesses = np.zeros(lam)
            for k_ in range(lam):
                if elapsed() >= end_time or elapsed() >= max_time * 0.92:
                    return
                fitnesses[k_] = evaluate(arx[k_])
            
            arindex = np.argsort(fitnesses)
            gb = fitnesses[arindex[0]]
            if gb >= prev_best:
                stag += 1
            else:
                stag = 0
            prev_best = gb
            if stag > 10 + 30 * n / lam:
                return
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for k_ in range(mu):
                C += cmu_val * weights[k_] * np.outer(artmp[k_], artmp[k_])
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
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
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                return
    
    # Multiple CMA-ES restarts
    sigma_base = 0.3 * np.max(ranges)
    n_restarts = 8
    pop_sorted = np.argsort(pop_fit)
    
    for r in range(n_restarts):
        t = time_left()
        if t < 0.8:
            break
        budget = t / max(1, n_restarts - r) * 0.85
        
        if r == 0:
            x0 = best_params.copy()
            s0 = sigma_base * 0.3
        elif r == 1:
            x0 = best_params.copy()
            s0 = sigma_base
        elif r < 4:
            # From DE top solutions
            idx = pop_sorted[min(r - 1, pop_size - 1)]
            x0 = pop[idx].copy()
            s0 = sigma_base * 0.5
        else:
            # Random restart from diverse region
            x0 = lower + np.random.random(dim) * ranges
            s0 = sigma_base * 0.8
        
        cmaes_run(x0, s0, budget)

    # --- Phase 4: Nelder-Mead refinement ---
    if time_left() > 0.3 and best_params is not None:
        n = dim
        step = 0.005 * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([evaluate(simplex[i]) for i in range(n + 1)])
        
        while time_left() > 0.1:
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
                xc = clip(centroid + 0.5 * ((simplex[-1] if fr >= f_simplex[-1] else xr) - centroid))
                fc = evaluate(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                        f_simplex[i] = evaluate(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                break
    
    return best
