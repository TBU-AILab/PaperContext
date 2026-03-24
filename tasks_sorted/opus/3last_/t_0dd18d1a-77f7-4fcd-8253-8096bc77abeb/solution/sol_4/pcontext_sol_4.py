#Looking at the results, algorithm 1 (22.34) performed best with its full CMA-ES implementation. I'll improve it by:
#
#1. Better CMA-ES with proper restart strategy (IPOP-CMA-ES)
#2. More aggressive local search near the best solution
#3. Improved SHADE with better parameter adaptation
#4. Nelder-Mead simplex for final polishing
#5. Better time allocation
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def remaining():
        return max(0, max_time * 0.95 - elapsed())
    
    best_params = None
    
    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(30, 10 * dim), 200)
    init_pop_size = pop_size
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = (perm[j] + np.random.uniform()) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: L-SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    archive = []
    k = 0
    min_pop_size = max(4, dim)
    shade_end_time = 0.40
    
    while elapsed() < max_time * shade_end_time:
        S_F, S_CR, S_delta = [], [], []
        
        for i in range(pop_size):
            if elapsed() >= max_time * shade_end_time:
                break
            
            ri = np.random.randint(0, H)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            sorted_idx = np.argsort(fitness[:pop_size])
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(0, union_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, union_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = clip(np.where(cross_points, mutant, population[i]))
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                delta = fitness[i] - trial_fitness
                if delta > 0:
                    archive.append(population[i].copy())
                    if len(archive) > init_pop_size:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        if S_F:
            w = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k % H] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k % H] = np.sum(w * np.array(S_CR))
            k += 1
        
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / (max_time * shade_end_time))))
        if new_pop_size < pop_size:
            idx_sort = np.argsort(fitness[:pop_size])
            population = population[idx_sort[:new_pop_size]].copy()
            fitness = fitness[idx_sort[:new_pop_size]].copy()
            pop_size = new_pop_size
    
    # --- Phase 3: CMA-ES with IPOP restarts ---
    def run_cmaes(init_mean, init_sigma, lam_mult, deadline):
        nonlocal best, best_params
        
        n = dim
        lam = int(lam_mult * (4 + int(3 * np.log(n))))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        mean = init_mean.copy()
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        count_eval = 0
        stag_count = 0
        prev_best = best
        
        while elapsed() < deadline:
            arz = np.random.randn(lam, n)
            arx = np.array([clip(mean + sigma * (B @ (D * z))) for z in arz])
            
            fits = np.array([func(x) for x in arx])
            count_eval += lam
            
            if elapsed() >= deadline:
                break
            
            idx = np.argsort(fits)
            if fits[idx[0]] < best:
                best = fits[idx[0]]
                best_params = arx[idx[0]].copy()
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            mean = clip(mean)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * count_eval / lam)) / chiN < 1.4 + 2/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, max(ranges) * 2)
            
            if count_eval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                eigeneval = count_eval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-20))
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    break
                
                if np.max(D) > 1e7 * np.min(D):
                    break
            
            if sigma < 1e-14 or stag_count > 10 + int(30 * n / lam):
                break
    
    n_restarts = 0
    lam_mult = 1.0
    
    while elapsed() < max_time * 0.90:
        rem = remaining()
        if rem < 0.2:
            break
        
        time_share = min(0.5, max(0.15, rem * 0.4))
        deadline = elapsed() + time_share
        
        if n_restarts == 0:
            init_m = best_params.copy()
            init_s = 0.2 * np.mean(ranges)
        elif n_restarts <= 2:
            init_m = best_params + np.random.randn(dim) * ranges * 0.03
            init_m = clip(init_m)
            init_s = 0.1 * np.mean(ranges)
            lam_mult = min(lam_mult * 2, 8)
        else:
            init_m = lower + np.random.rand(dim) * ranges
            init_s = 0.3 * np.mean(ranges)
            lam_mult = min(lam_mult * 2, 10)
        
        try:
            run_cmaes(init_m, init_s, lam_mult, deadline)
        except:
            pass
        n_restarts += 1
    
    # --- Phase 4: Nelder-Mead polish ---
    if best_params is not None and remaining() > 0.5:
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step = 0.02 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.array([func(clip(x)) for x in simplex])
        for i in range(n + 1):
            if f_simplex[i] < best:
                best = f_simplex[i]
                best_params = simplex[i].copy()
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = func(xr)
            if fr < best:
                best = fr
                best_params = xr.copy()
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = func(xe)
                if fe < best:
                    best = fe
                    best_params = xe.copy()
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = func(xc)
                if fc < best:
                    best = fc
                    best_params = xc.copy()
                if fc < f_simplex[-1]:
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = func(simplex[i])
                        if f_simplex[i] < best:
                            best = f_simplex[i]
                            best_params = simplex[i].copy()
    
    return best
