#Looking at the results, algorithm 2 performed best (12.66). I'll improve it by:
#
#1. Better time allocation - more time for CMA-ES which seems most effective
#2. Separable CMA-ES for high dimensions to avoid costly eigendecomposition
#3. More restarts with diverse starting points
#4. Better local search with adaptive step sizes
#5. Fix the truncated Nelder-Mead code from algorithm 2
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
    
    best_params = None
    evals = 0
    
    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(40, 10 * dim), 300)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = (perm[j] + np.random.uniform()) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # --- Phase 2: L-SHADE ---
    H = 8
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    archive = []
    k = 0
    min_pop_size = max(4, dim)
    shade_end_frac = 0.25
    init_pop_size = pop_size
    
    while elapsed() < max_time * shade_end_frac:
        S_F, S_CR, S_delta = [], [], []
        sorted_idx = np.argsort(fitness[:pop_size])
        
        for i in range(pop_size):
            if elapsed() >= max_time * shade_end_frac:
                break
            
            ri = np.random.randint(0, H)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.1 * pop_size))
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
            trial = np.where(cross_points, mutant, population[i])
            
            trial_fitness = evaluate(trial)
            
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
        
        if S_F:
            w = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k % H] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            mean_cr = np.sum(w * np.array(S_CR))
            M_CR[k % H] = mean_cr if mean_cr > 0 else M_CR[k % H]
            k += 1
        
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / (max_time * shade_end_frac))))
        if new_pop_size < pop_size:
            idx_sort = np.argsort(fitness[:pop_size])
            population = population[idx_sort[:new_pop_size]].copy()
            fitness = fitness[idx_sort[:new_pop_size]].copy()
            pop_size = new_pop_size
    
    # Save top solutions
    top_k = min(5, pop_size)
    top_idx = np.argsort(fitness[:pop_size])[:top_k]
    top_solutions = population[top_idx].copy()
    
    # --- Phase 3: CMA-ES with IPOP restarts ---
    def run_cmaes(init_mean, init_sigma, lam, deadline):
        n = dim
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        mean = clip(init_mean.copy())
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_sep = n > 40
        
        if use_sep:
            C_diag = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            C = np.eye(n)
        
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        count_eval = 0
        stag_count = 0
        best_in_run = float('inf')
        
        while elapsed() < deadline:
            if use_sep:
                sqrt_C = np.sqrt(C_diag)
                arx = np.array([clip(mean + sigma * sqrt_C * np.random.randn(n)) for _ in range(lam)])
            else:
                arz = np.random.randn(lam, n)
                arx = np.array([clip(mean + sigma * (B @ (D * z))) for z in arz])
            
            fits = np.full(lam, float('inf'))
            for j in range(lam):
                if elapsed() >= deadline:
                    return
                fits[j] = evaluate(arx[j])
            count_eval += lam
            
            idx = np.argsort(fits)
            if fits[idx[0]] < best_in_run - 1e-10:
                best_in_run = fits[idx[0]]
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = clip(np.sum(weights[:, None] * arx[idx[:mu]], axis=0))
            
            diff = (mean - old_mean) / sigma
            
            if use_sep:
                inv_sqrt_C = 1.0 / np.sqrt(C_diag + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C * diff
                hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * count_eval / lam)) / chiN < 1.4 + 2/(n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C_diag = (1 - c1 - cmu_val) * C_diag + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag) + cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
                hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * count_eval / lam)) / chiN < 1.4 + 2/(n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, max(ranges) * 2)
            
            if not use_sep and count_eval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
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
            
            if sigma < 1e-15 or stag_count > 12 + int(30 * n / lam):
                break
    
    base_lam = 4 + int(3 * np.log(dim))
    n_restarts = 0
    lam_mult = 1
    
    while elapsed() < max_time * 0.88:
        rem = max_time * 0.88 - elapsed()
        if rem < 0.2:
            break
        
        time_share = min(rem * 0.5, max(rem * 0.15, 1.5 + n_restarts * 0.3))
        deadline = elapsed() + time_share
        
        if n_restarts == 0:
            init_m = best_params.copy()
            init_s = 0.12 * np.mean(ranges)
            cur_lam = base_lam
        elif n_restarts == 1:
            init_m = best_params + np.random.randn(dim) * ranges * 0.01
            init_m = clip(init_m)
            init_s = 0.05 * np.mean(ranges)
            cur_lam = base_lam
        elif n_restarts < 4:
            sol_idx = (n_restarts - 2) % len(top_solutions)
            init_m = top_solutions[sol_idx] + np.random.randn(dim) * ranges * 0.02
            init_m = clip(init_m)
            init_s = 0.1 * np.mean(ranges)
            lam_mult = min(lam_mult * 2, 8)
            cur_lam = base_lam * lam_mult
        else:
            lam_mult = min(lam_mult * 2, 12)
            init_m = lower + np.random.rand(dim) * ranges
            init_s = 0.3 * np.mean(ranges)
            cur_lam = base_lam * lam_mult
        
        try:
            run_cmaes(init_m, init_s, cur_lam, deadline)
        except:
            pass
        n_restarts += 1
    
    # --- Phase 4: Coordinate descent polish ---
    if best_params is not None:
        x = best_params.copy()
        fx = best
        step = 0.005 * ranges.copy()
        
        while elapsed() < max_time * 0.95:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                for sign in [1, -1]:
                    x_new = x.copy()
                    x_new[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    f_new = evaluate(x_new)
                    if f_new < fx:
                        while elapsed() < max_time * 0.95:
                            x = x_new.copy()
                            fx = f_new
                            step[d] *= 1.5
                            x_new = x.copy()
                            x_new[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                            f_new = evaluate(x_new)
                            if f_new >= fx:
                                break
                            x = x_new.copy()
                            fx = f_new
                        improved = True
