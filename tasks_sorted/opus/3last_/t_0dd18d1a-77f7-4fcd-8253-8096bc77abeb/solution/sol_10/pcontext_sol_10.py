#
#
#Looking at the results, algorithm 3 performed best (12.66). I'll analyze what worked and improve:
#
#1. L-SHADE + CMA-ES combination is effective
#2. Need better time allocation and more CMA-ES restarts
#3. Fix the truncated Nelder-Mead code
#4. Add sep-CMA for high dimensions
#5. Better adaptive strategies
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
    
    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(50, 12 * dim), 300)
    init_pop_size = pop_size
    
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
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.85)
    archive = []
    k = 0
    min_pop_size = max(4, dim)
    shade_end_frac = 0.25
    
    gen = 0
    while elapsed() < max_time * shade_end_frac:
        gen += 1
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
            
            p = max(2, int(0.11 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = idxs[np.random.randint(0, len(idxs))]
            
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
            trial = clip(trial)
            
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
        
        # Linear population size reduction
        progress = elapsed() / (max_time * shade_end_frac)
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * progress)))
        if new_pop_size < pop_size:
            idx_sort = np.argsort(fitness[:pop_size])
            population = population[idx_sort[:new_pop_size]].copy()
            fitness = fitness[idx_sort[:new_pop_size]].copy()
            pop_size = new_pop_size
    
    # Save top solutions
    top_k = min(8, pop_size)
    top_idx = np.argsort(fitness[:pop_size])[:top_k]
    top_solutions = population[top_idx].copy()
    top_fitness = fitness[top_idx].copy()
    
    # --- Phase 3: CMA-ES with BIPOP restarts ---
    def run_cmaes(init_mean, init_sigma, lam, deadline, use_sep=False):
        nonlocal best, best_params
        
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
                sqrt_C = np.sqrt(np.maximum(C_diag, 1e-20))
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
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            
            if use_sep:
                inv_sqrt_C = 1.0 / (sqrt_C + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C * diff
                norm_ps = np.linalg.norm(ps)
                hsig = float(norm_ps / np.sqrt(1 - (1 - cs)**(2 * count_eval / lam) + 1e-30) / chiN < 1.4 + 2/(n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
                C_diag = (1 - c1 - cmu_val) * C_diag + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag) + cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
                norm_ps = np.linalg.norm(ps)
                hsig = float(norm_ps / np.sqrt(1 - (1 - cs)**(2 * count_eval / lam) + 1e-30) / chiN < 1.4 + 2/(n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
                C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
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
            
            if sigma < 1e-15 or stag_count > 10 + int(30 * n / lam):
                break
    
    base_lam = 4 + int(3 * np.log(dim))
    use_sep = dim > 40
    n_restarts = 0
    large_lam_mult = 1
    small_budget_used = 0
    large_budget_used = 0
    
    cma_end_frac = 0.90
    
    while elapsed() < max_time * cma_end_frac:
        rem = max_time * cma_end_frac - elapsed()
        if rem < 0.15:
            break
        
        # BIPOP strategy: alternate between small and large population restarts
        if n_restarts == 0:
            # First restart: best known solution, small pop
            init_m = best_params.copy()
            init_s = 0.1 * np.mean(ranges)
            cur_lam = base_lam
            time_share = min(rem * 0.3, max(rem * 0.1, 2.0))
        elif n_restarts == 1:
            # Tighter search around best
            init_m = best_params + np.random.randn(dim) * ranges * 0.005
            init_m = clip(init_m)
            init_s = 0.03 * np.mean(ranges)
            cur_lam = base_lam
            time_share = min(rem * 0.25, max(rem * 0.1, 1.5))
        elif small_budget_used <= large_budget_used:
            # Small population restart near best or top solutions
            if np.random.random() < 0.5 and len(top_solutions) > 1:
                sol_idx = np.random.randint(0, len(top_solutions))
                init_m = top_solutions[sol_idx] + np.random.randn(dim) * ranges * 0.02
            else:
                init_m = best_params + np.random.randn(dim) * ranges * 0.05
            init_m = clip(init_m)
            init_s = 0.08 * np.mean(ranges) * (0.5 + np.random.random())
            cur_lam = max(base_lam, int(base_lam * (0.5 + np.random.random())))
            time_share = min(rem * 0.2, max(0.5, rem * 0.08))
        else:
            # Large population restart for exploration
            large_lam_mult = min(large_lam_mult * 2, 10)
            init_m = lower + np.random.rand(dim) * ranges
            init_s = 0.3 * np.mean(ranges)
            cur_lam = base_lam * large_lam_mult
            time_share = min(rem * 0.4, max(1.0, rem * 0.15))
        
        deadline = elapsed() + time_share
        
