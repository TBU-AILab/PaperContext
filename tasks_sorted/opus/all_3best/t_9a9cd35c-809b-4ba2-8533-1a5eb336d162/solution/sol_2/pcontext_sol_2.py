#Looking at the previous results, the best score was 141.71 from a CMA-ES with IPOP restarts. I'll improve by:
#
#1. **Separable CMA-ES (sep-CMA-ES)** for high dimensions to avoid O(n²) covariance overhead
#2. **Differential Evolution hybrid** for better global exploration
#3. **Local search refinement** (Powell-like coordinate descent) on best solutions
#4. **Better restart strategy** mixing IPOP and BIPOP
#5. **Optimized time allocation** between phases
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = [0]
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube + Sobol-like initialization ---
    n_init = min(max(20 * dim, 100), 500)
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fits = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_func(init_points[i])
        init_fits.append((f, i))
    
    init_fits.sort()
    top_k = min(8, len(init_fits))
    start_points = [init_points[init_fits[i][1]].copy() for i in range(top_k)]
    
    if best_x is None:
        best_x = (lower + upper) / 2.0
    
    # --- Phase 2: Differential Evolution for global search ---
    pop_size = min(max(10 * dim, 40), 200)
    
    # Initialize population with mix of LHS top points and random
    pop = np.zeros((pop_size, dim))
    pop_f = np.full(pop_size, float('inf'))
    
    # Fill with top init points
    for i in range(min(top_k, pop_size)):
        pop[i] = start_points[i]
        pop_f[i] = init_fits[i][0]
    
    # Fill rest randomly
    for i in range(top_k, pop_size):
        pop[i] = lower + np.random.rand(dim) * ranges
        if time_left() <= 0:
            return best
        pop_f[i] = eval_func(pop[i])
    
    # DE parameters - adaptive (JADE-style)
    mu_F = 0.5
    mu_CR = 0.5
    archive = []
    
    de_time_frac = 0.35
    de_end_time = elapsed() + time_left() * de_time_frac
    
    gen = 0
    while elapsed() < de_end_time and time_left() > 1.0:
        S_F = []
        S_CR = []
        
        # Sort population for current-to-pbest
        sort_idx = np.argsort(pop_f)
        
        for i in range(pop_size):
            if time_left() <= 0:
                return best
            
            # Adaptive parameters
            F = np.clip(np.random.standard_cauchy() * 0.1 + mu_F, 0.1, 1.0)
            CR = np.clip(np.random.randn() * 0.1 + mu_CR, 0.0, 1.0)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = sort_idx[np.random.randint(0, p)]
            
            # Select r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            candidates.remove(r1)
            
            # r2 from pop + archive
            if archive:
                all_pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            else:
                all_pool = list(range(pop_size))
            all_pool = [x for x in all_pool if x != i and x != r1]
            r2_idx = np.random.choice(all_pool)
            
            if r2_idx < pop_size:
                r2_vec = pop[r2_idx]
            else:
                r2_vec = archive[r2_idx - pop_size]
            
            # Mutation
            mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - r2_vec)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            f_trial = eval_func(trial)
            
            if f_trial <= pop_f[i]:
                if f_trial < pop_f[i]:
                    archive.append(pop[i].copy())
                    S_F.append(F)
                    S_CR.append(CR)
                pop[i] = trial
                pop_f[i] = f_trial
        
        # Trim archive
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        # Update adaptive parameters
        if S_F:
            mu_F = 0.9 * mu_F + 0.1 * (np.sum(np.array(S_F)**2) / np.sum(S_F))
            mu_CR = 0.9 * mu_CR + 0.1 * np.mean(S_CR)
        
        gen += 1
    
    # --- Phase 3: CMA-ES with BIPOP restarts from best solutions ---
    base_lam = max(4 + int(3 * np.log(dim)), 12)
    use_sep = dim > 40  # Use separable CMA-ES for high dim
    
    restart_count = 0
    small_restarts = 0
    large_lam = base_lam
    
    def run_cmaes(x0, sigma0, lam):
        nonlocal best, best_x
        
        mean = x0.copy()
        sigma = sigma0
        n = dim
        mu_count = lam // 2
        
        weights = np.log(mu_count + 0.5) - np.log(np.arange(1, mu_count + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        if use_sep:
            diagC = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        
        counteval = 0
        gen_local = 0
        best_local = float('inf')
        stag_count = 0
        
        while True:
            if time_left() <= 0.2:
                return
            
            arx = np.zeros((lam, n))
            fitnesses = np.zeros(lam)
            
            for k in range(lam):
                if time_left() <= 0.2:
                    return
                z = np.random.randn(n)
                if use_sep:
                    x = mean + sigma * np.sqrt(diagC) * z
                else:
                    x = mean + sigma * (B @ (D * z))
                x = clip(x)
                arx[k] = x
                fitnesses[k] = eval_func(x)
                counteval += 1
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            fitnesses = fitnesses[idx]
            
            if fitnesses[0] < best_local:
                best_local = fitnesses[0]
                stag_count = 0
            else:
                stag_count += 1
            
            old_mean = mean.copy()
            mean = np.dot(weights, arx[:mu_count])
            
            mu_diff = mean - old_mean
            
            if use_sep:
                inv_sqrt_diag = 1.0 / np.sqrt(diagC)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_diag * mu_diff) / sigma
                hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen_local + 1))) / chiN < 1.4 + 2 / (n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mu_diff / sigma
                
                artmp = (arx[:mu_count] - old_mean) / sigma
                diagC = (1 - c1 - cmu_val) * diagC + \
                    c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                    cmu_val * np.sum(weights[:, None] * artmp ** 2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mu_diff) / sigma
                hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen_local + 1))) / chiN < 1.4 + 2 / (n + 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mu_diff / sigma
                
                artmp = (arx[:mu_count] - old_mean) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (artmp.T @ np.diag(weights) @ artmp)
                
                if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except np.linalg.LinAlgError:
                        return
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 2)
            
            gen_local += 1
            
            if sigma < 1e-14 or stag_count > 10 + 30 * n / lam:
                return
            if use_sep:
                if sigma * np.max(np.sqrt(diagC)) < 1e-12 * np.max(ranges):
                    return
            else:
                if sigma * np.max(D) < 1e-12 * np.max(ranges):
                    return
    
    # Collect best DE solutions for CMA-ES starts
    de_sort = np.argsort(pop_f)
    de_starts = [pop[de_sort[i]].copy() for i in range(min(5, pop_size))]
    
    sp_idx = 0
    all_starts = de_starts + start_points
    
    while time_left() > 0.5:
        # BIPOP: alternate between small and large restarts
        if restart_count % 3 == 0 and restart_count > 0:
            # Small restart near best
            x0 = best_x + np.random.randn(dim) * ranges * 0.05
            x0 = clip(x0)
            lam = base_lam
            sigma0 = np.mean(ranges) / 10.0
        elif sp_idx < len(all_starts):
            x0 = all_starts[sp_idx]
            sp_idx += 1
            lam = base_lam
            sigma0 = np.mean(ranges) / 4.0
        else:
            # Large restart
            x0 = lower + np.random.rand(dim) * ranges
            large_lam = int(base_lam * (2 ** (restart_count * 0.4)))
            lam = min(large_lam, max(base_lam, 300))
            sigma0 = np.mean(ranges) / 3.0
        
        run_cmaes(x0, sigma0, lam)
        restart_count += 1
    
    # --- Phase 4: Local refinement on best ---
    if time_left() > 0.3 and best_x is not None:
        step = ranges * 0.001
        for _ in range(3):
            if time_left() <= 0.1:
                break
            for d in range(dim):
                if time_left() <= 0.1:
                    break
                for sign in [1, -1]:
                    trial = best_x.copy()
                    trial[d] += sign * step[d]
                    trial = clip(trial)
                    eval_func(trial)
    
    return best
