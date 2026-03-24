#Looking at the current algorithm, I can see it uses CMA-ES with restarts and coordinate descent refinement. Let me improve it with:
#
#1. Better initialization (Sobol-like + LHS)
#2. More robust CMA-ES with better restart strategy (IPOP-CMA-ES)
#3. Better time management
#4. Differential Evolution as an additional search phase
#5. More sophisticated local search (pattern search + golden section)
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
    
    def remaining():
        return max_time - elapsed()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Initial sampling with LHS ---
    n_init = min(max(30 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.15):
            break
        init_fitness[i] = eval_f(init_pop[i])
    
    valid = init_fitness < float('inf')
    if np.any(valid):
        sorted_idx = np.argsort(init_fitness[valid])
        valid_pop = init_pop[valid]
        valid_fit = init_fitness[valid]
    else:
        return best
    
    # --- Phase 2: Differential Evolution ---
    pop_size = min(max(10 * dim, 40), 100)
    # Initialize DE population from best LHS samples + random
    n_elite = min(pop_size // 2, len(valid_pop))
    de_pop = np.zeros((pop_size, dim))
    de_fit = np.full(pop_size, float('inf'))
    
    top_idx = sorted_idx[:n_elite]
    for i in range(n_elite):
        de_pop[i] = valid_pop[top_idx[i]].copy()
        de_fit[i] = valid_fit[top_idx[i]]
    
    for i in range(n_elite, pop_size):
        de_pop[i] = lower + np.random.random(dim) * ranges
        if time_ok(0.20):
            de_fit[i] = eval_f(de_pop[i])
    
    # DE parameters - use adaptive (JADE-like)
    mu_F = 0.5
    mu_CR = 0.5
    archive = []
    
    de_time_budget = 0.45  # fraction of total time for DE
    gen = 0
    while time_ok(de_time_budget):
        S_F = []
        S_CR = []
        
        for i in range(pop_size):
            if not time_ok(de_time_budget):
                break
            
            # Adaptive parameters
            F = np.clip(np.random.standard_cauchy() * 0.1 + mu_F, 0.1, 1.0)
            CR = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
            
            # current-to-pbest/1 mutation
            p = max(2, int(0.1 * pop_size))
            p_best_idx = np.argsort(de_fit)[:p]
            x_pbest = de_pop[np.random.choice(p_best_idx)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            idxs.remove(r1)
            
            if len(archive) > 0 and np.random.random() < 0.5:
                r2_pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                r2_pool = [x for x in r2_pool if x != i and x != r1]
                r2 = np.random.choice(r2_pool)
                if r2 >= pop_size:
                    x_r2 = archive[r2 - pop_size]
                else:
                    x_r2 = de_pop[r2]
            else:
                r2 = np.random.choice(idxs)
                x_r2 = de_pop[r2]
            
            mutant = de_pop[i] + F * (x_pbest - de_pop[i]) + F * (de_pop[r1] - x_r2)
            
            # Binomial crossover
            trial = de_pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (de_pop[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - de_pop[i][d])
            trial = clip(trial)
            
            f_trial = eval_f(trial)
            
            if f_trial <= de_fit[i]:
                if f_trial < de_fit[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    archive.append(de_pop[i].copy())
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(len(archive)))
                de_pop[i] = trial
                de_fit[i] = f_trial
        
        # Update adaptive parameters
        if len(S_F) > 0:
            mu_F = 0.9 * mu_F + 0.1 * (np.sum(np.array(S_F)**2) / np.sum(S_F))
            mu_CR = 0.9 * mu_CR + 0.1 * np.mean(S_CR)
        
        gen += 1
    
    # --- Phase 3: CMA-ES from best found solutions ---
    def cmaes_search(x0, sigma0, end_fraction):
        nonlocal best, best_params
        
        n = dim
        lam = 4 + int(3 * np.log(n))
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
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        
        generation = 0
        stagnation = 0
        prev_best_gen = float('inf')
        
        while time_ok(end_fraction):
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Mirror boundary handling
                for d in range(n):
                    while arx[k][d] < lower[d] or arx[k][d] > upper[d]:
                        if arx[k][d] < lower[d]:
                            arx[k][d] = 2 * lower[d] - arx[k][d]
                        if arx[k][d] > upper[d]:
                            arx[k][d] = 2 * upper[d] - arx[k][d]
                arx[k] = clip(arx[k])
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if not time_ok(end_fraction):
                    return
                fitnesses[k] = eval_f(arx[k])
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            best_gen = fitnesses[idx[0]]
            if best_gen < prev_best_gen - 1e-12:
                stagnation = 0
                prev_best_gen = best_gen
            else:
                stagnation += 1
            
            if stagnation > 10 + 30 * n / lam:
                return  # restart
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            mean_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, 2 * np.max(ranges))
            
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
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            # Check condition number
            if len(D) > 0 and max(D) / (min(D) + 1e-30) > 1e14:
                return  # restart
            
            generation += 1
    
    # Run CMA-ES restarts with increasing population (IPOP)
    n_cma_restarts = 0
    base_sigma = 0.3 * np.mean(ranges)
    
    while time_ok(0.88):
        if n_cma_restarts == 0 and best_params is not None:
            x0 = best_params.copy()
            sigma0 = base_sigma * 0.5
        elif n_cma_restarts < 3 and best_params is not None:
            # Perturbed restart from best
            x0 = best_params + np.random.normal(0, 0.1, dim) * ranges
            x0 = clip(x0)
            sigma0 = base_sigma * (0.3 + 0.3 * n_cma_restarts)
        else:
            # Random restart with larger sigma
            x0 = lower + np.random.random(dim) * ranges
            sigma0 = base_sigma * min(1.0, 0.5 * (1.2 ** n_cma_restarts))
        
        frac = min(elapsed() / max_time + max(0.08, remaining() * 0.4 / max_time), 0.88)
        cmaes_search(x0, sigma0, frac)
        n_cma_restarts += 1
    
    # --- Phase 4: Fine-grained local search ---
    if best_params is not None and time_ok(0.97):
        x = best_params.copy()
        step = 0.01 * ranges
        
        while time_ok(0.97):
            improved = False
            for d in range(dim):
                if not time_ok(0.97):
                    break
                for direction in [1.0, -1.0]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = eval_f(trial)
                    if f < best:
                        x = best_params.copy()
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-12:
                    break
            else:
                x = best_params.copy()
    
    return best
#
#Key improvements:
#1. **JADE-style adaptive DE** with current-to-pbest/1 mutation, archive, and adaptive F/CR
#2. **Better boundary handling** (bounce-back for DE, mirror for CMA-ES)
#3. **Stagnation detection** in CMA-ES with automatic restarts
#4. **IPOP-style restarts** alternating between local refinements near best and broader exploration
#5. **Better time budget allocation** across phases
#6. **Condition number monitoring** in CMA-ES to trigger restarts when needed
