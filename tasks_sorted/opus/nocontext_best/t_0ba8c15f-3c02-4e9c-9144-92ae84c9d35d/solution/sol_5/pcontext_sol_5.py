#Looking at the current algorithm, I see it uses CMA-ES with restarts and coordinate descent refinement. Let me improve it with:
#
#1. Better CMA-ES implementation with boundary handling
#2. Differential Evolution as an alternative global search
#3. More efficient time budgeting
#4. Better local search (pattern search + golden section)
#5. Surrogate-assisted optimization for expensive functions
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
    
    def update(f, x):
        nonlocal best, best_params
        if f < best:
            best = f
            best_params = x.copy()
    
    def eval_func(x):
        x_clipped = np.clip(x, lower, upper)
        f = func(x_clipped)
        update(f, x_clipped)
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(30 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.15:
            n_init = i
            init_pop = init_pop[:n_init]
            init_fitness = init_fitness[:n_init]
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    if n_init == 0:
        return best
    
    # --- Phase 2: Differential Evolution ---
    pop_size = min(max(10 * dim, 40), 100)
    sorted_idx = np.argsort(init_fitness)
    
    # Initialize DE population from best LHS samples
    pop = np.zeros((pop_size, dim))
    pop_fit = np.full(pop_size, float('inf'))
    n_from_init = min(pop_size, n_init)
    for i in range(n_from_init):
        pop[i] = init_pop[sorted_idx[i]].copy()
        pop_fit[i] = init_fitness[sorted_idx[i]]
    for i in range(n_from_init, pop_size):
        pop[i] = lower + np.random.random(dim) * ranges
        pop_fit[i] = eval_func(pop[i])
        if elapsed() >= max_time * 0.2:
            pop_size = i + 1
            pop = pop[:pop_size]
            pop_fit = pop_fit[:pop_size]
            break

    # DE with adaptive parameters (SHADE-like)
    F_memory = np.full(20, 0.5)
    CR_memory = np.full(20, 0.5)
    mem_idx = 0
    
    de_budget = 0.55
    gen = 0
    while elapsed() < max_time * de_budget:
        S_F = []
        S_CR = []
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_budget:
                break
            
            # Sample F and CR from memory
            r = np.random.randint(len(F_memory))
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + F_memory[r], 0.1, 1.0)
            CRi = np.clip(np.random.normal(CR_memory[r], 0.1), 0.0, 1.0)
            
            # current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.argsort(pop_fit)[:p]
            xpbest = pop[np.random.choice(pbest_idx)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - pop[r2])
            
            # Binomial crossover
            jrand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[jrand] = True
            trial = np.where(mask, mutant, pop[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (pop[i, d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - pop[i, d])
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_func(trial)
            
            if f_trial <= pop_fit[i]:
                if f_trial < pop_fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                pop[i] = trial
                pop_fit[i] = f_trial
        
        # Update memory
        if len(S_F) > 0:
            F_memory[mem_idx % len(F_memory)] = np.mean(np.array(S_F)**2) / np.mean(S_F) if np.mean(S_F) > 0 else 0.5
            CR_memory[mem_idx % len(CR_memory)] = np.mean(S_CR)
            mem_idx += 1
        
        gen += 1
    
    # --- Phase 3: CMA-ES from best solution ---
    def cmaes_search(x0, sigma0, end_frac):
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
        no_improve_count = 0
        prev_best = best
        
        while elapsed() < max_time * end_frac:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], lower, upper)
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= max_time * end_frac:
                    return
                fitnesses[k] = eval_func(arx[k])
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            mean = np.clip(mean, lower, upper)
            
            mean_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) / chiN < 1.4 + 2/(n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges))
            
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu_val) / n / 10:
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
                    sigma = sigma0 * 0.5
            
            generation += 1
            
            if best < prev_best:
                no_improve_count = 0
                prev_best = best
            else:
                no_improve_count += 1
            
            if no_improve_count > 20 + 5 * dim:
                break
            
            if sigma < 1e-16:
                break
    
    # Run CMA-ES restarts
    n_cma_restarts = 3
    for r in range(n_cma_restarts):
        if elapsed() >= max_time * 0.85:
            break
        if best_params is None:
            x0 = lower + np.random.random(dim) * ranges
        else:
            if r == 0:
                x0 = best_params.copy()
            else:
                # Perturb best
                x0 = best_params + np.random.randn(dim) * ranges * 0.1
                x0 = np.clip(x0, lower, upper)
        
        sigma0 = 0.2 * np.mean(ranges) * (0.5 ** r)
        end_frac = min(0.55 + (r + 1) * 0.12, 0.88)
        cmaes_search(x0, sigma0, end_frac)
    
    # --- Phase 4: Local refinement - Nelder-Mead ---
    if best_params is not None and elapsed() < max_time * 0.92:
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex around best
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step = 0.05 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([eval_func(simplex[i]) for i in range(n + 1) if elapsed() < max_time * 0.92])
        if len(f_simplex) < n + 1:
            return best
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[n]), lower, upper)
            fr = eval_func(xr)
            if elapsed() >= max_time * 0.95:
                break
            
            if fr < f_simplex[0]:
                # Expand
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n] = xe
                    f_simplex[n] = fe
                else:
                    simplex[n] = xr
                    f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr
                f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    # Outside contraction
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                else:
                    # Inside contraction
                    xc = np.clip(centroid + rho * (simplex[n] - centroid), lower, upper)
                fc = eval_func(xc)
                if fc < min(fr, f_simplex[n]):
                    simplex[n] = xc
                    f_simplex[n] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        if elapsed() >= max_time * 0.95:
                            return best
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = eval_func(simplex[i])
            
            # Convergence check
            if np.max(np.abs(simplex[n] - simplex[0])) < 1e-12:
                break
    
    return best
