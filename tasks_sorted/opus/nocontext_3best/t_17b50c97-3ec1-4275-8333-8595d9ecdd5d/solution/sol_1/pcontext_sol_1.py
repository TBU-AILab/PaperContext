#Looking at the current best algorithm, I can see it uses LHS initialization + CMA-ES + pattern search. Let me improve it with:
#
#1. Better initialization with Sobol-like quasi-random sampling
#2. More robust CMA-ES with better restart strategy (IPOP-CMA-ES)
#3. Differential Evolution as an additional global search phase
#4. Better time management
#5. Nelder-Mead simplex for local refinement
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
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling initialization ---
    n_init = min(max(30 * dim, 200), 1000)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if time_left() < max_time * 0.05:
            break
        init_fitness[i] = evaluate(init_points[i])
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: Differential Evolution ---
    pop_size = min(max(10 * dim, 40), 200)
    # Initialize population from best init points + random
    n_elite = min(pop_size // 2, np.sum(init_fitness < float('inf')))
    pop = np.zeros((pop_size, dim))
    pop_fit = np.full(pop_size, float('inf'))
    
    for i in range(n_elite):
        idx = sorted_idx[i]
        pop[i] = init_points[idx].copy()
        pop_fit[i] = init_fitness[idx]
    for i in range(n_elite, pop_size):
        pop[i] = lower + np.random.random(dim) * ranges
        pop_fit[i] = evaluate(pop[i])
    
    # DE parameters with adaptive F and CR
    F_base = 0.5
    CR_base = 0.9
    
    de_budget = 0.40  # fraction of total time for DE
    
    generation = 0
    while time_left() > max_time * (1.0 - de_budget) and time_left() > 1.0:
        generation += 1
        # Adaptive parameters
        F = F_base + 0.1 * np.random.randn()
        F = np.clip(F, 0.2, 1.0)
        
        for i in range(pop_size):
            if time_left() < max_time * (1.0 - de_budget):
                break
            
            # DE/current-to-pbest/1 strategy
            p_best_size = max(2, pop_size // 5)
            p_best_idx = np.argsort(pop_fit)[:p_best_size]
            pbest = pop[p_best_idx[np.random.randint(p_best_size)]]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.1, 1.0)
            
            mutant = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - pop[r2])
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CR
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (pop[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - pop[i][d])
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            if f_trial <= pop_fit[i]:
                pop[i] = trial
                pop_fit[i] = f_trial
    
    # --- Phase 3: CMA-ES from best solution ---
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
        
        stag_count = 0
        prev_best_gen = float('inf')
        
        while elapsed() < end_time and elapsed() < max_time * 0.95:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= end_time or elapsed() >= max_time * 0.95:
                    return
                fitnesses[k] = evaluate(arx[k])
            
            arindex = np.argsort(fitnesses)
            best_gen = fitnesses[arindex[0]]
            
            if best_gen >= prev_best_gen:
                stag_count += 1
            else:
                stag_count = 0
            prev_best_gen = best_gen
            
            if stag_count > 10 + 30 * n / lam:
                return
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for k in range(mu):
                C += cmu_val * weights[k] * np.outer(artmp[k], artmp[k])
            
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
    
    # IPOP-CMA-ES: multiple restarts with increasing population
    remaining = time_left()
    if remaining > 1.0 and best_params is not None:
        n_restarts = 5
        sigma_base = 0.3 * np.max(ranges)
        for r in range(n_restarts):
            t = time_left()
            if t < 0.5:
                break
            budget = t / max(1, n_restarts - r)
            
            if r == 0:
                x0 = best_params.copy()
                s0 = sigma_base
            else:
                # Restart from perturbed best or random top point
                top_k = min(10, len(sorted_idx))
                ridx = sorted_idx[np.random.randint(top_k)]
                x0 = init_points[ridx] + np.random.randn(dim) * sigma_base * 0.3
                x0 = clip(x0)
                s0 = sigma_base * (0.5 + 0.5 * np.random.random())
            
            cmaes_run(x0, s0, budget)
    
    # --- Phase 4: Nelder-Mead refinement ---
    if time_left() > 0.5 and best_params is not None:
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        step = 0.01 * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([evaluate(clip(simplex[i])) for i in range(n + 1)])
        
        while time_left() > 0.2:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = evaluate(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = evaluate(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                break
    
    return best
