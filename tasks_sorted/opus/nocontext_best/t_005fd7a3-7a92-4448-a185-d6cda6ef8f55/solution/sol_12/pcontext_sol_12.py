#Looking at the current algorithm, I need to improve convergence. Key issues:
#
#1. The SHADE implementation has bugs in index selection for mutation
#2. No dedicated local search phase (e.g., Nelder-Mead or CMA-ES style)
#3. Population size management could be improved (L-SHADE style reduction)
#4. Need a multi-phase approach: global exploration → refinement → local search
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
        return elapsed() < max_time * 0.97
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: L-SHADE ---
    init_pop_size = min(18 * dim, 300)
    min_pop_size = max(4, dim)
    pop_size = init_pop_size
    max_evals_shade = None  # we'll use time
    
    # Latin hypercube initialization
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, d] = lower[d] + (perm + np.random.rand(pop_size)) / pop_size * ranges[d]
    
    fitness = np.array([eval_func(population[i]) for i in range(pop_size) if time_left() or i == 0])
    actual = len(fitness)
    if actual < pop_size:
        population = population[:actual]
        pop_size = actual
    fitness = fitness[:pop_size]
    
    # Memory
    mem_size = max(5, dim)
    M_F = np.full(mem_size, 0.3)
    M_CR = np.full(mem_size, 0.8)
    mem_idx = 0
    
    archive = []
    max_archive = pop_size
    total_evals = pop_size
    max_total_evals_est = pop_size * 200  # rough estimate
    
    generation = 0
    
    while time_left() and elapsed() < max_time * 0.65:
        generation += 1
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        S_F, S_CR, S_w = [], [], []
        trials = np.empty_like(population)
        trial_fit = np.empty(pop_size)
        
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(p_min, int(0.2 * pop_size))
        
        for i in range(pop_size):
            if not time_left() or elapsed() >= max_time * 0.65:
                break
            
            ri = np.random.randint(mem_size)
            # Cauchy for F
            Fi = M_F[ri]
            while True:
                Fi_c = np.random.standard_cauchy() * 0.1 + Fi
                if Fi_c > 0:
                    break
            Fi = min(Fi_c, 1.0)
            
            # Normal for CR
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
            
            # current-to-pbest/1
            pi = np.random.randint(p_min, p_max + 1)
            p_best_idx = np.random.randint(pi)
            
            # r1 != i
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i:
                r1 += 1
            
            # r2 from pop + archive, != i, != r1
            total_pool = pop_size + len(archive)
            r2 = np.random.randint(total_pool - 2)
            candidates = [j for j in range(total_pool) if j != i and j != r1]
            r2 = candidates[r2 % len(candidates)]
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            jrand = np.random.randint(dim)
            cross = np.random.rand(dim) < CRi
            cross[jrand] = True
            trial = np.where(cross, mutant, population[i])
            
            # Bounce-back boundary
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i, d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i, d]) / 2
            
            t_f = eval_func(trial)
            total_evals += 1
            trials[i] = trial
            trial_fit[i] = t_f
            
            if t_f <= fitness[i]:
                if t_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    w = abs(fitness[i] - t_f)
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_w.append(w)
                population[i] = trial
                fitness[i] = t_f
        
        # Update memory
        if S_F:
            weights = np.array(S_w)
            weights /= weights.sum() + 1e-30
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(weights * sf * sf) / (np.sum(weights * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(weights * scr)
            mem_idx = (mem_idx + 1) % mem_size
        
        # L-SHADE population reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * (elapsed() / (max_time * 0.65)))))
        if new_pop_size < pop_size:
            sort_idx = np.argsort(fitness)
            population = population[sort_idx[:new_pop_size]]
            fitness = fitness[sort_idx[:new_pop_size]]
            pop_size = new_pop_size
            # Trim archive
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            max_archive = pop_size
    
    if best_params is None:
        return best
    
    # --- Phase 2: CMA-ES-like local search ---
    sigma = 0.1
    mean = best_params.copy()
    n = dim
    lam = max(4 + int(3 * np.log(n)), 8)
    mu = lam // 2
    
    weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights_cma = weights_raw / weights_raw.sum()
    mueff = 1.0 / np.sum(weights_cma ** 2)
    
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    
    pc = np.zeros(n)
    ps = np.zeros(n)
    
    # Use diagonal covariance for efficiency in high dim
    C_diag = (ranges * 0.1) ** 2  # diagonal elements
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    gen_cma = 0
    while time_left():
        gen_cma += 1
        sqrt_C = np.sqrt(np.maximum(C_diag, 1e-30))
        
        # Generate offspring
        arz = np.random.randn(lam, n)
        arx = np.empty((lam, n))
        for k in range(lam):
            arx[k] = mean + sigma * sqrt_C * arz[k]
        
        # Evaluate
        arfitness = np.empty(lam)
        evaluated = 0
        for k in range(lam):
            if not time_left():
                return best
            arfitness[k] = eval_func(arx[k])
            evaluated += 1
        
        # Sort
        idx = np.argsort(arfitness)
        arx = arx[idx]
        arz = arz[idx]
        
        # Recombination
        old_mean = mean.copy()
        mean = np.sum(weights_cma[:, None] * arx[:mu], axis=0)
        mean = np.clip(mean, lower, upper)
        
        zmean = np.sum(weights_cma[:, None] * arz[:mu], axis=0)
        
        # CSA
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen_cma + 1))) / chiN < 1.4 + 2.0 / (n + 1))
        
        # CMA
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / (sigma * sqrt_C + 1e-30)
        
        # Diagonal CMA update
        C_diag = ((1 - c1 - cmu_val) * C_diag +
                   c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * C_diag))
        for k in range(mu):
            C_diag += cmu_val * weights_cma[k] * ((arx[k] - old_mean) / (sigma + 1e-30)) ** 2
        
        C_diag = np.maximum(C_diag, 1e-30)
        
        # Sigma adaptation
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.max(ranges))
        
        if sigma < 1e-15:
            # Restart local search with perturbation
            sigma = 0.01
            mean = best_params.copy() + np.random.randn(n) * ranges * 0.001
            mean = np.clip(mean, lower, upper)
            C_diag = (ranges * 0.01) ** 2
            pc = np.zeros(n)
            ps = np.zeros(n)
    
    # --- Phase 3: Final Nelder-Mead style refinement ---
    if time_left() and best_params is not None:
        # Quick coordinate descent
        step = ranges * 0.001
        x = best_params.copy()
        fx = best
        for _ in range(5):
            if not time_left():
                break
            for d in range(dim):
                if not time_left():
                    break
                for sign in [1, -1]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step[d], lower[d], upper[d])
                    ft = eval_func(trial)
                    if ft < fx:
                        x = trial
                        fx = ft
            step *= 0.5
    
    return best
