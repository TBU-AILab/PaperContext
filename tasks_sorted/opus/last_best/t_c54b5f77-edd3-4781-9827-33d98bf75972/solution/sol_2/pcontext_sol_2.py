#Looking at the previous algorithm, I see several potential improvements:
#
#1. **Population size reduction (LPSR)** - gradually reduce population size like L-SHADE
#2. **Better CMA-ES integration** - add a proper CMA-ES phase for local refinement
#3. **Multiple restart strategies** with different scales
#4. **Pattern search** as additional local optimizer
#5. **Fix initialization bug** - the list comprehension with condition can miss evaluations
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: L-SHADE ---
    N_init = min(max(40, 10 * dim), 400)
    N_min = 4
    pop_size = N_init
    max_evals_shade = None  # we use time-based
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.zeros(pop_size)
    for i in range(pop_size):
        if elapsed() >= max_time * 0.08:
            fitness_vals[i] = float('inf')
        else:
            fitness_vals[i] = eval_f(population[i])
    
    # Remove unevaluated individuals
    valid = fitness_vals < float('inf')
    if np.sum(valid) < 4:
        # Not enough time even for init, do random search
        while elapsed() < max_time * 0.95:
            x = lower + np.random.random(dim) * ranges
            eval_f(x)
        return best
    
    population = population[valid]
    fitness_vals = fitness_vals[valid]
    pop_size = len(population)
    
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k_idx = 0
    
    archive = []
    archive_max = N_init
    
    p_min = max(2.0 / pop_size, 0.05)
    p_max = 0.2
    
    generation = 0
    stagnation_count = 0
    prev_best = best
    evals_at_shade_start = evals
    
    def shade_time_limit():
        return max_time * 0.72
    
    while elapsed() < shade_time_limit() and pop_size >= N_min:
        generation += 1
        
        sort_idx = np.argsort(fitness_vals)
        population = population[sort_idx]
        fitness_vals = fitness_vals[sort_idx]
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        trial_vectors = []
        trial_params = []
        
        for i in range(pop_size):
            if elapsed() >= shade_time_limit():
                break
            
            r = np.random.randint(H)
            
            # Cauchy for F
            for _ in range(20):
                Fi = M_F[r] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
            # Normal for CR
            CRi = np.clip(M_CR[r] + 0.1 * np.random.randn(), 0, 1)
            
            p = p_min + np.random.random() * (p_max - p_min)
            p_count = max(1, int(p * pop_size))
            x_pbest = population[np.random.randint(p_count)]
            
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            
            total = pop_size + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(total)
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (x_pbest - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            
            if f_trial <= new_fit[i]:
                if len(archive) < archive_max:
                    archive.append(population[i].copy())
                elif len(archive) > 0:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                
                delta = fitness_vals[i] - f_trial
                if delta > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness_vals = new_fit
        
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            
            mean_F = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            mean_CR = np.sum(weights * scr)
            
            M_F[k_idx] = mean_F
            M_CR[k_idx] = mean_CR
            k_idx = (k_idx + 1) % H
        
        # Linear population size reduction
        evals_used = evals - evals_at_shade_start
        time_ratio = elapsed() / shade_time_limit()
        new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * time_ratio)))
        
        if new_pop_size < pop_size:
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx[:new_pop_size]]
            fitness_vals = fitness_vals[sort_idx[:new_pop_size]]
            pop_size = new_pop_size
            # Trim archive
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(prev_best - best) < 1e-15:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 30 + dim:
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx]
            fitness_vals = fitness_vals[sort_idx]
            keep = max(2, pop_size // 4)
            for i in range(keep, pop_size):
                if elapsed() >= shade_time_limit():
                    break
                if np.random.random() < 0.5 and best_x is not None:
                    scale = 0.1 * ranges * np.random.random()
                    population[i] = clip(best_x + scale * np.random.randn(dim))
                else:
                    population[i] = lower + np.random.random(dim) * ranges
                fitness_vals[i] = eval_f(population[i])
            stagnation_count = 0
            archive = []

    # --- Phase 2: CMA-ES local search ---
    if best_x is not None and elapsed() < max_time * 0.95:
        sigma0 = 0.01 * np.mean(ranges)
        
        for restart in range(5):
            if elapsed() >= max_time * 0.95:
                break
            
            if restart == 0:
                x_mean = best_x.copy()
                sigma = sigma0
            else:
                sigma = sigma0 * (10 ** restart)
                x_mean = best_x.copy() + sigma * np.random.randn(dim) * 0.1
                x_mean = clip(x_mean)
            
            n = dim
            lam = max(4 + int(3 * np.log(n)), 8)
            mu = lam // 2
            
            weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights_cma = weights_raw / weights_raw.sum()
            mueff = 1.0 / np.sum(weights_cma**2)
            
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
            
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            if n <= 100:
                C = np.eye(n)
                use_full = True
            else:
                # Use sep-CMA for high dimensions
                D = np.ones(n)
                use_full = False
            
            chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            cma_gen = 0
            while elapsed() < max_time * 0.95:
                cma_gen += 1
                
                if use_full:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-20)
                        D_diag = np.sqrt(eigvals)
                        B = eigvecs
                    except:
                        break
                
                arz = np.random.randn(lam, n)
                arx = np.zeros((lam, n))
                
                for k_i in range(lam):
                    if use_full:
                        arx[k_i] = x_mean + sigma * (B @ (D_diag * arz[k_i]))
                    else:
                        arx[k_i] = x_mean + sigma * D * arz[k_i]
                    arx[k_i] = clip(arx[k_i])
                
                fit = np.array([eval_f(arx[k_i]) for k_i in range(lam) if elapsed() < max_time * 0.95]
                               + [float('inf')] * lam)[:lam]
                
                if elapsed() >= max_time * 0.95:
                    break
                
                idx = np.argsort(fit)
                
                x_old = x_mean.copy()
                x_mean = np.sum(weights_cma[:, None] * arx[idx[:mu]], axis=0)
                
                if use_full:
                    z_mean = np.sum(weights_cma[:, None] * arz[idx[:mu]], axis=0)
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_mean)
                else:
                    diff = (x_mean - x_old) / sigma
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / D
                
                hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(cma_gen+1))) / chiN < 1.4 + 2/(n+1))
                
                if use_full:
                    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ (D_diag * z_mean))
                    
                    artmp = (arx[idx[:mu]] - x_old) / sigma
                    C = (1 - c1 - cmu_val) * C + \
                        c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                        cmu_val * sum(weights_cma[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
                    
                    C = np.triu(C) + np.triu(C, 1).T
                else:
                    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / D
                    D = np.sqrt((1 - c1 - cmu_val) * D**2 + 
                                c1 * pc**2 + 
                                cmu_val * np.sum(weights_cma[:, None] * ((arx[idx[:mu]] - x_old)/sigma)**2, axis=0))
                    D = np.maximum(D, 1e-20)
                
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
                sigma = min(sigma, np.mean(ranges))
                
                if sigma < 1e-16 * np.mean(ranges):
                    break
                if cma_gen > 100 + 50 * n / lam:
                    break
    
    # --- Phase 3: Quick pattern search refinement ---
    if best_x is not None and elapsed() < max_time * 0.98:
        x_curr = best_x.copy()
        f_curr = best
        step = 0.001 * ranges
        
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                
                x_try = x_curr.copy()
                x_try[d] += step[d]
                f_try = eval_f(x_try)
                if f_try < f_curr:
                    x_curr = x_try
                    f_curr = f_try
                    improved = True
                    continue
                
                x_try = x_curr.copy()
                x_try[d] -= step[d]
                f_try = eval_f(x_try)
                if f_try < f_curr:
                    x_curr = x_try
                    f_curr = f_try
                    improved = True
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break
    
    return best
