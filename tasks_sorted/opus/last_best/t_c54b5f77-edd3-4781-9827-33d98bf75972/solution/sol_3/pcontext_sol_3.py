#Looking at the previous algorithms, I can identify several improvements:
#
#1. **Add CMA-ES as a powerful local optimizer** (the best algorithm lacks it)
#2. **Linear population size reduction (LPSR)** for SHADE
#3. **Multiple CMA-ES restarts with increasing sigma**
#4. **Better time allocation**
#5. **Improved stagnation handling**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: L-SHADE ---
    N_init = min(max(30, 8 * dim), 300)
    N_min = 4
    pop_size = N_init
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.10:
            break
        fitness_vals[i] = eval_f(population[i])
    
    # Remove unevaluated
    valid = fitness_vals < float('inf')
    if np.sum(valid) < N_min:
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
    shade_start_time = elapsed()
    shade_time_limit = max_time * 0.60
    
    while elapsed() < shade_time_limit and pop_size >= N_min:
        generation += 1
        
        sort_idx = np.argsort(fitness_vals)
        population = population[sort_idx]
        fitness_vals = fitness_vals[sort_idx]
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= shade_time_limit:
                break
            
            r = np.random.randint(H)
            
            for _ in range(20):
                Fi = M_F[r] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
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
            M_F[k_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[k_idx] = np.sum(weights * scr)
            k_idx = (k_idx + 1) % H
        
        # LPSR
        time_ratio = (elapsed() - shade_start_time) / (shade_time_limit - shade_start_time + 1e-30)
        time_ratio = min(time_ratio, 1.0)
        new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * time_ratio)))
        
        if new_pop_size < pop_size:
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx[:new_pop_size]]
            fitness_vals = fitness_vals[sort_idx[:new_pop_size]]
            pop_size = new_pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(prev_best - best) < 1e-15:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 20 + dim:
            sort_idx = np.argsort(fitness_vals)
            population = population[sort_idx]
            fitness_vals = fitness_vals[sort_idx]
            keep = max(2, pop_size // 5)
            for i in range(keep, pop_size):
                if elapsed() >= shade_time_limit:
                    break
                if np.random.random() < 0.5 and best_x is not None:
                    sigma = 0.05 * ranges * (np.random.random() + 0.1)
                    population[i] = clip(best_x + sigma * np.random.randn(dim))
                else:
                    population[i] = lower + np.random.random(dim) * ranges
                fitness_vals[i] = eval_f(population[i])
            stagnation_count = 0
            archive = []

    # --- Phase 2: CMA-ES with restarts ---
    if best_x is not None:
        for restart in range(10):
            if elapsed() >= max_time * 0.96:
                break
            
            n = dim
            if restart == 0:
                x_mean = best_x.copy()
                sigma = 0.005 * np.mean(ranges)
            elif restart == 1:
                x_mean = best_x.copy()
                sigma = 0.05 * np.mean(ranges)
            elif restart == 2:
                x_mean = best_x.copy()
                sigma = 0.2 * np.mean(ranges)
            else:
                # broader restarts with perturbation
                x_mean = clip(best_x + 0.3 * ranges * np.random.randn(dim))
                sigma = 0.1 * np.mean(ranges) * (restart - 1)
                sigma = min(sigma, 0.5 * np.mean(ranges))
            
            lam = max(4 + int(3 * np.log(n)), 8)
            mu = lam // 2
            
            w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w_raw / w_raw.sum()
            mueff = 1.0 / np.sum(w**2)
            
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu_c = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
            chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            use_full = (n <= 80)
            if use_full:
                C = np.eye(n)
                eigeneval = 0
                B = np.eye(n)
                D_diag = np.ones(n)
            else:
                diag_C = np.ones(n)
            
            cma_gen = 0
            no_improve = 0
            cma_best_before = best
            
            while elapsed() < max_time * 0.96:
                cma_gen += 1
                
                if use_full and cma_gen % max(1, int(1/(10*n*(c1+cmu_c)))) == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-20)
                        D_diag = np.sqrt(eigvals)
                        B = eigvecs
                    except:
                        break
                
                arz = np.random.randn(lam, n)
                arx = np.zeros((lam, n))
                
                for ki in range(lam):
                    if use_full:
                        arx[ki] = x_mean + sigma * (B @ (D_diag * arz[ki]))
                    else:
                        arx[ki] = x_mean + sigma * np.sqrt(diag_C) * arz[ki]
                    arx[ki] = clip(arx[ki])
                
                fit = np.full(lam, float('inf'))
                for ki in range(lam):
                    if elapsed() >= max_time * 0.96:
                        break
                    fit[ki] = eval_f(arx[ki])
                
                if elapsed() >= max_time * 0.96:
                    break
                
                idx = np.argsort(fit)
                
                x_old = x_mean.copy()
                x_mean = np.sum(w[:, None] * arx[idx[:mu]], axis=0)
                
                diff = (x_mean - x_old) / sigma
                
                if use_full:
                    invsqrtC_diff = np.linalg.solve(B @ np.diag(D_diag), B.T @ diff) if n <= 80 else diff
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC_diff
                else:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / np.sqrt(diag_C + 1e-30)
                
                hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(cma_gen))) / chiN < 1.4 + 2/(n+1))
                
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                if use_full:
                    artmp = (arx[idx[:mu]] - x_old) / sigma
                    C = (1 - c1 - cmu_c) * C + \
                        c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                        cmu_c * sum(w[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
                    C = np.triu(C) + np.triu(C, 1).T
                else:
                    diag_C = (1 - c1 - cmu_c) * diag_C + \
                             c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diag_C) + \
                             cmu_c * np.sum(w[:, None] * ((arx[idx[:mu]] - x_old)/sigma)**2, axis=0)
                    diag_C = np.maximum(diag_C, 1e-20)
                
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
                sigma = min(sigma, np.mean(ranges) * 2)
                
                if sigma < 1e-16 * np.mean(ranges):
                    break
                if cma_gen > 200 + 100 * n / lam:
                    break
                
                if best < cma_best_before - 1e-12:
                    no_improve = 0
                    cma_best_before = best
                else:
                    no_improve += 1
                if no_improve > 30 + 10 * n / lam:
                    break

    # --- Phase 3: Pattern search ---
    if best_x is not None and elapsed() < max_time * 0.99:
        x_c = best_x.copy()
        f_c = best
        step = 0.001 * ranges
        
        while elapsed() < max_time * 0.99:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.99:
                    break
                for sign in [1, -1]:
                    x_t = x_c.copy()
                    x_t[d] += sign * step[d]
                    f_t = eval_f(x_t)
                    if f_t < f_c:
                        x_c = clip(x_t)
                        f_c = f_t
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break
    
    return best
