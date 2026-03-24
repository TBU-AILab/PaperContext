#Looking at the current algorithm, I can see it's already a solid L-SHADE with local search. Let me analyze potential improvements:
#
#1. **Better initial sampling** with Sobol-like sequences
#2. **Improved DE with current-to-pbest/1 with archive and rank-based adaptation**
#3. **CMA-ES local search** instead of just coordinate descent
#4. **Adaptive time budgeting**
#5. **Opposition-based learning for restarts**
#6. **More aggressive local search with multiple strategies**
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
    
    def time_left():
        return max_time - elapsed()
    
    def clip_b(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_x, evals
        x = clip_b(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Initialization with quasi-random + opposition ---
    init_pop_size = min(max(40, 15 * dim), 400)
    pop_size = init_pop_size
    min_pop_size = max(4, dim)
    
    # Latin Hypercube Sampling
    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = eval_func(population[i])
    
    # Opposition-based population: evaluate opposites, keep best
    opp_pop = lower + upper - population
    opp_fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.85:
            break
        opp_fitness[i] = eval_func(opp_pop[i])
    
    # Merge and keep best pop_size
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.concatenate([fitness, opp_fitness])
    sidx = np.argsort(all_fit)[:pop_size]
    population = all_pop[sidx]
    fitness = all_fit[sidx]

    # --- Phase 2: L-SHADE with enhancements ---
    H = 8
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    max_archive = pop_size
    
    no_improve_count = 0
    prev_best = best
    
    de_time_budget = max_time * 0.78
    
    generation = 0
    while elapsed() < de_time_budget:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.1 * pop_size))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Pre-generate parameters for speed
        ri_all = np.random.randint(0, H, pop_size)
        
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            ri = ri_all[i]
            
            # Generate F via Cauchy
            F = -1
            for _ in range(10):
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F > 0:
                    break
            if F <= 0:
                F = 0.01
            F = min(F, 1.0)
            
            # Generate CR
            if M_CR[ri] < 0:
                CR = 0.0
            else:
                CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            # p-best index (rank-greedily weighted)
            pi = np.random.randint(0, p_best_size)
            p_best_idx = sorted_idx[pi]
            
            # r1 != i
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            
            # r2 from pop + archive, != i, != r1
            combined_size = pop_size + len(archive)
            r2 = i
            for _ in range(30):
                r2 = np.random.randint(0, combined_size)
                if r2 != i and r2 != r1:
                    break
            
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + F * (population[p_best_idx] - population[i]) + F * (population[r1] - x_r2)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back
            out_low = trial < lower
            out_high = trial > upper
            if np.any(out_low):
                trial[out_low] = (lower[out_low] + population[i][out_low]) / 2.0
            if np.any(out_high):
                trial[out_high] = (upper[out_high] + population[i][out_high]) / 2.0
            trial = clip_b(trial)
            
            trial_f = eval_func(trial)
            
            if trial_f <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_f
                if trial_f < fitness[i]:
                    delta = fitness[i] - trial_f
                    S_F.append(F)
                    S_CR.append(CR)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (np.sum(weights) + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            
            M_F[k] = np.sum(weights * sf ** 2) / (np.sum(weights * sf) + 1e-30)
            mean_cr = np.sum(weights * scr)
            if np.max(scr) == 0:
                M_CR[k] = -1.0
            else:
                M_CR[k] = mean_cr
            k = (k + 1) % H
        
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # Population size reduction
        frac = elapsed() / de_time_budget
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * frac)))
        
        if new_pop_size < pop_size:
            sidx2 = np.argsort(fitness)[:new_pop_size]
            population = population[sidx2]
            fitness = fitness[sidx2]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation restart
        if no_improve_count > 40 and pop_size > min_pop_size + 2:
            no_improve_count = 0
            sidx2 = np.argsort(fitness)
            n_restart = max(1, pop_size // 3)
            for kk in range(n_restart):
                idx = sidx2[pop_size - 1 - kk]
                if np.random.random() < 0.3:
                    population[idx] = lower + upper - best_x + np.random.randn(dim) * ranges * 0.01
                elif np.random.random() < 0.5:
                    population[idx] = best_x + np.random.randn(dim) * ranges * 0.05
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip_b(population[idx])
                fitness[idx] = eval_func(population[idx])

    # --- Phase 3: CMA-ES-like local search ---
    def mini_cma(x0, sigma0, budget_frac):
        nonlocal best, best_x
        n = len(x0)
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        sigma = sigma0
        
        xmean = x0.copy()
        fmean = eval_func(xmean)
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        eigeneval = 0
        
        deadline = elapsed() + max_time * budget_frac
        
        gen = 0
        while elapsed() < deadline:
            gen += 1
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for kk in range(lam):
                arx[kk] = xmean + sigma * (B @ (D * arz[kk]))
                arx[kk] = clip_b(arx[kk])
            
            arfitness = np.array([eval_func(arx[kk]) for kk in range(lam)])
            
            idx_sort = np.argsort(arfitness)
            
            xold = xmean.copy()
            xmean = np.zeros(n)
            for kk in range(mu):
                xmean += weights[kk] * arx[idx_sort[kk]]
            
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (B @ (1.0/(D+1e-30) * (B.T @ (xmean - xold)))) / (sigma + 1e-30)
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean - xold) / (sigma + 1e-30)
            
            artmp = (1.0/(sigma+1e-30)) * np.array([arx[idx_sort[kk]] - xold for kk in range(mu)])
            
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1-hsig) * cc*(2-cc) * C)
            for kk in range(mu):
                C += cmu_val * weights[kk] * np.outer(artmp[kk], artmp[kk])
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            eigeneval += lam
            if eigeneval >= lam * (n+1):
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    Dvals, Bmat = np.linalg.eigh(C)
                    Dvals = np.maximum(Dvals, 1e-20)
                    D = np.sqrt(Dvals)
                    B = Bmat
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            
            if sigma * np.max(D) < 1e-14 * np.max(ranges):
                break
    
    # Run CMA on best candidates
    sidx_final = np.argsort(fitness)
    candidates = []
    if best_x is not None:
        candidates.append(best_x.copy())
    for i in range(min(3, pop_size)):
        c = population[sidx_final[i]]
        if not any(np.allclose(c, cc) for cc in candidates):
            candidates.append(c.copy())
    
    time_per_cand = max(0.01, time_left() * 0.8 / max(len(candidates), 1))
    for cand in candidates:
        if time_left() < max_time * 0.02:
            break
        sigma0 = 0.05 * np.max(ranges)
        mini_cma(cand, sigma0, time_per_cand / max_time)
    
    # --- Phase 4: Final coordinate descent polish ---
    if best_x is not None and time_left() > max_time * 0.01:
        x = best_x.copy()
        fx = best
        step = 0.01 * ranges.copy()
        while time_left() > max_time * 0.002:
            improved = False
            for j in range(dim):
                if time_left() < max_time * 0.002:
                    return best
                t1 = x.copy(); t1[j] = np.clip(t1[j] + step[j], lower[j], upper[j])
                f1 = eval_func(t1)
                if f1 < fx:
                    fx = f1; x = t1; step[j] *= 1.5; improved = True; continue
                t2 = x.copy(); t2[j] = np.clip(t2[j] - step[j], lower[j], upper[j])
                f2 = eval_func(t2)
                if f2 < fx:
                    fx = f2; x = t2; step[j] *= 1.5; improved = True; continue
                step[j] *= 0.5
            if not improved and np.max(step / ranges) < 1e-15:
                break
    
    return best
