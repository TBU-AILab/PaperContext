#Looking at the results, the best algorithm (11.70) uses SHADE + Nelder-Mead + coordinate search. The CMA-ES version (16.40) performed worse, likely due to implementation issues. I'll improve by:
#
#1. **Better CMA-ES implementation** with proper eigendecomposition caching
#2. **Smarter time allocation** - more time for local search
#3. **Multiple diverse restarts** from different promising regions
#4. **Improved SHADE with better F generation** (avoid infinite loop)
#5. **Adaptive restart of NM/CMA-ES when stagnating**
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
    
    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: LHS Initial Sampling ---
    pop_size = min(max(30, 12 * dim), 250)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_f(population[i])
    
    # --- Phase 2: SHADE ---
    H = 50
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    archive_max = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    min_pop = max(6, dim)
    init_pop_size = pop_size
    
    de_end_frac = 0.50
    
    while elapsed() < max_time * de_end_frac:
        generation += 1
        S_F, S_CR, S_df = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        sorted_indices = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * (de_end_frac - 0.01):
                break
            
            ri = np.random.randint(0, H)
            # Generate F
            Fi = -1
            for _ in range(20):
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.05
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # p-best
            p = max(2, int(0.11 * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, p)]
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            pool = pop_size + len(archive)
            r2 = np.random.randint(0, pool)
            attempts = 0
            while (r2 == i or r2 == r1) and attempts < 20:
                r2 = np.random.randint(0, pool)
                attempts += 1
            
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - x_r2)
            
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(fitness[i] - f_trial)
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    elif archive:
                        archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        if S_F:
            w = np.array(S_df)
            w = w / (w.sum() + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        if best >= prev_best - 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # L-SHADE population reduction
        if generation % 10 == 0 and pop_size > min_pop:
            new_size = max(min_pop, pop_size - max(1, pop_size // 15))
            if new_size < pop_size:
                keep = np.argsort(fitness)[:new_size]
                population = population[keep]
                fitness = fitness[keep]
                pop_size = new_size
        
        if stagnation > 25:
            stagnation = 0
            n_replace = max(1, pop_size // 3)
            worst_indices = np.argsort(fitness)[-n_replace:]
            for idx in worst_indices:
                if np.random.random() < 0.7 and best_x is not None:
                    sigma = 0.05 * ranges * (0.2 + 0.8 * np.random.random())
                    population[idx] = best_x + np.random.randn(dim) * sigma
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = eval_f(population[idx])
    
    # Collect top candidates
    top_k = min(8, pop_size)
    top_indices = np.argsort(fitness)[:top_k]
    candidates = [(fitness[i], population[i].copy()) for i in top_indices]
    if best_x is not None:
        candidates.insert(0, (best, best_x.copy()))
    candidates.sort(key=lambda c: c[0])
    # Deduplicate
    unique_cands = [candidates[0]]
    for c in candidates[1:]:
        if all(np.linalg.norm(c[1] - u[1]) > 1e-8 * np.linalg.norm(ranges) for u in unique_cands):
            unique_cands.append(c)
    candidates = unique_cands[:6]
    
    # --- Phase 3: Nelder-Mead on multiple starts ---
    def nelder_mead(x0, time_limit, scale_factor=0.05):
        n = dim
        ns = n + 1
        simplex = np.zeros((ns, n))
        simplex[0] = x0.copy()
        scale = scale_factor * ranges
        for i in range(1, ns):
            simplex[i] = x0.copy()
            idx = (i - 1) % n
            simplex[i][idx] += scale[idx] * (1 if np.random.random() > 0.5 else -1)
            simplex[i] = np.clip(simplex[i], lower, upper)
        
        fs = np.array([eval_f(simplex[i]) for i in range(ns)])
        alpha, gamma, rho, sigma_s = 1.0, 2.0, 0.5, 0.5
        no_imp = 0
        
        while elapsed() < time_limit:
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            
            centroid = simplex[:-1].mean(axis=0)
            old_best_f = fs[0]
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = eval_f(xr)
            
            if fr < fs[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], fs[-1] = xe, fe
                else:
                    simplex[-1], fs[-1] = xr, fr
            elif fr < fs[-2]:
                simplex[-1], fs[-1] = xr, fr
            else:
                if fr < fs[-1]:
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1], fs[-1] = xc, fc
                    else:
                        for j in range(1, ns):
                            simplex[j] = simplex[0] + sigma_s * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], lower, upper)
                            fs[j] = eval_f(simplex[j])
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, lower, upper)
                    fc = eval_f(xc)
                    if fc < fs[-1]:
                        simplex[-1], fs[-1] = xc, fc
                    else:
                        for j in range(1, ns):
                            simplex[j] = simplex[0] + sigma_s * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], lower, upper)
                            fs[j] = eval_f(simplex[j])
            
            if fs[np.argsort(fs)[0]] >= old_best_f - 1e-15:
                no_imp += 1
            else:
                no_imp = 0
            if no_imp > 30 * dim:
                break
            diam = np.max(np.abs(simplex - simplex[0]))
            if diam < 1e-14 * np.max(ranges):
                break
    
    # Allocate time for NM across candidates
    nm_end = max_time * 0.88
    time_remaining = nm_end - elapsed()
    if time_remaining > 0 and candidates:
        # Give more time to the best candidate
        weights_alloc = [3.0] + [1.0] * (len(candidates) - 1)
        total_w = sum(weights_alloc)
        
        for ci, (f_c, x_c) in enumerate(candidates):
            if elapsed() >= nm_end:
                break
            alloc = time_remaining * weights_alloc[ci] / total_w
            tl = min(elapsed() + alloc, nm_end)
            sf = 0.02 if ci == 0 else 0.05
            nelder_mead(x_c, tl, scale_factor=sf)
    
    # --- Phase 4: CMA-ES from best ---
    def run_cmaes(x0, sigma0, time_limit):
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        gen_count = 0
        
        while elapsed() < time_limit:
            gen_count += 1
            
            # Sample
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for j in range(lam):
                arx[j] = mean + sigma * (B @ (D * arz[j]))
            arx = np.clip(arx, lower, upper)
            
            arfit = np.array([eval_f(arx[j]) for j in range(lam)])
            if elapsed() >= time_limit:
                break
            
            idx = np.argsort(arfit)
            old_mean = mean.copy()
            mean = np.zeros(n)
            for j in range(mu):
                mean += w[j] * arx[idx[j]]
            
            zmean = np.zeros(n)
            for j in range(mu):
                zmean += w[j] * arz[idx[j]]
            
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (B @ zmean)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*gen_count)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
            
            # Rank-mu update
            artmp = np.zeros((mu, n))
