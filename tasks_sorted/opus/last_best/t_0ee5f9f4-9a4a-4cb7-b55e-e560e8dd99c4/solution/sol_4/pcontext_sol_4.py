#Looking at the previous algorithms, I can identify several issues:
#
#1. **Coordinate descent in Phase 3 wastes evaluations** by calling `eval_func(x)` redundantly
#2. **CMA-ES restart strategy is suboptimal** - needs both large-population and small-population restarts (BIPOP)
#3. **Missing Differential Evolution** which can complement CMA-ES well
#4. **Nelder-Mead local search** was better than coordinate descent
#5. **Need more diverse initial sampling strategies**
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
    
    def time_ok(fraction=0.98):
        return elapsed() < max_time * fraction
    
    def eval_func(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Phase 1: Latin Hypercube Sampling
    n_init = min(max(20 * dim, 100), 500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        init_pop[:, d] = lower[d] + (perm + np.random.random(n_init)) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    actual_init = 0
    for i in range(n_init):
        if not time_ok(0.75):
            break
        init_fitness[i] = eval_func(init_pop[i])
        actual_init = i + 1
    
    sorted_idx = np.argsort(init_fitness[:actual_init])
    
    # Archive of good solutions
    archive_size = min(40, actual_init)
    archive = init_pop[sorted_idx[:archive_size]].copy()
    archive_fit = init_fitness[sorted_idx[:archive_size]].copy()
    
    def update_archive(x, f):
        nonlocal archive, archive_fit
        worst_idx = np.argmax(archive_fit)
        if f < archive_fit[worst_idx]:
            archive[worst_idx] = x.copy()
            archive_fit[worst_idx] = f
    
    # Phase 2: Differential Evolution
    pop_size = min(max(8 * dim, 40), 100)
    if actual_init >= pop_size:
        pop = init_pop[sorted_idx[:pop_size]].copy()
        pop_fit = init_fitness[sorted_idx[:pop_size]].copy()
    else:
        pop = np.zeros((pop_size, dim))
        pop_fit = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if i < actual_init:
                pop[i] = init_pop[sorted_idx[i]].copy()
                pop_fit[i] = init_fitness[sorted_idx[i]]
            else:
                pop[i] = lower + np.random.rand(dim) * ranges
                if time_ok(0.75):
                    pop_fit[i] = eval_func(pop[i])
    
    # Adaptive DE parameters (SHADE-like)
    mem_size = 20
    mem_F = np.full(mem_size, 0.5)
    mem_CR = np.full(mem_size, 0.5)
    mem_idx = 0
    
    de_gens = 0
    while time_ok(0.55):
        S_F = []
        S_CR = []
        S_df = []
        
        # Current-to-pbest/1
        p_best_rate = max(0.05, 0.2 - 0.15 * de_gens / (de_gens + 100))
        p_best_size = max(1, int(pop_size * p_best_rate))
        
        sorted_pop_idx = np.argsort(pop_fit)
        
        for i in range(pop_size):
            if not time_ok(0.55):
                break
            
            ri = np.random.randint(mem_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + mem_F[ri], 0, 1)
            CRi = np.clip(np.random.randn() * 0.1 + mem_CR[ri], 0, 1)
            
            # current-to-pbest/1
            p_best_idx = sorted_pop_idx[np.random.randint(p_best_size)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # Use archive + population for r2
            combined = np.vstack([pop, archive])
            r2 = np.random.randint(len(combined))
            while r2 == i or r2 == r1:
                r2 = np.random.randint(len(combined))
            
            mutant = pop[i] + Fi * (pop[p_best_idx] - pop[i]) + Fi * (pop[r1] - combined[r2])
            
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounce-back boundary
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.rand() * (pop[i][j] - lower[j])
                elif trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.rand() * (upper[j] - pop[i][j])
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_func(trial)
            update_archive(trial, f_trial)
            
            if f_trial <= pop_fit[i]:
                df = pop_fit[i] - f_trial
                if df > 0:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(df)
                pop[i] = trial
                pop_fit[i] = f_trial
        
        # Update memory
        if len(S_F) > 0:
            weights_s = np.array(S_df)
            weights_s = weights_s / (np.sum(weights_s) + 1e-30)
            mean_F = np.sum(weights_s * np.array(S_F)**2) / (np.sum(weights_s * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights_s * np.array(S_CR))
            mem_F[mem_idx] = mean_F
            mem_CR[mem_idx] = mean_CR
            mem_idx = (mem_idx + 1) % mem_size
        
        de_gens += 1
    
    # Phase 3: CMA-ES with restarts from best solutions
    def cma_es(x0, sigma0, lam=None):
        nonlocal best, best_params
        n = dim
        if lam is None:
            lam = 4 + int(3 * np.log(n))
        lam = max(lam, 6)
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
        
        sigma = sigma0
        xmean = x0.copy()
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        eigeneval = 0
        counteval = 0
        gen = 0
        stag_count = 0
        best_in_run = float('inf')
        
        while time_ok(0.96):
            arx = np.zeros((lam, n))
            arfitness = np.full(lam, float('inf'))
            
            for k in range(lam):
                if not time_ok(0.96):
                    return
                z = np.random.randn(n)
                arx[k] = xmean + sigma * (B @ (D * z))
                for dd in range(n):
                    while arx[k, dd] < lower[dd] or arx[k, dd] > upper[dd]:
                        if arx[k, dd] < lower[dd]:
                            arx[k, dd] = 2 * lower[dd] - arx[k, dd]
                        if arx[k, dd] > upper[dd]:
                            arx[k, dd] = 2 * upper[dd] - arx[k, dd]
                arx[k] = np.clip(arx[k], lower, upper)
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            for idx in arindex[:3]:
                update_archive(arx[idx], arfitness[idx])
            
            gen_best = arfitness[arindex[0]]
            if gen_best < best_in_run - 1e-10 * (abs(best_in_run) + 1e-20):
                best_in_run = gen_best
                stag_count = 0
            else:
                stag_count += 1
            
            xold = xmean.copy()
            xmean = np.sum(weights[:, None] * arx[arindex[:mu]], axis=0)
            
            diff = (xmean - xold) / (sigma + 1e-30)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            
            norm_ps = np.linalg.norm(ps)
            hsig = float(norm_ps / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[arindex[:mu]] - xold) / (sigma + 1e-30)
            
            C = ((1 - c1 - cmu_val) * C 
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu_val * (weights[:, None] * artmp).T @ artmp)
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            gen += 1
            
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                eigeneval = counteval
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
            if stag_count > 10 + 30 * n / lam:
                return
            if np.max(D) > 1e7 * np.min(D):
                return
    
    base_lam = 4 + int(3 * np.log(dim))
    restart = 0
    large_pop_factor = 0
    
    # Get top solutions from DE + archive
    all_candidates = np.vstack([pop, archive])
    all_fits = np.concatenate([pop_fit, archive_fit])
    cand_order = np.argsort(all_fits)
    
    while time_ok(0.88):
        if restart < min(3, len(cand_order)):
            x0 = all_candidates[cand_order[restart]].copy()
            sigma0 = 0.2 * np.mean(ranges)
            lam_r = base_lam
        elif restart % 4 == 0:
            large_pop_factor += 1
            x0 = lower + np.random.rand(dim) * ranges
            sigma0 = 0.4 * np.mean(ranges)
            lam_r = min(base_lam * (2 ** large_pop_factor), 200)
        elif restart % 4 == 1:
            x0 = best_params.copy() + 0.01 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.03 * np.mean(ranges)
            lam_r = base_lam
        elif restart % 4 == 2:
            idx_a = np.argmin(archive_fit)
            x0 = archive[idx_a].copy() + 0.05 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.1 * np.mean(ranges)
            lam_r = base_lam
        else:
            x0 = best_params.copy() + 0.15 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            sigma0 = 0.2 * np.mean(ranges)
            lam_r = base_lam * 2
        
        cma_es(x0, sigma0, lam_r)
        restart += 1
    
    # Phase 4: Nelder-Mead polish
    if best_params is not None and time_ok(0.92):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex_f = np.full(n + 1, float('inf'))
        simplex[0] = best_params.copy()
        simplex_f[0] = best
        
        step = 0.002 * ranges
        for i in range(n):
            if not time_ok(0.97):
                break
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] = np.clip(simplex[i+1][i] + step[i], lower[i], upper[i])
            simplex_f[i+1] = eval_func(simplex[i+1])
        
        for _ in range(50000):
            if not time_ok(0.998):
                break
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if fr < simplex_f[0]:
                xe = np.clip(centroid + 2.0 * (xr - centroid), lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1], simplex_f[-1] = xe, fe
                else:
                    simplex[-1], simplex_f[-1] = xr, fr
            elif fr < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = xr, fr
            else:
                if fr < simplex_f[-1]:
                    xc = np.clip(centroid + 0.5 * (xr - centroid), lower, upper)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1], simplex_f[-1] = xc, fc
                        continue
                else:
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lower, upper)
                    fc = eval_func(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = xc, fc
                        continue
                # Shrink
                for i in range(1, n+1):
                    if not time_ok(0.998):
                        break
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    simplex_f[i] = eval_func(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                break
    
    return best
