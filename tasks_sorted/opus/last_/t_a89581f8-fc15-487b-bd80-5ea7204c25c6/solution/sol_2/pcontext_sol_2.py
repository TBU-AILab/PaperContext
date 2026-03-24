#Looking at the previous algorithm, it's a SHADE-based DE with local search. I'll improve it with several enhancements:
#
#1. Better local search (Nelder-Mead style simplex + pattern search)
#2. CMA-ES as an additional search strategy
#3. Better restart mechanism
#4. More efficient time management
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_pos = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.95 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_pos
        v = func(x)
        if v < best:
            best = v
            best_pos = x.copy()
        return v
    
    # --- LHS initialization ---
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    # --- CMA-ES ---
    def run_cmaes(x0, sigma0, budget_time):
        nonlocal best, best_pos
        t_start = elapsed()
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_ = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dim
        use_sep = (n > 50)
        
        if use_sep:
            C_diag = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        
        counteval = 0
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            # Sample
            arx = np.zeros((lam, n))
            arz = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.2:
                    return
                arz[k] = np.random.randn(n)
                if use_sep:
                    arx[k] = clip(mean + sigma * np.sqrt(C_diag) * arz[k])
                else:
                    arx[k] = clip(mean + sigma * (B @ (D * arz[k])))
                arfitness[k] = eval_f(arx[k])
                counteval += 1
            
            idx = np.argsort(arfitness)
            
            # Recombination
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            mean = clip(mean)
            
            # CSA
            if use_sep:
                invsqrt_diag = 1.0 / np.sqrt(C_diag + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * invsqrt_diag * (mean - old_mean) / sigma
            else:
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ ((mean - old_mean) / sigma))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * ((mean - old_mean) / sigma)
            
            # Covariance update
            if use_sep:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C_diag = (1 - c1 - cmu_) * C_diag + c1 * (pc**2 + (1-hsig)*cc*(2-cc)*C_diag) + cmu_ * np.sum(weights[:, None] * artmp**2, axis=0)
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                C = (1 - c1 - cmu_) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
                for k in range(mu):
                    C += cmu_ * weights[k] * np.outer(artmp[k], artmp[k])
                
                if counteval - eigeneval > lam / (c1 + cmu_) / n / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        B = np.eye(n)
                        D = np.ones(n)
                        C = np.eye(n)
                        invsqrtC = np.eye(n)
            
            # Step size update
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, max(ranges))
            
            # Check convergence
            if sigma < 1e-12:
                break
            if not use_sep:
                if np.max(D) > 1e7 * np.min(D):
                    break
    
    # --- SHADE ---
    def run_shade(pop_init, fit_init, budget_time):
        nonlocal best, best_pos
        t_start = elapsed()
        
        pop_size = len(pop_init)
        population = pop_init.copy()
        fitness = fit_init.copy()
        
        H = 50
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k_idx = 0
        archive = []
        archive_max = pop_size
        
        gen = 0
        stag = 0
        prev_best = best
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            gen += 1
            S_F, S_CR, delta_f = [], [], []
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            p_min = max(2, int(0.05 * pop_size))
            p_max = max(2, int(0.2 * pop_size))
            sorted_idx = np.argsort(fitness)
            
            for i in range(pop_size):
                if time_left() < 0.2:
                    return population, fitness
                
                ri = np.random.randint(H)
                Fi = -1
                while Fi <= 0:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                p_num = np.random.randint(p_min, p_max + 1)
                pbest_idx = sorted_idx[np.random.randint(min(p_num, pop_size))]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                pool_size = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
                
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, population[i])
                
                mask_lo = trial < lower
                mask_hi = trial > upper
                trial[mask_lo] = (lower[mask_lo] + population[i][mask_lo]) / 2
                trial[mask_hi] = (upper[mask_hi] + population[i][mask_hi]) / 2
                trial = clip(trial)
                
                trial_fit = eval_f(trial)
                
                if trial_fit <= fitness[i]:
                    if trial_fit < fitness[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - trial_fit))
                        if len(archive) < archive_max:
                            archive.append(population[i].copy())
                        elif archive_max > 0:
                            archive[np.random.randint(archive_max)] = population[i].copy()
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
            
            population = new_pop
            fitness = new_fit
            
            if len(S_F) > 0:
                w = np.array(delta_f)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k_idx] = np.sum(w * scr)
                k_idx = (k_idx + 1) % H
            
            if gen % 20 == 0:
                if abs(best - prev_best) < 1e-14:
                    stag += 1
                else:
                    stag = 0
                    prev_best = best
                if stag >= 3:
                    n_replace = pop_size // 2
                    worst = np.argsort(fitness)[-n_replace:]
                    for j in worst:
                        population[j] = lower + np.random.rand(dim) * ranges
                        fitness[j] = eval_f(population[j])
                    archive.clear()
                    stag = 0
                    prev_best = best
        
        return population, fitness
    
    # --- Nelder-Mead local search ---
    def nelder_mead(x0, budget_time, initial_scale=0.05):
        nonlocal best, best_pos
        t_start = elapsed()
        n = dim
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += initial_scale * ranges[i]
            simplex[i+1] = clip(simplex[i+1])
        
        f_vals = np.array([eval_f(simplex[i]) for i in range(n+1)])
        
        for _ in range(5000):
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < f_vals[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
            elif fr < f_vals[-2]:
                simplex[-1] = xr
                f_vals[-1] = fr
            else:
                if fr < f_vals[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = eval_f(xc)
                    if fc < f_vals[-1]:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
            
            spread = np.max(f_vals) - np.min(f_vals)
            if spread < 1e-15:
                break
    
    # --- Main orchestration ---
    pop_size = min(max(20, 8 * dim), 200)
    
    # Phase 1: SHADE exploration
    pop = lhs_sample(pop_size)
    fit = np.array([eval_f(p) for p in pop])
    
    shade_time = max_time * 0.40
    pop, fit = run_shade(pop, fit, shade_time)
    
    # Phase 2: CMA-ES from best found
    if time_left() > 1.0 and best_pos is not None:
        sigma0 = 0.2 * np.mean(ranges)
        run_cmaes(best_pos.copy(), sigma0, max_time * 0.30)
    
    # Phase 3: Nelder-Mead refinement
    if time_left() > 0.5 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.6, initial_scale=0.02)
    
    # Phase 4: Another CMA-ES with smaller sigma
    if time_left() > 1.0 and best_pos is not None:
        run_cmaes(best_pos.copy(), 0.05 * np.mean(ranges), time_left() * 0.5)
    
    # Phase 5: Final Nelder-Mead polish
    if time_left() > 0.5 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.8, initial_scale=0.005)
    
    return best
