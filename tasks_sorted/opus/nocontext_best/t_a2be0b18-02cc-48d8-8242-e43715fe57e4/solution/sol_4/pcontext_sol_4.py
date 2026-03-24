#
#
#Looking at the current algorithm, I can see several areas for improvement:
#
#1. **Better initial sampling** with more diverse strategies
#2. **CMA-ES** instead of or alongside DE for better local convergence
#3. **Multi-restart strategy** to escape local optima
#4. **Better time budget allocation**
#5. **Improved local search with multiple restarts**
#
#Let me implement a hybrid approach combining SHADE-style DE with CMA-ES-like adaptation and more aggressive local search:
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
    center = (lower + upper) / 2.0
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time * 0.98 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def bounce_back(mutant, parent):
        x = mutant.copy()
        for d in range(dim):
            if x[d] < lower[d]:
                x[d] = lower[d] + np.random.random() * (parent[d] - lower[d])
            elif x[d] > upper[d]:
                x[d] = upper[d] - np.random.random() * (upper[d] - parent[d])
        return x

    # --- Nelder-Mead local search ---
    def nelder_mead(x0, max_nm_time, initial_step_scale=0.05):
        nonlocal best, best_params
        nm_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        step = ranges * initial_step_scale
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += step[i] if x0[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([evaluate(clip(simplex[i])) for i in range(n + 1)])
        
        no_improve = 0
        while (elapsed() - nm_start) < max_nm_time and remaining() > 0:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            old_best = f_simplex[0]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            if remaining() <= 0: break
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if remaining() <= 0: break
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = evaluate(xc)
                    if remaining() <= 0: break
                    if fc < fr:
                        simplex[-1], f_simplex[-1] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if remaining() <= 0: break
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = evaluate(xc)
                    if remaining() <= 0: break
                    if fc < f_simplex[-1]:
                        simplex[-1], f_simplex[-1] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if remaining() <= 0: break
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
            
            if f_simplex[np.argsort(f_simplex)[0]] >= old_best - 1e-14:
                no_improve += 1
            else:
                no_improve = 0
            
            if no_improve > 50 * dim:
                break
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]))
            if spread < 1e-13:
                break
    
    # --- CMA-ES inspired search ---
    def cmaes_search(x0, sigma0, max_cma_time):
        nonlocal best, best_params
        cma_start = elapsed()
        n = dim
        
        mean = x0.copy()
        sigma = sigma0
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
        C = np.eye(n)
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen = 0
        while (elapsed() - cma_start) < max_cma_time and remaining() > 0:
            gen += 1
            
            # Update eigen decomposition periodically
            if gen % max(1, int(lam / (10 * n))) == 0 or gen == 1:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            
            # Sample offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
            
            # Evaluate
            fit = np.zeros(lam)
            for k in range(lam):
                if remaining() <= 0:
                    return
                fit[k] = evaluate(arx[k])
            
            # Sort
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen)) / chiN) < (1.4 + 2/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (weights[:, None] * artmp).T @ artmp
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(ranges))
            
            # Termination conditions
            if sigma < 1e-14:
                break
            if np.max(D) > 1e7 * np.min(D):
                break
    
    # --- Phase 1: Initial sampling with LHS ---
    pop_size = min(300, max(40, 20 * dim))
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if remaining() <= 0:
            return best
        fitness[i] = evaluate(population[i])
    
    idx = np.argsort(fitness)
    population = population[idx]
    fitness = fitness[idx]
    
    # --- Phase 2: SHADE DE ---
    H = 100
    MF = np.full(H, 0.5)
    MCR = np.full(H, 0.5)
    hist_pos = 0
    
    # Archive for DE
    archive = []
    archive_max = pop_size
    
    stagnation_counter = 0
    prev_best = best
    generation = 0
    
    de_budget = max_time * 0.50
    
    while remaining() > max_time * 0.45:
        generation += 1
        
        SF = []
        SCR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        ri = np.random.randint(0, H, pop_size)
        
        F_vals = np.zeros(pop_size)
        CR_vals = np.zeros(pop_size)
        for i in range(pop_size):
            while True:
                F_vals[i] = np.random.standard_cauchy() * 0.1 + MF[ri[i]]
                if F_vals[i] > 0:
                    break
            F_vals[i] = min(F_vals[i], 1.0)
            CR_vals[i] = np.clip(np.random.normal(MCR[ri[i]], 0.1), 0.0, 1.0)
        
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.20 * pop_size))
        
        for i in range(pop_size):
            if remaining() <= max_time * 0.45:
                break
            
            Fi = F_vals[i]
            CRi = CR_vals[i]
            
            # current-to-pbest/1 with archive
            p = np.random.randint(p_min, p_max + 1)
            pbest_idx = np.random.randint(0, p)
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            # r2 from pop + archive
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            
            if r2 < pop_size:
                xr2 = population[r2]
            else:
                xr2 = archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            mutant = bounce_back(mutant, population[i])
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            trial = clip(trial)
            f_trial = evaluate(trial)
            
            if f_trial <= new_fit[i]:
                if f_trial < new_fit[i]:
                    SF.append(Fi)
                    SCR.append(CRi)
                    S_delta.append(abs(new_fit[i] - f_trial))
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        # Update SHADE history
        if SF and sum(S_delta) > 0:
            weights = np.array(S_delta)
            weights = weights / np.sum(weights)
            sf_arr = np.array(SF)
            scr_arr = np.array(SCR)
            MF[hist_pos] = np.sum(weights * sf_arr**2) / (np.sum(weights * sf_arr) + 1e-30)
            MCR[hist_pos] = np.sum(weights * scr_arr)
            hist_pos = (hist_pos + 1) % H
        
        population = new_pop
        fitness = new_fit
        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]
        
        if abs(best - prev_best) < 1e-14:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            prev_best = best
        
        # Restart worst half if stagnant
        if stagnation_counter > 20 + dim:
            n_replace = pop_size // 2
            for k in range(n_replace):
                if remaining() <= max_time * 0.45:
                    break
                idx_r = pop_size - 1 - k
                # Generate near best or random
                if np.random.random() < 0.5:
                    new_ind = best_params + np.random.randn(dim) * ranges * 0.1
                else:
                    new_ind = lower + np.random.random(dim) * ranges
                new_ind = clip(new_ind)
                population[idx_r] = new_ind
                fitness[idx_r] = evaluate(new_ind)
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
            stagnation_counter = 0
            archive.clear()
    
    # --- Phase 3: CMA-ES from best found ---
    if best_params is not None and remaining() > max_time * 0.15:
        cma_time = remaining() - max_time * 0.15
        sigma0 = np.max(ranges) * 0.1
        cmaes_search(best_params.copy(), sigma0, cma_time)
    
    # --- Phase 4: Multi-start CMA-ES from diverse points ---
    if remaining() > max_time * 0.08:
        # Try a couple more CMA-ES from different starting points
        top_k = min(5, pop_size)
        for ki in range(top_k):
            if remaining() <= max_time * 0.03:
                break
            cma_time_each = remaining() / (top_k - ki + 1)
            start_x = population[ki].copy() if ki > 0 else best_params.copy()
            sigma0 = np.max(ranges) * (0.01 if ki == 0 else 0.05)
            cmaes_search(start_x, sigma0, cma_time_each)
    
    # --- Phase 5: Final Nelder-Mead polish ---
    if best_params is not None and remaining() > 0:
        nelder_mead(best_params.copy(), remaining() * 0.8, initial_step_scale=0.01)
    
    if best_params is not None and remaining() > 0:
        nelder_mead(best_params.copy(), remaining(), initial_step_scale=0.001)
    
    return best
#
#Key improvements over the previous version:
#
#1. **CMA-ES integration**: Added a full CMA-ES implementation that adapts the covariance matrix, giving much better convergence on many function landscapes
#2. **SHADE archive**: Added an external archive for the DE mutation strategy, improving diversity
#3. **Multi-phase approach**: LHS → SHADE-DE → CMA-ES → Multi-start CMA-ES → Nelder-Mead polish
#4. **Better time budgeting**: More time allocated to CMA-ES which is generally very effective
#5. **Improved restart strategy**: Mix of local perturbation and random restart on stagnation
#6. **Multiple Nelder-Mead passes** with decreasing step sizes for fine-tuning
#7. **Proper Cauchy sampling** for F values (rejecting negative values instead of clipping)
