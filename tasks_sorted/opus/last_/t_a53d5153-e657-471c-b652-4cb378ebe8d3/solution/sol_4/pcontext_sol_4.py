#Looking at the previous algorithm, I can identify several issues:
#
#1. The L-SHADE has bugs in index handling (r1 can go out of bounds)
#2. CMA-ES restarts aren't well-timed and the implementation has issues
#3. Time allocation between phases isn't optimal
#4. Need better local search at the end
#
#Let me implement a cleaner, more robust hybrid approach:
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
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Initialize with LHS ---
    pop_size = min(max(10 * dim, 60), 400)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.empty(pop_size)
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            fitness = fitness[:i]
            population = population[:i]
            break
        fitness[i] = eval_f(population[i])
    
    pop_size = len(fitness)
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # --- Phase 2: L-SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.85)
    k = 0
    archive = []
    init_pop_size = pop_size
    min_pop_size = max(4, dim // 2)
    
    stag_count = 0
    prev_best = best

    while elapsed() < max_time * 0.50 and pop_size >= min_pop_size:
        frac = min(1.0, elapsed() / (max_time * 0.50))
        S_F, S_CR, S_delta = [], [], []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.50:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 20:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            
            # Generate CR
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # p-best
            p_min = max(2, int(np.ceil(0.05 * pop_size)))
            p_max = max(p_min, int(np.ceil(0.25 * pop_size)))
            p = max(p_min, int(round(p_max - (p_max - p_min) * frac)))
            pbest_idx = np.random.randint(0, p)
            
            # Select r1 != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(0, len(candidates))]
            candidates.remove(r1)
            
            # Select r2 from pop + archive, != i, != r1
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined = [c for c in combined if c != i and c != r1]
            if len(combined) == 0:
                combined = [c for c in range(pop_size) if c != i]
            r2_idx = combined[np.random.randint(0, len(combined))]
            xr2 = population[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            # Mutation: current-to-pbest/1
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            jrand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < CRi
            mask[jrand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Bounce-back
            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2.0
            trial[above] = (upper[above] + population[i][above]) / 2.0
            trial = np.clip(trial, lower, upper)
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    archive.append(population[i].copy())
                    delta = fitness[i] - f_trial
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_delta.append(delta)
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Trim archive
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        # Update memory
        if len(S_F) > 0:
            w = np.array(S_delta)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        # Population size reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * frac)))
        if new_pop_size < pop_size:
            population = population[:new_pop_size]
            fitness = fitness[:new_pop_size]
            pop_size = new_pop_size
        
        # Stagnation handling
        if abs(prev_best - best) < 1e-15:
            stag_count += 1
        else:
            stag_count = 0
        prev_best = best
        
        if stag_count > 30:
            n_replace = max(1, pop_size // 4)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = best_params + 0.1 * ranges * np.random.randn(dim)
                population[j] = np.clip(population[j], lower, upper)
                fitness[j] = eval_f(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stag_count = 0

    # --- Phase 3: CMA-ES restarts ---
    def run_cmaes(init_mean, init_sigma, deadline_frac):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n+1.3)**2 + mu_eff)
        cmu_val = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + mu_eff))
        damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = init_mean.copy()
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 60)
        
        if use_full:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0
        else:
            diag_C = np.ones(n)
        
        gen = 0
        deadline = max_time * deadline_frac
        
        while elapsed() < deadline:
            gen += 1
            offspring = np.empty((lam, n))
            f_off = np.empty(lam)
            
            for i in range(lam):
                if elapsed() >= deadline:
                    return
                z = np.random.randn(n)
                if use_full:
                    offspring[i] = mean + sigma * (eigenvectors @ (eigenvalues * z))
                else:
                    offspring[i] = mean + sigma * np.sqrt(diag_C) * z
                offspring[i] = np.clip(offspring[i], lower, upper)
                f_off[i] = eval_f(offspring[i])
            
            order = np.argsort(f_off)
            selected = offspring[order[:mu]]
            
            old_mean = mean.copy()
            mean = selected.T @ w
            mean = np.clip(mean, lower, upper)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ diff)
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * diff / (np.sqrt(diag_C) + 1e-30)
            
            ps_norm = np.linalg.norm(ps)
            gen_factor = 1 - (1-cs)**(2*gen)
            hs = 1 if ps_norm / np.sqrt(max(gen_factor, 1e-30)) < (1.4 + 2/(n+1)) * chi_n else 0
            
            pc = (1-cc)*pc + hs * np.sqrt(cc*(2-cc)*mu_eff) * diff
            
            if use_full:
                artmp = ((selected - old_mean) / (sigma + 1e-30)).T  # n x mu
                C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * C + \
                    c1 * np.outer(pc, pc) + \
                    cmu_val * (artmp @ np.diag(w) @ artmp.T)
                C = (C + C.T) / 2
                
                eigen_countdown -= 1
                if eigen_countdown <= 0:
                    eigen_countdown = max(1, int(1 / (10*n*(c1 + cmu_val))))
                    try:
                        eig_vals, eigenvectors = np.linalg.eigh(C)
                        eig_vals = np.maximum(eig_vals, 1e-20)
                        eigenvalues = np.sqrt(eig_vals)
                        invsqrtC = eigenvectors @ np.diag(1.0/eigenvalues) @ eigenvectors.T
                    except:
                        C = np.eye(n)
                        eigenvalues = np.ones(n)
                        eigenvectors = np.eye(n)
                        invsqrtC = np.eye(n)
                        sigma *= 0.5
            else:
                artmp = (selected - old_mean) / (sigma + 1e-30)  # mu x n
                diag_C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * diag_C + \
                         c1 * pc**2 + \
                         cmu_val * np.sum(w[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp(min(0.6, (cs/damps) * (ps_norm/chi_n - 1)))
            sigma = np.clip(sigma, 1e-16 * np.mean(ranges), 2.0 * np.mean(ranges))
            
            # Convergence check
            max_std = sigma * (np.max(eigenvalues) if use_full else np.max(np.sqrt(diag_C)))
            if max_std < 1e-13 * np.mean(ranges):
                return

    # Multiple CMA-ES restarts with different settings
    restart_configs = [
        (best_params.copy(), 0.15 * np.mean(ranges), 0.72),
        (best_params.copy(), 0.02 * np.mean(ranges), 0.85),
        (best_params + 0.1 * ranges * np.random.randn(dim), 0.08 * np.mean(ranges), 0.92),
        (best_params.copy(), 0.005 * np.mean(ranges), 0.96),
    ]
    
    for init_m, init_s, deadline_f in restart_configs:
        if elapsed() < max_time * (deadline_f - 0.05):
            init_m = np.clip(init_m, lower, upper)
            run_cmaes(init_m, init_s, deadline_f)

    # --- Phase 4: Nelder-Mead style local search ---
    if elapsed() < max_time * 0.97 and best_params is not None:
        n = dim
        scale = 0.001 * ranges
        simplex = np.empty((n+1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i]
            simplex[i+1] = np.clip(simplex[i+1], lower, upper)
        
        f_simplex = np.array([eval_f(simplex[i]) for i in range(n+1) if elapsed() < max_time * 0.97])
        if len(f_simplex) < n+1:
            pass
        else:
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            while elapsed() < max_time * 0.998:
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
                fr = eval_f(xr)
                
                if fr < f_simplex[0]:
                    xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1], f_simplex[-1] = xe, fe
                    else:
                        simplex[-1], f_simplex[-1] = xr, fr
                elif fr < f_simplex[-2]:
                    simplex[-1], f_simplex[-1] = xr, fr
                else:
                    if fr < f_simplex[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    else:
                        xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = eval_f(xc)
                    if fc < min(fr, f_simplex[-1]):
                        simplex[-1], f_simplex[-1] = xc, fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])
                            if elapsed() >= max_time * 0.998:
                                break

    # --- Phase 5: Final random perturbation ---
    scale = 0.0001
    while elapsed() < max_time * 0.999:
        candidate = best_params + scale * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best
