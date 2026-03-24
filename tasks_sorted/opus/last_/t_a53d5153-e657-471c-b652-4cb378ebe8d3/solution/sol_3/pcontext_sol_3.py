#Looking at the previous algorithm, I can identify several issues:
#
#1. **L-SHADE implementation has bugs** (r2 selection logic is flawed)
#2. **CMA-ES implementation is too simplistic** - missing path evolution, proper step-size adaptation
#3. **Time allocation could be better** - need more exploitation time
#4. **No multi-restart strategy** for escaping local optima
#
#Let me implement a cleaner, more robust approach combining a well-tuned DE with proper CMA-ES restarts.
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
    
    evals = [0]
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling initialization ---
    pop_size = min(max(8 * dim, 50), 300)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.array([eval_f(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.95])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
    pop_size = len(fitness)
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # --- Phase 2: L-SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    init_pop_size = pop_size
    min_pop_size = 4

    generation = 0
    stagnation_count = 0
    prev_best = best

    while elapsed() < max_time * 0.55:
        generation += 1
        S_F, S_CR, S_delta = [], [], []
        
        trial_pop = np.empty_like(population)
        trial_fit = np.full(pop_size, float('inf'))
        
        # Fraction of time used in this phase
        frac = min(1.0, elapsed() / (max_time * 0.55))
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(0, H)
            
            # Generate F
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 1.0:
                    Fi = 1.0
            
            # Generate CR
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            # p-best selection
            p = max(2, int(np.ceil(max(0.05, 0.2 * (1 - frac)) * pop_size)))
            pbest_idx = np.random.randint(0, p)
            
            # r1 != i
            r1 = np.random.randint(0, pop_size - 1)
            if r1 >= i:
                r1 += 1
            
            # r2 from pop + archive, != i, != r1
            total = pop_size + len(archive)
            r2 = np.random.randint(0, total)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, total)
            
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            # current-to-pbest/1
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            
            # Binomial crossover
            jrand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < CRi
            mask[jrand] = True
            trial = np.where(mask, mutant, population[i])
            
            # Bounce-back boundary handling
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
                trial_pop[i] = trial
                trial_fit[i] = f_trial
            else:
                trial_pop[i] = population[i]
                trial_fit[i] = fitness[i]
        
        population = trial_pop[:pop_size]
        fitness = trial_fit[:pop_size]
        
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
        
        # Linear population size reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * frac)))
        if new_pop_size < pop_size:
            population = population[:new_pop_size]
            fitness = fitness[:new_pop_size]
            pop_size = new_pop_size
        
        # Stagnation check
        if abs(prev_best - best) < 1e-14:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if stagnation_count > 50 and pop_size > min_pop_size + 2:
            n_replace = max(1, pop_size // 3)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = best_params + 0.05 * ranges * np.random.randn(dim)
                population[j] = np.clip(population[j], lower, upper)
                fitness[j] = eval_f(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation_count = 0

    # --- Phase 3: CMA-ES with restarts ---
    def run_cmaes(init_mean, init_sigma, time_limit_frac):
        nonlocal best, best_params
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = init_mean.copy()
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Use diagonal covariance for high dim
        use_full = (n <= 50)
        
        if use_full:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            invsqrtC = np.eye(n)
        else:
            diag_C = np.ones(n)
        
        update_interval = max(1, int(1 / (c1 + cmu) / n / 10))
        gen_since_update = 0
        
        deadline = max_time * time_limit_frac
        
        while elapsed() < deadline:
            # Sample
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
            
            diff = (mean - old_mean) / sigma
            
            # Update evolution paths
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff / np.sqrt(diag_C + 1e-30)
            
            hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(evals[0]/lam+1))) < (1.4 + 2/(n+1)) * chi_n else 0
            
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * diff
            
            # Update covariance
            if use_full:
                artmp = (selected - old_mean).T / sigma  # n x mu
                C = (1 - c1 - cmu + (1-hs)*c1*cc*(2-cc)) * C + \
                    c1 * np.outer(pc, pc) + \
                    cmu * (artmp @ np.diag(w) @ artmp.T)
                
                # Enforce symmetry
                C = np.triu(C) + np.triu(C, 1).T
                
                gen_since_update += 1
                if gen_since_update >= update_interval:
                    gen_since_update = 0
                    try:
                        eigenvalues_sq, eigenvectors = np.linalg.eigh(C)
                        eigenvalues_sq = np.maximum(eigenvalues_sq, 1e-20)
                        eigenvalues = np.sqrt(eigenvalues_sq)
                        invsqrtC = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
                    except:
                        C = np.eye(n)
                        eigenvalues = np.ones(n)
                        eigenvectors = np.eye(n)
                        invsqrtC = np.eye(n)
                        sigma = init_sigma * 0.5
            else:
                artmp = (selected - old_mean) / sigma  # mu x n
                diag_C = (1 - c1 - cmu + (1-hs)*c1*cc*(2-cc)) * diag_C + \
                         c1 * pc**2 + \
                         cmu * np.sum(w[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-16 * np.mean(ranges))
            sigma = min(sigma, 2.0 * np.mean(ranges))
            
            # Check convergence
            if use_full:
                if sigma * np.max(eigenvalues) < 1e-12 * np.mean(ranges):
                    return
            else:
                if sigma * np.max(np.sqrt(diag_C)) < 1e-12 * np.mean(ranges):
                    return
    
    # Run CMA-ES from best found solution
    run_cmaes(best_params.copy(), 0.1 * np.mean(ranges), 0.80)
    
    # Second CMA-ES with smaller sigma
    if elapsed() < max_time * 0.90:
        run_cmaes(best_params.copy(), 0.01 * np.mean(ranges), 0.90)
    
    # Third restart from a perturbed position
    if elapsed() < max_time * 0.95:
        run_cmaes(best_params + 0.05 * ranges * np.random.randn(dim), 0.05 * np.mean(ranges), 0.95)
    
    # --- Phase 4: Final fine-tuning with coordinate-wise search ---
    scale = 0.001
    while elapsed() < max_time * 0.998:
        candidate = best_params + scale * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        f = eval_f(candidate)
        if f >= best:
            scale *= 0.95
            scale = max(scale, 1e-10)
    
    return best
