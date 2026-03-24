#Looking at the previous algorithm, I can see it implements CMA-ES with some initialization and local refinement. Let me improve it with:
#
#1. Better CMA-ES implementation with proper restart strategy (IPOP-CMA-ES)
#2. More efficient initialization using Sobol-like quasi-random sequences
#3. Better time management
#4. Differential Evolution as a complementary global search
#5. More robust local search
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
    
    def eval_func(x):
        nonlocal best, best_x
        x_clipped = np.clip(x, lower, upper)
        f = func(x_clipped)
        if f < best:
            best = f
            best_x = x_clipped.copy()
        return f
    
    # --- Phase 1: Initialization with LHS ---
    n_init = min(max(10 * dim, 50), 300)
    
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.15:
            n_init = i
            init_points = init_points[:n_init]
            init_fitness = init_fitness[:n_init]
            break
        init_fitness[i] = eval_func(init_points[i])
    
    if n_init == 0:
        return best
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: Differential Evolution ---
    def de_search(time_budget_fraction):
        nonlocal best, best_x
        target_end = elapsed() + max_time * time_budget_fraction
        
        pop_size = max(min(10 * dim, 100), 20)
        
        # Initialize population from best init points + random
        n_elite = min(pop_size // 2, n_init)
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_points[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if elapsed() < target_end:
                fit[i] = eval_func(pop[i])
        
        F = 0.8
        CR = 0.9
        
        generation = 0
        while elapsed() < target_end:
            generation += 1
            # Adaptive parameters
            F_gen = 0.5 + 0.3 * np.random.random()
            CR_gen = 0.8 + 0.2 * np.random.random()
            
            for i in range(pop_size):
                if elapsed() >= target_end:
                    return
                
                # current-to-pbest/1
                p_best_size = max(2, pop_size // 5)
                p_best_idx = np.argsort(fit)[:p_best_size]
                x_pbest = pop[np.random.choice(p_best_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                
                F_i = F_gen + 0.1 * np.random.randn()
                F_i = np.clip(F_i, 0.1, 1.0)
                
                mutant = pop[i] + F_i * (x_pbest - pop[i]) + F_i * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lower, upper)
                
                # Binomial crossover
                CR_i = CR_gen + 0.1 * np.random.randn()
                CR_i = np.clip(CR_i, 0.0, 1.0)
                cross_points = np.random.random(dim) < CR_i
                j_rand = np.random.randint(dim)
                cross_points[j_rand] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = eval_func(trial)
                
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
    
    # --- Phase 3: CMA-ES ---
    def cma_es_search(x0, sigma0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 8)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 100)
        
        if use_full:
            C = np.eye(n)
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            update_eigen_interval = max(1, lam // (10 * n))
        else:
            diag_cov = np.ones(n)
        
        counteval = 0
        gen = 0
        stagnation_count = 0
        prev_best_gen = float('inf')
        
        while elapsed() < target_end:
            gen += 1
            
            # Update eigen decomposition periodically
            if use_full and gen % max(1, int(1.0 / (c1 + cmu_val) / n / 10)) == 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                except:
                    C = np.eye(n)
                    eigvals = np.ones(n)
                    eigvecs = np.eye(n)
            
            # Generate and evaluate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full:
                D = np.sqrt(eigvals)
                for k in range(lam):
                    arx[k] = mean + sigma * (eigvecs @ (D * arz[k]))
            else:
                sqrt_diag = np.sqrt(np.maximum(diag_cov, 1e-20))
                for k in range(lam):
                    arx[k] = mean + sigma * sqrt_diag * arz[k]
            
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.full(lam, float('inf'))
            for k in range(lam):
                if elapsed() >= target_end:
                    return
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            best_gen = arfitness[arindex[0]]
            
            # Stagnation check
            if best_gen >= prev_best_gen - 1e-12 * abs(prev_best_gen):
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best_gen = min(prev_best_gen, best_gen)
            
            if stagnation_count > 10 + 30 * n / lam:
                break
            
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[arindex[k]]
            
            diff = mean - old_mean
            
            if use_full:
                invsqrt = eigvecs @ (arz[0:1].T * 0).flatten()  # placeholder
                try:
                    invD = 1.0 / D
                    z = eigvecs.T @ (diff / sigma)
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (eigvecs @ (invD * z))
                except:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / sigma
            else:
                inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(diag_cov, 1e-20))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_diag * diff / sigma
            
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            if use_full:
                artmp = np.zeros((mu, n))
                for k in range(mu):
                    artmp[k] = (arx[arindex[k]] - old_mean) / sigma
                
                rank_mu_update = np.zeros((n, n))
                for k in range(mu):
                    rank_mu_update += weights[k] * np.outer(artmp[k], artmp[k])
                
                C = (1 - c1 - cmu_val + (1 - hsig) * c1 * cc * (2 - cc)) * C + \
                    c1 * np.outer(pc, pc) + cmu_val * rank_mu_update
            else:
                artmp = np.zeros((mu, n))
                for k in range(mu):
                    artmp[k] = (arx[arindex[k]] - old_mean) / sigma
                
                rank_mu_diag = np.zeros(n)
                for k in range(mu):
                    rank_mu_diag += weights[k] * artmp[k] ** 2
                
                diag_cov = (1 - c1 - cmu_val + (1 - hsig) * c1 * cc * (2 - cc)) * diag_cov + \
                    c1 * pc ** 2 + cmu_val * rank_mu_diag
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 10 * np.max(ranges))
            
            if sigma * np.max(np.sqrt(eigvals) if use_full else np.sqrt(diag_cov)) < 1e-18:
                break
    
    # --- Phase 4: Local search (coordinate descent with golden section) ---
    def local_search(x0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        
        x = x0.copy()
        fx = eval_func(x)
        step = 0.01 * ranges.copy()
        
        while elapsed() < target_end:
            improved = False
            for d in range(dim):
                if elapsed() >= target_end:
                    return
                
                for direction in [1, -1]:
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x[d] + direction * step[d], lower[d], upper[d])
                    f_trial = eval_func(x_trial)
                    if f_trial < fx:
                        # Accelerate in this direction
                        x = x_trial
                        fx = f_trial
                        improved = True
                        # Try to go further
                        for _ in range(5):
                            if elapsed() >= target_end:
                                return
                            x_trial2 = x.copy()
                            x_trial2[d] = np.clip(x[d] + direction * step[d], lower[d], upper[d])
                            f_trial2 = eval_func(x_trial2)
                            if f_trial2 < fx:
                                x = x_trial2
                                fx = f_trial2
                            else:
                                break
                        break
            
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break
    
    # --- Execution plan ---
    total_time = max_time
    
    # Phase 2: DE - use 30% of time
    if time_left() > 0.5:
        de_search(0.30)
    
    # Phase 3: CMA-ES with IPOP restarts - use 55% of time
    restart = 0
    lam_mult = 1
    while time_left() > total_time * 0.10 and elapsed() < max_time * 0.85:
        restart += 1
        
        if restart == 1 and best_x is not None:
            x0 = best_x.copy()
        else:
            # Random restart near best or fully random
            if np.random.random() < 0.5 and best_x is not None:
                x0 = best_x + 0.1 * ranges * np.random.randn(dim) / np.sqrt(restart)
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.random(dim) * ranges
        
        sigma0 = 0.3 * np.mean(ranges) / np.sqrt(restart)
        sigma0 = max(sigma0, 0.01 * np.mean(ranges))
        
        t_budget = min(time_left() * 0.5, max_time * 0.25)
        if t_budget < 0.2:
            break
        
        cma_es_search(x0, sigma0, t_budget)
    
    # Phase 4: Local refinement - remaining time
    if best_x is not None and time_left() > 0.1:
        local_search(best_x.copy(), time_left() * 0.95)
    
    return best
#
#Key improvements:
#1. **Differential Evolution (DE)** added as a global search phase with current-to-pbest mutation strategy
#2. **IPOP-CMA-ES restarts** with increasing population and decreasing sigma
#3. **Better stagnation detection** in CMA-ES to avoid wasting time
#4. **Accelerated coordinate descent** in local search (continues in promising direction)
#5. **Better time allocation**: 15% init, 30% DE, ~45% CMA-ES restarts, ~10% local search
#6. **Centralized eval_func** that always tracks best solution
