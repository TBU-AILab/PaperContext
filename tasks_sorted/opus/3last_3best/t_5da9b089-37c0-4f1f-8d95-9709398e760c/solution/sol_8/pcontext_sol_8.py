#Looking at the results, algorithm 2 (pure CMA-ES, score 1.098) was the best. I'll improve it by:
#
#1. **IPOP-CMA-ES** with proper restarts (increasing population)
#2. **Better eigendecomposition frequency** to reduce overhead
#3. **Multiple restart strategies** - local refinement + global exploration
#4. **Nelder-Mead polish** at the end
#5. **Smarter stagnation detection and restart logic**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clamp(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clamp(x)
        val = func(x)
        if val < best:
            best = val
            best_params = x.copy()
        return val
    
    # Phase 1: LHS initial sampling
    n_init = min(50 * dim, max(200, 20 * dim))
    population = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        population[:, d] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    fitness_vals = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.90:
            return best
        fitness_vals[i] = evaluate(population[i])
    
    # Sort initial population
    sorted_init = np.argsort(fitness_vals)
    
    # CMA-ES function
    def run_cmaes(mean_init, sigma_init, pop_multiplier, time_limit):
        nonlocal best, best_params
        
        n = dim
        lam = max(6, int((4 + int(3 * np.log(n))) * pop_multiplier))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        
        mean = mean_init.copy()
        sigma = sigma_init
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        use_full = n <= 200
        
        if use_full:
            C = np.eye(n)
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            D = np.ones(n)
            inv_D = np.ones(n)
            needs_eigen = False
            eigen_counter = 0
            eigen_interval = max(1, int(lam / (c1 + c_mu_val) / n / 10))
        else:
            diag_C = np.ones(n)
        
        generation = 0
        stagnation_count = 0
        local_best = float('inf')
        f_history = []
        
        while elapsed() < time_limit:
            generation += 1
            
            if use_full:
                eigen_counter += 1
                if eigen_counter >= eigen_interval or generation == 1:
                    try:
                        C = (C + C.T) / 2.0
                        eigvals_raw, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals_raw, 1e-20)
                        D = np.sqrt(eigvals)
                        inv_D = 1.0 / D
                        eigen_counter = 0
                    except Exception:
                        C = np.eye(n)
                        eigvals = np.ones(n)
                        eigvecs = np.eye(n)
                        D = np.ones(n)
                        inv_D = np.ones(n)
            
            offspring = np.zeros((lam, n))
            offspring_fitness = np.full(lam, float('inf'))
            
            for i in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                if use_full:
                    y = eigvecs @ (D * z)
                else:
                    y = np.sqrt(diag_C) * z
                
                x = mean + sigma * y
                x = clamp(x)
                offspring[i] = x
                offspring_fitness[i] = evaluate(x)
            
            sorted_idx = np.argsort(offspring_fitness)
            
            old_mean = mean.copy()
            selected = offspring[sorted_idx[:mu]]
            mean = np.sum(weights[:, None] * selected, axis=0)
            
            mean_diff = (mean - old_mean) / sigma
            
            if use_full:
                inv_sqrt_C_times_md = eigvecs @ (inv_D * (eigvecs.T @ mean_diff))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C_times_md
            else:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean_diff / np.sqrt(diag_C))
            
            ps_norm = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * generation)) < (1.4 + 2.0 / (n + 1)) * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff
            
            if use_full:
                artmp = (selected - old_mean) / sigma
                rank_mu_update = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_update += weights[i] * np.outer(artmp[i], artmp[i])
                C = (1 - c1 - c_mu_val) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu_val * rank_mu_update
            else:
                artmp = (selected - old_mean) / sigma
                diag_C = (1 - c1 - c_mu_val) * diag_C + c1 * (p_c ** 2 + (1 - h_sigma) * c_c * (2 - c_c) * diag_C)
                for i in range(mu):
                    diag_C += c_mu_val * weights[i] * artmp[i] ** 2
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges) * 3)
            
            gen_best = offspring_fitness[sorted_idx[0]]
            f_history.append(gen_best)
            
            if gen_best < local_best - 1e-12 * (1 + abs(local_best)):
                local_best = gen_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count > 10 + 30 * n // lam:
                return
            if sigma < 1e-16 * np.mean(ranges):
                return
            if use_full and np.max(D) / (np.min(D) + 1e-30) > 1e14:
                return
    
    # Phase 2: IPOP-CMA-ES with restarts
    pop_mult = 1.0
    restart_count = 0
    
    # First run from best initial point
    if best_params is not None:
        tl = min(elapsed() + max(max_time * 0.08, (max_time * 0.92 - elapsed()) / 4), max_time * 0.92)
        run_cmaes(best_params.copy(), np.mean(ranges) * 0.2, 1.0, tl)
    
    while elapsed() < max_time * 0.92:
        restart_count += 1
        remaining = max_time * 0.92 - elapsed()
        time_alloc = min(remaining, max(max_time * 0.04, remaining / max(1, 4 - restart_count)))
        tl = elapsed() + time_alloc
        
        if restart_count % 4 <= 1 and best_params is not None:
            perturbation = np.random.randn(dim) * ranges * (0.005 + 0.02 * restart_count)
            m0 = clamp(best_params + perturbation)
            s0 = np.mean(ranges) * (0.05 + 0.05 * min(restart_count, 5))
            run_cmaes(m0, s0, pop_mult, tl)
        else:
            m0 = lower + np.random.rand(dim) * ranges
            s0 = np.mean(ranges) * 0.3
            pop_mult = min(pop_mult * 1.5, 6.0)
            run_cmaes(m0, s0, pop_mult, tl)
    
    # Phase 3: Coordinate descent polish
    if best_params is not None and elapsed() < max_time * 0.99:
        x_cur = best_params.copy()
        f_cur = best
        step = 0.001 * ranges
        for _ in range(500):
            if elapsed() >= max_time * 0.99:
                break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.99:
                    break
                for sign in [1, -1]:
                    xt = x_cur.copy()
                    xt[d] += sign * step[d]
                    xt = clamp(xt)
                    ft = evaluate(xt)
                    if ft < f_cur:
                        x_cur, f_cur = xt, ft
                        step[d] *= 1.5
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-15:
                    break
    
    return best
