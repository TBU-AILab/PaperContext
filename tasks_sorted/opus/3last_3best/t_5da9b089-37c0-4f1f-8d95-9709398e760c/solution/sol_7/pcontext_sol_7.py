#Looking at the results, algorithm 1 (pure CMA-ES, score 1.098) significantly outperformed algorithms 2 and 3 which used SHADE+local search combinations. I'll improve algorithm 1 by:
#
#1. **Better initial sampling** (proper LHS)
#2. **Multiple CMA-ES restarts** with increasing population (IPOP-CMA-ES strategy)
#3. **Better restart logic** - alternate between local restarts near best and global restarts
#4. **Faster eigendecomposition** - cache and update less frequently
#5. **Final local polish** with coordinate descent
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
            break
        fitness_vals[i] = evaluate(population[i])
    
    # Phase 2: IPOP-CMA-ES with restarts
    def run_cmaes(mean_init, sigma_init, pop_multiplier, time_limit):
        nonlocal best, best_params
        
        n = dim
        lam = int((4 + int(3 * np.log(n))) * pop_multiplier)
        lam = max(lam, 6)
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
        
        use_full = n <= 150
        
        if use_full:
            C = np.eye(n)
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            eigen_update_counter = 0
        else:
            diag_C = np.ones(n)
        
        generation = 0
        stagnation_count = 0
        local_best = float('inf')
        
        while elapsed() < time_limit:
            generation += 1
            
            if use_full:
                eigen_update_counter += 1
                if eigen_update_counter >= max(1, lam / (c1 + c_mu_val) / n / 10):
                    try:
                        C = (C + C.T) / 2.0
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-20)
                        eigen_update_counter = 0
                    except Exception:
                        C = np.eye(n)
                        eigvals = np.ones(n)
                        eigvecs = np.eye(n)
                
                D = np.sqrt(eigvals)
                inv_D = 1.0 / D
            
            offspring = np.zeros((lam, n))
            z_store = np.zeros((lam, n))
            offspring_fitness = np.full(lam, float('inf'))
            
            for i in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                z_store[i] = z
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
                inv_sqrt_C = eigvecs @ np.diag(inv_D) @ eigvecs.T
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ mean_diff)
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
            sigma = np.clip(sigma, 1e-20, np.mean(ranges) * 3)
            
            gen_best = offspring_fitness[sorted_idx[0]]
            if gen_best < local_best - 1e-12 * (1 + abs(local_best)):
                local_best = gen_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count > 15 + 3 * n // lam:
                return
            if sigma < 1e-16 * np.mean(ranges):
                return
    
    # Run IPOP-CMA-ES
    sorted_init = np.argsort(fitness_vals)
    elite = population[sorted_init[:max(1, min(10, n_init // 5))]]
    
    pop_mult = 1.0
    restart_count = 0
    
    while elapsed() < max_time * 0.92:
        restart_count += 1
        time_per_run = min((max_time * 0.92 - elapsed()), max(max_time * 0.05, (max_time * 0.92 - elapsed()) / 3))
        tl = elapsed() + time_per_run
        
        if restart_count % 3 == 1 and best_params is not None:
            m0 = best_params + np.random.randn(dim) * ranges * 0.01 * pop_mult
            s0 = np.mean(ranges) * 0.1 / pop_mult
            run_cmaes(clamp(m0), s0, pop_mult, tl)
        elif restart_count % 3 == 2 and best_params is not None:
            m0 = best_params + np.random.randn(dim) * ranges * 0.1
            s0 = np.mean(ranges) * 0.3
            run_cmaes(clamp(m0), s0, pop_mult, tl)
        else:
            m0 = lower + np.random.rand(dim) * ranges
            s0 = np.mean(ranges) * 0.3
            run_cmaes(m0, s0, pop_mult * 2, tl)
        
        if restart_count % 2 == 0:
            pop_mult = min(pop_mult * 1.5, 8.0)
    
    # Final coordinate descent polish
    if best_params is not None and elapsed() < max_time * 0.99:
        x_cur = best_params.copy()
        f_cur = best
        step = 0.001 * ranges
        for _ in range(300):
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
