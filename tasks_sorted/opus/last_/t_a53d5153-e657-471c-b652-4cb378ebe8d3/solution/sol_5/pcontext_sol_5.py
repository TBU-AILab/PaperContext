#Looking at the previous algorithm, I see several areas for improvement:
#
#1. The L-SHADE phase takes too much time (50%) but may not be the most efficient
#2. CMA-ES restarts are poorly timed with fixed deadlines
#3. The algorithm lacks adaptive restart strategies
#4. Need better coordination between exploration and exploitation phases
#
#Let me implement a cleaner, more effective hybrid:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def eval_f(x):
        nonlocal best, best_params, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # === Phase 1: Quasi-random initialization with Sobol-like sampling ===
    n_init = min(max(15 * dim, 100), 500)
    
    # Latin Hypercube Sampling
    population = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        population[:, d] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    fitness = np.empty(n_init)
    for i in range(n_init):
        if elapsed() >= max_time * 0.08:
            population = population[:i]
            fitness = fitness[:i]
            break
        fitness[i] = eval_f(population[i])
    
    pop_size = len(fitness)
    if pop_size == 0:
        return best
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # === Phase 2: CMA-ES with IPOP restarts ===
    def run_cmaes(init_mean, init_sigma, deadline, lam_mult=1):
        nonlocal best, best_params
        n = dim
        lam = max(4 + int(3 * np.log(n)), 6) * lam_mult
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n+1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + mu_eff))
        damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = np.clip(init_mean.copy(), lower, upper)
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 80)
        
        if use_full:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            invsqrtC = np.eye(n)
            eigen_counter = 0
            eigen_interval = max(1, int(1 / (10*n*(c1 + cmu) + 1e-30)))
        else:
            diag_C = np.ones(n)
        
        gen = 0
        best_gen_f = float('inf')
        stale = 0
        f_history = []
        
        while elapsed() < deadline:
            gen += 1
            offspring = np.empty((lam, n))
            f_off = np.empty(lam)
            
            for i in range(lam):
                if elapsed() >= deadline:
                    return best
                z = np.random.randn(n)
                if use_full:
                    offspring[i] = mean + sigma * (eigenvectors @ (eigenvalues * z))
                else:
                    offspring[i] = mean + sigma * np.sqrt(np.maximum(diag_C, 1e-20)) * z
                offspring[i] = np.clip(offspring[i], lower, upper)
                f_off[i] = eval_f(offspring[i])
            
            order = np.argsort(f_off)
            selected = offspring[order[:mu]]
            
            old_mean = mean.copy()
            mean = np.dot(weights, selected)
            mean = np.clip(mean, lower, upper)
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ diff)
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * diff / (np.sqrt(np.maximum(diag_C, 1e-20)))
            
            ps_norm = np.linalg.norm(ps)
            gen_thresh = 1 - (1-cs)**(2*(gen))
            hs = 1 if ps_norm / np.sqrt(max(gen_thresh, 1e-30)) < (1.4 + 2/(n+1)) * chi_n else 0
            
            pc = (1-cc)*pc + hs * np.sqrt(cc*(2-cc)*mu_eff) * diff
            
            if use_full:
                artmp = ((selected - old_mean) / max(sigma, 1e-30)).T
                rank_one = np.outer(pc, pc)
                rank_mu = artmp @ np.diag(weights) @ artmp.T
                C = (1 - c1 - cmu + (1-hs)*c1*cc*(2-cc)) * C + c1 * rank_one + cmu * rank_mu
                C = np.triu(C) + np.triu(C, 1).T
                
                eigen_counter += 1
                if eigen_counter >= eigen_interval:
                    eigen_counter = 0
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
            else:
                artmp = (selected - old_mean) / max(sigma, 1e-30)
                diag_C = (1 - c1 - cmu + (1-hs)*c1*cc*(2-cc)) * diag_C + \
                         c1 * pc**2 + cmu * np.sum(weights[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp(np.clip((cs/damps) * (ps_norm/chi_n - 1), -0.6, 0.6))
            sigma = np.clip(sigma, 1e-17 * np.mean(ranges), 2.0 * np.mean(ranges))
            
            cur_best = f_off[order[0]]
            f_history.append(cur_best)
            
            if cur_best < best_gen_f - 1e-15:
                best_gen_f = cur_best
                stale = 0
            else:
                stale += 1
            
            # Check convergence
            max_std = sigma * (np.max(eigenvalues) if use_full else np.max(np.sqrt(diag_C)))
            if max_std < 1e-14 * np.mean(ranges):
                return best
            
            if stale > 10 + 30 * n // lam:
                return best
            
            if len(f_history) > 20:
                recent = f_history[-20:]
                if max(recent) - min(recent) < 1e-14 * (abs(best) + 1e-30):
                    return best
        
        return best

    # Main CMA-ES loop with IPOP restarts
    lam_multiplier = 1
    n_restarts = 0
    
    while elapsed() < max_time * 0.85:
        remaining = max_time - elapsed()
        if remaining < 0.5:
            break
        
        # Choose start point
        if n_restarts == 0:
            start_point = best_params.copy()
            sig = 0.2 * np.mean(ranges)
        elif n_restarts == 1:
            start_point = best_params.copy()
            sig = 0.05 * np.mean(ranges)
        elif n_restarts % 3 == 0:
            # Random restart for exploration
            start_point = lower + np.random.rand(dim) * ranges
            sig = 0.3 * np.mean(ranges)
            lam_multiplier = min(lam_multiplier * 2, 8)
        else:
            # Perturbed best
            start_point = best_params + 0.1 * ranges * np.random.randn(dim)
            start_point = np.clip(start_point, lower, upper)
            sig = 0.1 * np.mean(ranges) / (1 + n_restarts * 0.2)
        
        budget_frac = min(0.85, elapsed() / max_time + max(0.08, 0.25 / (1 + n_restarts)))
        deadline = max_time * budget_frac
        
        run_cmaes(start_point, sig, deadline, lam_multiplier if n_restarts % 3 == 0 else 1)
        n_restarts += 1

    # === Phase 3: Nelder-Mead refinement ===
    if elapsed() < max_time * 0.97 and best_params is not None:
        n = dim
        scale = 0.0005 * ranges
        simplex = np.empty((n+1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i]
            simplex[i+1] = np.clip(simplex[i+1], lower, upper)
        
        f_simplex = np.empty(n+1)
        for i in range(n+1):
            if elapsed() >= max_time * 0.97:
                break
            f_simplex[i] = eval_f(simplex[i])
        else:
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            nm_iters = 0
            while elapsed() < max_time * 0.995:
                nm_iters += 1
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
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
                        fc = eval_f(xc)
                        if fc <= fr:
                            simplex[-1], f_simplex[-1] = xc, fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                                if elapsed() >= max_time * 0.995:
                                    break
                    else:
                        xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                        fc = eval_f(xc)
                        if fc < f_simplex[-1]:
                            simplex[-1], f_simplex[-1] = xc, fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                                if elapsed() >= max_time * 0.995:
                                    break

    # === Phase 4: Final micro-perturbations ===
    sc = 1e-5
    while elapsed() < max_time * 0.999:
        candidate = best_params + sc * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best
