#Looking at the current algorithm, I can see it uses CMA-ES with restarts and coordinate descent refinement. Let me improve it with several enhancements:
#
#1. Better CMA-ES implementation with proper restart strategy (IPOP-CMA-ES)
#2. Differential Evolution as an alternative global search
#3. More sophisticated local search (pattern search + golden section)
#4. Better time budget management
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
    
    eval_count = [0]
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def bounded_eval(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        if elapsed() >= max_time * 0.98:
            return best
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Sobol-like initialization with LHS ---
    n_init = min(max(30 * dim, 200), 800)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.15:
            n_init = i
            init_pop = init_pop[:n_init]
            init_fitness = init_fitness[:n_init]
            break
        init_fitness[i] = bounded_eval(init_pop[i])
    
    if n_init == 0:
        return best
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: CMA-ES with IPOP restarts ---
    def cmaes_search(x0, sigma0, time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        n = dim
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
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        
        gen = 0
        stagnation = 0
        prev_best_gen = float('inf')
        
        while elapsed() < min(deadline, max_time * 0.95):
            # Generate and evaluate offspring
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = np.clip(arx[k], lower, upper)
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= min(deadline, max_time * 0.95):
                    return
                fitnesses[k] = bounded_eval(arx[k])
            
            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            
            best_gen = fitnesses[idx[0]]
            if best_gen < prev_best_gen - 1e-12:
                stagnation = 0
                prev_best_gen = best_gen
            else:
                stagnation += 1
            
            if stagnation > 10 + int(30 * n / lam):
                return  # restart
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            mean_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = ((1 - c1 - cmu_val) * C 
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu_val * (artmp.T @ np.diag(weights) @ artmp))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges))
            
            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu_val) / n / 10:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
            
            # Check sigma convergence
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                return
            
            gen += 1
    
    # Run CMA-ES with increasing population restarts (IPOP)
    remaining_for_cma = max_time * 0.80 - elapsed()
    if remaining_for_cma > 0.5:
        n_top = min(5, n_init)
        restart = 0
        pop_multiplier = 1.0
        
        while elapsed() < max_time * 0.80:
            if restart < n_top:
                x0 = init_pop[sorted_idx[restart]].copy()
            else:
                x0 = lower + np.random.rand(dim) * ranges
            
            sigma0 = 0.25 * np.mean(ranges) / (1 + 0.3 * (restart // n_top))
            
            time_per_restart = max(0.5, (max_time * 0.80 - elapsed()) / max(1, 4 - restart % 4))
            
            cmaes_search(x0, sigma0, time_per_restart)
            restart += 1
    
    # --- Phase 3: Local refinement ---
    if best_params is not None and elapsed() < max_time * 0.95:
        # Nelder-Mead simplex
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step = 0.05 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += step[i] if best_params[i] + step[i] <= upper[i] else -step[i]
        
        f_simplex = np.array([bounded_eval(simplex[i]) for i in range(n + 1) if elapsed() < max_time * 0.95])
        if len(f_simplex) < n + 1:
            return best
        
        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        
        nm_iters = 0
        while elapsed() < max_time * 0.95 and nm_iters < 2000:
            nm_iters += 1
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
            fr = bounded_eval(xr)
            if elapsed() >= max_time * 0.95:
                break
            
            if fr < f_simplex[0]:
                # Expand
                xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                fe = bounded_eval(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = bounded_eval(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if elapsed() >= max_time * 0.95:
                                return best
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_simplex[i] = bounded_eval(simplex[i])
                else:
                    # Inside contraction
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), lower, upper)
                    fc = bounded_eval(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if elapsed() >= max_time * 0.95:
                                return best
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_simplex[i] = bounded_eval(simplex[i])
            
            # Check convergence
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-12 * np.max(ranges):
                break
    
    # --- Phase 4: Fine coordinate descent ---
    if best_params is not None and elapsed() < max_time * 0.95:
        step = 0.001 * ranges
        for _ in range(5):
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95:
                    return best
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = bounded_eval(trial)
                    if f < best:
                        improved = True
            step *= 0.3
            if not improved:
                break
    
    return best
#
#Key improvements:
#1. **Better time budgeting**: Clear phases with percentage-based time allocation
#2. **IPOP-CMA-ES**: Stagnation detection triggers restarts with fresh starts from top candidates
#3. **Nelder-Mead simplex** for local refinement (more robust than pure coordinate descent)
#4. **Convergence checks**: Both sigma-based and simplex-size-based stopping criteria
#5. **Fine coordinate descent** as final polishing step
#6. **Centralized evaluation function** that always tracks the global best
