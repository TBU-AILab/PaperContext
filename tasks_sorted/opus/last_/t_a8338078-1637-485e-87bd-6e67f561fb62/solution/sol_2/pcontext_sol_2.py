#Looking at the previous algorithm, I can identify several issues:
#
#1. The CMA-ES restarts too aggressively with doubling population - most time is spent on large populations that may not help
#2. The initial sampling takes too much time for high dimensions
#3. The local refinement phase is too simplistic
#4. Need better balance between exploration and exploitation
#
#Let me implement a more refined approach with better CMA-ES, Nelder-Mead local search, and smarter restart strategy:
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
    center = (lower + upper) / 2.0
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def eval_f(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Sobol-like quasi-random + LHS sampling ---
    n_init = min(max(20 * dim, 100), 800)
    if dim > 50:
        n_init = min(n_init, 200)
    
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i] + lower[i]
    
    population = []
    for i in range(n_init):
        if not time_ok(0.12):
            break
        f = eval_f(init_points[i])
        population.append((f, init_points[i].copy()))
    
    # Also try center and corners
    if time_ok(0.12):
        eval_f(center.copy())
    
    population.sort(key=lambda t: t[0])
    
    # --- Phase 2: CMA-ES with smart restarts ---
    def cmaes_run(x0, sigma0, lam_mult=1.0, max_budget_frac=0.3):
        nonlocal best, best_x
        
        run_start = elapsed()
        max_run_time = max_budget_frac * max_time
        
        n = dim
        lam = max(int((4 + int(3 * np.log(n))) * lam_mult), 6)
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        xmean = np.clip(x0.copy(), lower, upper)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 80)
        
        if use_full:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0
        else:
            diagC = np.ones(n)
        
        gen = 0
        best_local = float('inf')
        stag_count = 0
        flat_count = 0
        
        while time_ok(0.93):
            run_elapsed = elapsed() - run_start
            if run_elapsed > max_run_time:
                return
            
            # Eigendecomposition for full covariance
            if use_full and eigen_countdown <= 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, B = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    eigen_countdown = max(1, int(lam / (10 * n)))
                except:
                    return
            if use_full:
                eigen_countdown -= 1
            
            # Generate offspring with mirrored sampling
            arxs = []
            arfitness = []
            
            for k in range(lam):
                if not time_ok(0.93):
                    return
                
                z = np.random.randn(n)
                if use_full:
                    y = B @ (D * z)
                else:
                    y = np.sqrt(diagC) * z
                
                x = xmean + sigma * y
                # Bounce back boundary handling
                for d_i in range(n):
                    while x[d_i] < lower[d_i] or x[d_i] > upper[d_i]:
                        if x[d_i] < lower[d_i]:
                            x[d_i] = 2 * lower[d_i] - x[d_i]
                        if x[d_i] > upper[d_i]:
                            x[d_i] = 2 * upper[d_i] - x[d_i]
                
                x = np.clip(x, lower, upper)
                f = eval_f(x)
                arxs.append(x)
                arfitness.append(f)
            
            idx = np.argsort(arfitness)
            
            local_best = arfitness[idx[0]]
            if local_best < best_local - 1e-12 * (abs(best_local) + 1):
                best_local = local_best
                stag_count = 0
                flat_count = 0
            else:
                stag_count += 1
            
            # Check for flat fitness
            if abs(arfitness[idx[0]] - arfitness[idx[-1]]) < 1e-15 * (abs(arfitness[idx[0]]) + 1e-30):
                flat_count += 1
                if flat_count > 5:
                    return
            
            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arxs[idx[i]]
            
            diff = xmean - xold
            
            if use_full:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / max(sigma, 1e-30)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / (np.sqrt(diagC) * max(sigma, 1e-30) + 1e-30)
            
            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / max(sigma, 1e-30)
            
            if use_full:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                
                rank_mu_update = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_update += weights[i] * np.outer(artmp[i], artmp[i])
                
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * rank_mu_update
            else:
                artmp_sq = np.zeros(n)
                for i in range(mu):
                    d_vec = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                    artmp_sq += weights[i] * d_vec**2
                
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * artmp_sq
                diagC = np.maximum(diagC, 1e-20)
            
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5 * np.max(ranges))
            
            gen += 1
            
            if sigma < 1e-15:
                return
            if stag_count > 15 + 5 * n / lam:
                return
            if gen > 200 + 100 * n / lam:
                return
            
            # Condition number check for full cov
            if use_full and gen % 10 == 0:
                try:
                    cond = np.max(D) / (np.min(D) + 1e-30)
                    if cond > 1e14:
                        return
                except:
                    pass
    
    # Run CMA-ES restarts with IPOP strategy
    lam_mult = 1.0
    restart = 0
    num_good = min(5, len(population))
    
    while time_ok(0.93):
        rem = max_time - elapsed()
        
        if restart == 0 and best_x is not None:
            x0 = best_x.copy()
            sig0 = 0.15 * np.mean(ranges)
            budget_frac = min(0.25, rem * 0.4 / max_time)
        elif restart < num_good and restart < len(population):
            x0 = population[restart][1].copy()
            sig0 = 0.25 * np.mean(ranges)
            budget_frac = min(0.2, rem * 0.35 / max_time)
        else:
            # Random restart from perturbation of best or random
            if np.random.random() < 0.4 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * 0.3 * ranges
                x0 = np.clip(x0, lower, upper)
                sig0 = 0.2 * np.mean(ranges)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.3 * np.mean(ranges)
            budget_frac = min(0.2, rem * 0.3 / max_time)
        
        cmaes_run(x0, sig0, lam_mult, budget_frac)
        restart += 1
        
        # IPOP: increase population, but with a cap and reset cycle
        if restart <= num_good:
            lam_mult = 1.0
        else:
            lam_mult = min(lam_mult * 2, 16)
            if lam_mult >= 16:
                lam_mult = 1.0
    
    # --- Phase 3: Pattern search local refinement ---
    if best_x is not None and time_ok(0.95):
        x_curr = best_x.copy()
        f_curr = best
        
        for scale_exp in range(-2, -7, -1):
            if not time_ok(0.97):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved_any = True
            while improved_any and time_ok(0.98):
                improved_any = False
                for i in range(dim):
                    if not time_ok(0.98):
                        break
                    for sign in [1.0, -1.0]:
                        x_trial = x_curr.copy()
                        x_trial[i] = np.clip(x_trial[i] + sign * step_size, lower[i], upper[i])
                        f_trial = eval_f(x_trial)
                        if f_trial < f_curr:
                            f_curr = f_trial
                            x_curr = x_trial
                            improved_any = True
                            break  # Move to next dimension after improvement
    
    # --- Phase 4: Final Nelder-Mead if time remains ---
    if best_x is not None and time_ok(0.96) and dim <= 30:
        n = dim
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = [best_x.copy()]
        simplex_f = [best]
        
        delta = 0.01 * ranges
        for i in range(n):
            if not time_ok(0.96):
                break
            xi = best_x.copy()
            xi[i] += delta[i]
            xi = np.clip(xi, lower, upper)
            fi = eval_f(xi)
            simplex.append(xi)
            simplex_f.append(fi)
        
        if len(simplex) == n + 1:
            for nm_iter in range(500 * n):
                if not time_ok(0.99):
                    break
                
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                centroid = np.mean([simplex[i] for i in range(n)], axis=0)
                
                # Reflect
                xr = centroid + alpha_nm * (centroid - simplex[-1])
                xr = np.clip(xr, lower, upper)
                fr = eval_f(xr)
                
                if fr < simplex_f[0]:
                    xe = centroid + gamma_nm * (xr - centroid)
                    xe = np.clip(xe, lower, upper)
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho_nm * (xr - centroid)
                    else:
                        xc = centroid + rho_nm * (simplex[-1] - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = eval_f(xc)
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.99):
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            simplex_f[i] = eval_f(simplex[i])
                
                spread = max(np.max(np.abs(simplex[-1] - simplex[0])), 1e-30)
                if spread < 1e-14:
                    break
    
    return best
