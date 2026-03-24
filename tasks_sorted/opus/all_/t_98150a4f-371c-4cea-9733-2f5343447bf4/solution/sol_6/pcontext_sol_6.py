#Looking at the results, Algorithm 4 (score 22.77) performed best. Let me analyze what made it successful and what can be improved:
#
#1. SHADE with Lehmer mean worked well
#2. CMA-ES with restarts was effective
#3. Local search helped refine
#
#Key improvements I'll make:
#- Better L-SHADE with improved parameter adaptation and population reduction
#- More aggressive CMA-ES restart strategy with BIPOP-like approach
#- Add Nelder-Mead for medium-dim refinement
#- Better time management with more CMA-ES restarts
#- Improved mutation strategies in DE phase
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
    
    eval_count = [0]
    
    def eval_func(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        eval_count[0] += 1
        if f < best:
            best = f
            best_x = x_c.copy()
        return f
    
    # --- LHS + Opposition Initialization ---
    n_init = min(max(11 * dim, 50), 500)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    opp_points = lower + upper - init_points
    all_init = np.vstack([init_points, opp_points])
    np.random.shuffle(all_init)
    
    init_fitness = []
    init_xs = []
    for i in range(len(all_init)):
        if elapsed() >= max_time * 0.07:
            break
        f = eval_func(all_init[i])
        init_fitness.append(f)
        init_xs.append(all_init[i].copy())
    
    if len(init_fitness) == 0:
        x0 = lower + np.random.random(dim) * ranges
        return eval_func(x0)
    
    init_fitness = np.array(init_fitness)
    init_xs = np.array(init_xs)
    sorted_idx = np.argsort(init_fitness)
    
    # Keep a pool of diverse good solutions
    top_k = min(20, len(init_xs))
    elite_pool = [init_xs[sorted_idx[i]].copy() for i in range(top_k)]
    elite_fit = [init_fitness[sorted_idx[i]] for i in range(top_k)]
    
    def update_pool(x, f):
        nonlocal elite_pool, elite_fit
        if len(elite_pool) < 30:
            elite_pool.append(x.copy())
            elite_fit.append(f)
        elif f < max(elite_fit):
            worst = np.argmax(elite_fit)
            elite_pool[worst] = x.copy()
            elite_fit[worst] = f
    
    # --- L-SHADE ---
    def lshade_search(time_frac):
        nonlocal best, best_x
        target_end = elapsed() + max_time * time_frac
        
        N_init = max(min(10 * dim, 150), 30)
        N_min = 4
        pop_size = N_init
        H = 100
        
        n_elite = min(pop_size // 3, len(init_xs))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_xs[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if elapsed() < target_end:
                fit[i] = eval_func(pop[i])
        
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k = 0
        archive = []
        archive_max = int(N_init * 2.6)
        
        max_nfe = 15000 * dim
        nfe_start = eval_count[0]
        
        gen = 0
        while elapsed() < target_end:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            sorted_fit_idx = np.argsort(fit[:pop_size])
            
            for i in range(pop_size):
                if elapsed() >= target_end:
                    # Update pool with current best
                    for idx in sorted_fit_idx[:3]:
                        update_pool(pop[idx], fit[idx])
                    return
                
                ri = np.random.randint(H)
                mu_F = M_F[ri]
                mu_CR = M_CR[ri]
                
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + 0.1 * np.random.standard_cauchy()
                F_i = min(F_i, 1.0)
                
                CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0, 1)
                if mu_CR < 0:
                    CR_i = 0
                
                p_ratio = np.random.uniform(2.0/pop_size, 0.2)
                p = max(2, int(p_ratio * pop_size))
                p_best_idx = sorted_fit_idx[:p]
                x_pbest = pop[np.random.choice(p_best_idx)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                combined_size = pop_size + len(archive)
                if combined_size <= 1:
                    r2 = r1
                else:
                    r2 = np.random.randint(combined_size)
                    attempts = 0
                    while (r2 == i or r2 == r1) and attempts < 10:
                        r2 = np.random.randint(combined_size)
                        attempts += 1
                
                x_r2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + F_i * (x_pbest - pop[i]) + F_i * (pop[r1] - x_r2)
                
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i][d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i][d]) / 2
                
                cross = np.random.random(dim) < CR_i
                cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, pop[i])
                
                f_trial = eval_func(trial)
                
                if f_trial <= fit[i]:
                    df = fit[i] - f_trial
                    if f_trial < fit[i]:
                        archive.append(pop[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        S_df.append(df + 1e-30)
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            
            if S_F:
                S_df_a = np.array(S_df)
                w = S_df_a / np.sum(S_df_a)
                S_F_a = np.array(S_F)
                S_CR_a = np.array(S_CR)
                M_F[k] = np.sum(w * S_F_a**2) / (np.sum(w * S_F_a) + 1e-30)
                if np.max(S_CR_a) == 0:
                    M_CR[k] = -1
                else:
                    M_CR[k] = np.sum(w * S_CR_a**2) / (np.sum(w * S_CR_a) + 1e-30)
                k = (k + 1) % H
            
            nfe_used = eval_count[0] - nfe_start
            ratio = min(nfe_used / max_nfe, 1.0)
            new_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            if new_size < pop_size:
                keep = np.sort(np.argsort(fit[:pop_size])[:new_size])
                pop = pop[keep]
                fit = fit[keep]
                pop_size = new_size
        
        sorted_fit_idx = np.argsort(fit[:pop_size])
        for idx in sorted_fit_idx[:min(5, pop_size)]:
            update_pool(pop[idx], fit[idx])
    
    # --- CMA-ES ---
    def cma_es_search(x0, sigma0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 8)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n)*(1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = n <= 80
        if use_full:
            C = np.eye(n)
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            need_decomp = True
            decomp_delay = 0
        else:
            diag_cov = np.ones(n)
        
        counteval = 0
        stag = 0
        prev_best = float('inf')
        
        while elapsed() < target_end:
            if use_full and need_decomp:
                try:
                    C = (C + C.T) / 2
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    need_decomp = False
                    decomp_delay = 0
                except:
                    C = np.eye(n); eigvals = np.ones(n); eigvecs = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full:
                D = np.sqrt(eigvals)
                BD = eigvecs * D[np.newaxis, :]
                for kk in range(lam):
                    arx[kk] = mean + sigma * (BD @ arz[kk])
            else:
                sq = np.sqrt(np.maximum(diag_cov, 1e-20))
                for kk in range(lam):
                    arx[kk] = mean + sigma * sq * arz[kk]
            
            arx = np.clip(arx, lower, upper)
            
            arfit = np.full(lam, float('inf'))
            for kk in range(lam):
                if elapsed() >= target_end:
                    return
                arfit[kk] = eval_func(arx[kk])
                counteval += 1
            
            idx = np.argsort(arfit)
            
            gen_best = arfit[idx[0]]
            if gen_best >= prev_best - 1e-12 * (abs(prev_best) + 1e-30):
                stag += 1
            else:
                stag = 0
            prev_best = min(prev_best, gen_best)
            
            if stag > 10 + 30*n/lam:
                break
            
            old_mean = mean.copy()
            mean = weights @ arx[idx[:mu]]
            
            diff = mean - old_mean
            
            if use_full:
                invD = 1.0 / D
                z = eigvecs.T @ (diff / sigma)
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (eigvecs @ (invD * z))
            else:
                inv_sd = 1.0/np.sqrt(np.maximum(diag_cov, 1e-20))
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * inv_sd * diff / sigma
            
            nps = np.linalg.norm(ps)
            hsig = int(nps / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff) * diff / sigma
            
            if use_full:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                rank_mu = np.einsum('i,ij,ik->jk', weights, artmp, artmp)
                C = (1-c1-cmu_v+(1-hsig)*c1*cc*(2-cc))*C + c1*np.outer(pc,pc) + cmu_v*rank_mu
                decomp_delay += 1
                if decomp_delay >= max(1, int(1.0/(c1+cmu_v)/n/10)):
                    need_decomp = True
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                rm_d = np.sum(weights[:, None] * artmp**2, axis=0)
                diag_cov = (1-c1-cmu_v+(1-hsig)*c1*cc*(2-cc))*diag_cov + c1*pc**2 + cmu_v*rm_d
            
            sigma *= np.exp((cs/damps)*(nps/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5*np.max(ranges))
            
            mx = sigma * np.max(np.sqrt(eigvals) if use_full else np.sqrt(np.maximum(diag_cov,1e-20)))
            if mx < 1e-18:
                break
        
        update_pool(best_x, best)
    
    # --- CMA-ES with large population for exploration ---
    def cma_es_large(x0, sigma0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        
        n = dim
        lam = min(4 + int(3 * np.log(n)), 20) * 3  # 3x normal population
        lam = max(lam, 16)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n)*(1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        diag_cov = np.ones(n)
        
        counteval = 0
        stag = 0
        prev_best = float('inf')
        
        while elapsed() < target_end:
            sq = np.sqrt(np.maximum(diag_cov, 1e-20))
            arx = np.zeros((lam, n))
            for kk in range(lam):
                arx[kk] = mean + sigma * sq * np.random.randn(n)
            arx = np.clip(arx, lower, upper)
            
            arfit = np.full(lam, float('inf'))
            for kk in range(lam):
                if elapsed() >= target_end:
                    return
                arfit[kk] = eval_func(arx[kk])
                counteval += 1
            
            idx = np.argsort(arfit)
            gen_best = arfit[idx[0]]
            if gen_best >= prev_best - 1e-12 * (abs(prev_best) + 1e-30):
                stag += 1
            else:
                stag = 0
            prev_best = min(prev_best, gen_best)
            if stag > 15 + 30*n/lam:
                break
            
            old_mean = mean.copy()
            mean = weights @ arx[idx[:mu]]
            diff = mean - old_mean
            
            inv_sd = 1.0/np.sqrt(np.maximum(diag_cov, 1e-20))
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * inv_sd * diff / sigma
            nps = np.linalg.norm(ps)
            hsig = int(nps / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff) * diff / sigma
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            rm_d = np.sum(weights[:, None] * artmp**2, axis=0)
            diag_cov = (1-c1-cmu_v+(1-hsig)*c1*cc*(2-cc))*diag_cov + c1*pc**2 + cmu_v*rm_d
            
            sigma *= np.exp((cs/damps)*(nps/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5*np.max(ranges))
            
            if sigma * np.max(np.sqrt(np.maximum(diag_cov,1e-20))) < 1e-18:
                break
    
    # --- Nelder-Mead ---
    def nelder_mead(x0, time_budget, scale=0.03):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        n = dim
        
        alpha_nm, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        step = scale * ranges
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] = np.clip(x0[i] + step[i], lower[i], upper[i])
        
        f_simplex = np.full(n+1, float('inf'))
        for i in range(n+1):
            if elapsed() >= target_end: return
            f_simplex[i] = eval_func(simplex[i])
        
        for _ in range(10000):
            if elapsed() >= target_end:
                return
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(centroid + alpha_nm*(centroid - simplex[-1]), lower, upper)
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                xe = np.clip(centroid + gamma*(xr - centroid), lower, upper)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + rho*(xr - centroid), lower, upper)
                else:
                    xc = np.clip(centroid + rho*(simplex[-1] - centroid), lower, upper)
                fc = eval_func(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc; f_simplex[-1] = fc
                else:
                    for i in range(1, n+1):
                        if elapsed() >= target_end: return
                        simplex[i] = simplex[0] + shrink*(simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        f_simplex[i] = eval_func(simplex[i])
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30))
            if spread < 1e-16:
                break
    
    # --- Coordinate descent with acceleration ---
    def coord_descent(x0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        x = x0.copy()
        fx = eval_func(x)
        step = 0.002 * ranges.copy()
        
        while elapsed() < target_end:
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if elapsed() >= target_end: return
                for direction in [1, -1]:
                    xt = x.copy()
                    xt[d] = np.clip(x[d] + direction*step[d], lower[d], upper[d])
                    ft = eval_func(xt)
                    if ft < fx:
                        x = xt; fx = ft; improved = True
                        accel = 2.0
                        for _ in range(20):
                            if elapsed() >= target_end: return
                            xt2 = x.copy()
                            xt2[d] = np.clip(x[d] + direction*step[d]*accel, lower[d], upper[d])
                            ft2 = eval_func(xt2)
                            if ft2 < fx:
                                x = xt2; fx = ft2
                                accel *= 1.5
                            else:
                                break
                        break
            if not improved:
                step *= 0.5
                if np.max(step/ranges) < 1e-16:
                    break
    
    # --- Gradient estimation search ---
    def gradient_search(x0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        x = x0.copy()
        fx = eval_func(x)
        
        h = 1e-6 * ranges
        alpha = 0.001 * np.mean(ranges)
        
        for iteration in range(300):
            if elapsed() >= target_end:
                return
            
            grad = np.zeros(dim)
            if dim <= 30:
                dims_to_check = range(dim)
            else:
                dims_to_check = np.random.choice(dim, 30, replace=False)
            
            for d in dims_to_check:
                if elapsed() >= target_end: return
                xp = x.copy(); xp[d] = min(x[d] + h[d], upper[d])
                xm = x.copy(); xm[d] = max(x[d] - h[d], lower[d])
                fp = eval_func(xp)
                fm = eval_func(xm)
                grad[d] = (fp - fm) / (xp[d] - xm[d] + 1e-30)
            
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-20:
                break
            
            direction = -grad / gnorm
            
            step = alpha
            improved = False
            for _ in range(12):
                if elapsed() >= target_end: return
                x_new = np.clip(x + step * direction, lower, upper)
                f_new = eval_func(x_new)
                if f_new < fx - 1e-8 * step * gnorm:
                    x = x_new; fx = f_new
                    alpha = min(step * 1.3, np.mean(ranges))
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                alpha *= 0.5
                if alpha < 1e-16 * np.mean(ranges):
                    break
    
    # === Execute ===
    # Phase 1: L-SHADE (30% time)
    if time_left() > 0.5:
        lshade_search(0.30)
    
    # Phase 2: CMA-ES restarts with BIPOP-like strategy (40% time)
    restart = 0
    small_budget_total = 0
    large_budget_total = 0
    
    while time_left() > max_time * 0.25:
        restart += 1
        
        tb = min(time_left() * 0.35, max_time * 0.15)
        if tb < 0.3:
            break
        
        # Alternate between small (local) and large (global) restarts
        if restart <= 2:
            # First restarts: from best solution
            if best_x is not None:
                x0 = best_x.copy()
                s0 = 0.2 * np.mean(ranges)
            else:
                x0 = lower + np.random.random(dim) * ranges
                s0 = 0.3 * np.mean(ranges)
            cma_es_search(x0, s0, tb)
        elif restart % 3 == 0:
            # Large restart for exploration
            x0 = lower + np.random.random(dim) * ranges
            s0 = 0.4 * np.mean(ranges)
            cma_es_large(x0, s0, tb)
        elif best_x is not None and np.random.random() < 0.6:
            # Small restart near best
            perturbation = 0.1 * ranges * np.random.randn(dim) / np.sqrt(max(1, restart - 2))
            x0 = np.clip(best_x + perturbation, lower, upper)
            s0 = max(0.15 * np.mean(ranges) / np.sqrt(max(1, restart - 2)), 0.005 * np.mean(ranges))
            cma_es_search(x0, s0, tb)
        else:
            # Restart from a random elite
            if elite_pool:
                idx = np.random.randint(len(elite_pool))
                x0 = elite_pool[idx].copy()
                s0 = 0.15 * np.mean(ranges)
            else:
                x0 = lower + np.random.random(dim) * ranges
                s0 = 0.3 * np.mean(ranges)
            cma_es_search(x0, s0, tb)
    
    # Phase 3: Gradient-based local search
    if best_x is not None and time_left() > max_time * 0.12:
        gradient_search(best_x.copy(), time_left() * 0.3)
    
    # Phase 4: Nelder-Mead
    if best_x is not None and time_left() > max_time * 0.07:
        nelder_mead(best_x.copy(), time_left() * 0.4, scale=0.02)
    
    # Phase 5: Final coordinate descent
    if best_x is not None and time_left() > 0.1:
        coord_descent(best_x.copy(), time_left() * 0.9)
    
    return best
