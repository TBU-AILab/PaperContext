#Looking at the results, Algorithm 6 (score 22.77) performed best. Let me analyze what worked and identify improvements:
#
#1. SHADE with Lehmer mean was effective
#2. CMA-ES with restarts helped
#3. Simple local search was sufficient for refinement
#
#Key issues to address:
#- Better diversity maintenance during search
#- More aggressive exploration-exploitation balance
#- Improved CMA-ES restart strategy with population size increases
#- Better time allocation based on dimension
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
    
    evals = [0]
    
    def eval_func(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x_c.copy()
        return f
    
    # --- LHS + Opposition Initialization ---
    n_init = min(max(12 * dim, 60), 400)
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
    
    # Elite pool
    top_k = min(25, len(init_xs))
    elite_pool = [init_xs[sorted_idx[i]].copy() for i in range(top_k)]
    elite_fit = [init_fitness[sorted_idx[i]] for i in range(top_k)]
    
    def update_pool(x, f):
        if len(elite_pool) < 40:
            elite_pool.append(x.copy())
            elite_fit.append(f)
        elif f < max(elite_fit):
            worst = int(np.argmax(elite_fit))
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
        nfe_start = evals[0]
        
        while elapsed() < target_end:
            S_F, S_CR, S_df = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            sorted_fit_idx = np.argsort(fit[:pop_size])
            
            for i in range(pop_size):
                if elapsed() >= target_end:
                    for idx2 in sorted_fit_idx[:5]:
                        update_pool(pop[idx2], fit[idx2])
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
                if combined_size > 1:
                    r2 = np.random.randint(combined_size)
                    att = 0
                    while (r2 == i or r2 == r1) and att < 10:
                        r2 = np.random.randint(combined_size)
                        att += 1
                else:
                    r2 = r1
                
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
            
            nfe_used = evals[0] - nfe_start
            ratio = min(nfe_used / max_nfe, 1.0)
            new_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            if new_size < pop_size:
                keep = np.sort(np.argsort(fit[:pop_size])[:new_size])
                pop = pop[keep]
                fit = fit[keep]
                pop_size = new_size
        
        sorted_fit_idx = np.argsort(fit[:pop_size])
        for idx2 in sorted_fit_idx[:min(5, pop_size)]:
            update_pool(pop[idx2], fit[idx2])
    
    # --- CMA-ES ---
    def cma_es_search(x0, sigma0, time_budget, pop_mult=1.0):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        
        n = dim
        lam = int((4 + int(3 * np.log(n))) * pop_mult)
        lam = max(lam, 6)
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
        
        use_full = n <= 60
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
                    update_pool(best_x, best)
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
    
    # --- Local search with acceleration ---
    def local_search(x0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        x = x0.copy()
        fx = eval_func(x)
        step = 0.005 * ranges.copy()
        
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
                        for _ in range(15):
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
    
    # --- Nelder-Mead ---
    def nelder_mead(x0, time_budget, scale=0.02):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        n = dim
        
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
        
        alpha_nm, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(10000):
            if elapsed() >= target_end: return
            
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
            
            if np.max(np.abs(simplex[-1] - simplex[0]) / (ranges + 1e-30)) < 1e-16:
                break
    
    # === Execute ===
    # Phase 1: L-SHADE (33% time)
    if time_left() > 0.5:
        lshade_search(0.33)
    
    # Phase 2: CMA-ES restarts with IPOP strategy (40% time)
    restart = 0
    ipop_mult = 1.0
    while time_left() > max_time * 0.20:
        restart += 1
        tb = min(time_left() * 0.35, max_time * 0.15)
        if tb < 0.3:
            break
        
        if restart == 1 and best_x is not None:
            x0 = best_x.copy()
            s0 = 0.2 * np.mean(ranges)
            pm = 1.0
        elif restart % 5 == 0:
            # Large population global restart (IPOP)
            ipop_mult *= 2
            x0 = lower + np.random.random(dim) * ranges
            s0 = 0.4 * np.mean(ranges)
            pm = min(ipop_mult, 6.0)
        elif best_x is not None and np.random.random() < 0.55:
            # Small restart near best
            scale = 0.12 / np.sqrt(max(1, restart - 1))
            x0 = np.clip(best_x + scale * ranges * np.random.randn(dim), lower, upper)
            s0 = max(0.15 * np.mean(ranges) / np.sqrt(max(1, restart - 1)), 0.005 * np.mean(ranges))
            pm = 1.0
        elif elite_pool:
            idx = np.random.randint(len(elite_pool))
            x0 = elite_pool[idx].copy() + 0.05 * ranges * np.random.randn(dim)
            x0 = np.clip(x0, lower, upper)
            s0 = 0.15 * np.mean(ranges)
            pm = 1.0
        else:
            x0 = lower + np.random.random(dim) * ranges
            s0 = 0.3 * np.mean(ranges)
            pm = 1.0
        
        cma_es_search(x0, s0, tb, pop_mult=pm)
    
    # Phase 3: Nelder-Mead
    if best_x is not None and time_left() > max_time * 0.08:
        nelder_mead(best_x.copy(), time_left() * 0.4)
    
    # Phase 4: Final local search
    if best_x is not None and time_left() > 0.1:
        local_search(best_x.copy(), time_left() * 0.9)
    
    return best
