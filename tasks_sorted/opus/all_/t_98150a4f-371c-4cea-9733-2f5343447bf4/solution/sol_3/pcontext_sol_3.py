#Looking at the results, Algorithm 1 (SHADE + CMA-ES + local search) performed best with 22.77. Let me analyze weaknesses and improve:
#
#1. **Better SHADE with linear population size reduction (L-SHADE)**
#2. **Improved CMA-ES with proper eigendecomposition caching**
#3. **Nelder-Mead as additional local search**
#4. **Better time allocation and restart strategies**
#5. **Opposition-based learning for initialization**
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
    
    # --- Phase 1: LHS + Opposition-based Initialization ---
    n_init = min(max(12 * dim, 60), 500)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    # Add opposition-based points
    opp_points = lower + upper - init_points
    all_init = np.vstack([init_points, opp_points])
    np.random.shuffle(all_init)
    
    init_fitness = []
    init_xs = []
    for i in range(len(all_init)):
        if elapsed() >= max_time * 0.10:
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
    
    # --- L-SHADE ---
    def lshade_search(time_frac):
        nonlocal best, best_x
        target_end = elapsed() + max_time * time_frac
        
        N_init = max(min(10 * dim, 150), 30)
        N_min = max(4, dim // 2)
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
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = pop_size
        
        max_nfe = 10000 * dim
        nfe_start = evals[0]
        
        gen = 0
        while elapsed() < target_end:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            sorted_fit_idx = np.argsort(fit)
            
            for i in range(pop_size):
                if elapsed() >= target_end:
                    pop = new_pop; fit = new_fit
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
                
                p = max(2, int(np.clip(0.05 + 0.15 * np.random.random(), 0.05, 0.2) * pop_size))
                p_best_idx = sorted_fit_idx[:p]
                x_pbest = pop[np.random.choice(p_best_idx)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                combined_pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                combined_pool = [c for c in combined_pool if c != i and c != r1]
                if not combined_pool:
                    combined_pool = [c for c in range(pop_size) if c != i]
                r2_idx = np.random.choice(combined_pool)
                x_r2 = pop[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
                
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
                S_df = np.array(S_df)
                w = S_df / np.sum(S_df)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                M_F[k] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
            
            # Linear population size reduction
            nfe_used = evals[0] - nfe_start
            ratio = min(nfe_used / max_nfe, 1.0)
            new_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            if new_size < pop_size:
                worst_idx = np.argsort(fit)[::-1]
                keep = np.sort(np.argsort(fit)[:new_size])
                pop = pop[keep]
                fit = fit[keep]
                pop_size = new_size
                archive_max = pop_size
    
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
        
        use_full = n <= 100
        if use_full:
            C = np.eye(n)
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            need_update = True
        else:
            diag_cov = np.ones(n)
        
        counteval = 0
        stag = 0
        prev_med = float('inf')
        
        while elapsed() < target_end:
            if use_full and need_update:
                try:
                    C = (C + C.T) / 2
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    need_update = False
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
            
            med = np.median(arfit)
            if med >= prev_med - 1e-12 * (abs(prev_med) + 1e-30):
                stag += 1
            else:
                stag = 0
            prev_med = min(prev_med, med)
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
                need_update = True
            else:
                artmp = (arx[idx[:mu]] - old_mean) / sigma
                rm_d = np.sum(weights[:, None] * artmp**2, axis=0)
                diag_cov = (1-c1-cmu_v+(1-hsig)*c1*cc*(2-cc))*diag_cov + c1*pc**2 + cmu_v*rm_d
            
            sigma *= np.exp((cs/damps)*(nps/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5*np.max(ranges))
            
            mx = sigma * np.max(np.sqrt(eigvals) if use_full else np.sqrt(np.maximum(diag_cov,1e-20)))
            if mx < 1e-18:
                break
    
    # --- Nelder-Mead ---
    def nelder_mead(x0, time_budget):
        nonlocal best, best_x
        target_end = elapsed() + time_budget
        n = dim
        
        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        step = 0.05 * ranges
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] = np.clip(x0[i] + step[i], lower[i], upper[i])
        
        f_simplex = np.array([eval_func(simplex[i]) for i in range(n+1) if elapsed() < target_end])
        if len(f_simplex) < n+1:
            return
        
        for _ in range(5000):
            if elapsed() >= target_end:
                return
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(centroid + alpha*(centroid - simplex[-1]), lower, upper)
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
            
            if np.max(np.abs(simplex[-1] - simplex[0])/ranges) < 1e-16:
                break
    
    # --- Coordinate descent ---
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
                        for _ in range(10):
                            if elapsed() >= target_end: return
                            xt2 = x.copy()
                            xt2[d] = np.clip(x[d] + direction*step[d]*2, lower[d], upper[d])
                            ft2 = eval_func(xt2)
                            if ft2 < fx:
                                x = xt2; fx = ft2
                            else:
                                break
                        break
            if not improved:
                step *= 0.5
                if np.max(step/ranges) < 1e-16:
                    break
    
    # === Execute ===
    if time_left() > 0.5:
        lshade_search(0.35)
    
    restart = 0
    while time_left() > max_time * 0.18:
        restart += 1
        if restart == 1 and best_x is not None:
            x0 = best_x.copy()
        elif best_x is not None and np.random.random() < 0.5:
            x0 = best_x + 0.15*ranges*np.random.randn(dim)/np.sqrt(restart)
            x0 = np.clip(x0, lower, upper)
        else:
            x0 = lower + np.random.random(dim)*ranges
        
        s0 = max(0.25*np.mean(ranges)/np.sqrt(restart), 0.005*np.mean(ranges))
        tb = min(time_left()*0.4, max_time*0.2)
        if tb < 0.3: break
        cma_es_search(x0, s0, tb)
    
    if best_x is not None and time_left() > max_time * 0.08:
        nelder_mead(best_x.copy(), time_left() * 0.45)
    
    if best_x is not None and time_left() > 0.1:
        local_search(best_x.copy(), time_left() * 0.9)
    
    return best
