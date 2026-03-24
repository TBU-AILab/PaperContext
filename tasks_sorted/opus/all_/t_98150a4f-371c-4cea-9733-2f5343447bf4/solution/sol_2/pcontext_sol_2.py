#Looking at the previous algorithms, I can see they both implement CMA-ES with initialization and local search. Algorithm 1 (score 32.52) is better than Algorithm 2 (162.46). Let me analyze weaknesses and improve:
#
#1. **Better global search**: Use SHADE (Success-History based Adaptive DE) which is more effective than basic DE
#2. **Improved CMA-ES**: Fix the eigendecomposition bug in algorithm 1 and add proper restart logic
#3. **Multi-strategy approach**: Combine multiple search strategies
#4. **Better time management**: More adaptive time allocation
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
        x_clipped = np.clip(x, lower, upper)
        f = func(x_clipped)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x_clipped.copy()
        return f
    
    # --- Phase 1: LHS Initialization ---
    n_init = min(max(15 * dim, 80), 400)
    init_points = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_points[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.12:
            n_init = i
            break
        init_fitness[i] = eval_func(init_points[i])
    
    init_points = init_points[:n_init]
    init_fitness = init_fitness[:n_init]
    
    if n_init == 0:
        x0 = lower + np.random.random(dim) * ranges
        return eval_func(x0)
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: SHADE (Success-History Adaptive DE) ---
    def shade_search(time_budget_frac):
        nonlocal best, best_x
        target_end = elapsed() + max_time * time_budget_frac
        
        pop_size = max(min(8 * dim, 120), 30)
        H = 50  # history size
        
        # Initialize population
        n_elite = min(pop_size // 3, n_init)
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_points[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if elapsed() < target_end:
                fit[i] = eval_func(pop[i])
        
        # Memory for F and CR
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0  # memory index
        
        # Archive
        archive = []
        archive_max = pop_size
        
        gen = 0
        while elapsed() < target_end:
            gen += 1
            S_F = []
            S_CR = []
            S_df = []
            
            # Generate trial vectors
            for i in range(pop_size):
                if elapsed() >= target_end:
                    return
                
                # Select from memory
                ri = np.random.randint(H)
                mu_F = M_F[ri]
                mu_CR = M_CR[ri]
                
                # Generate F (Cauchy)
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + 0.1 * np.random.standard_cauchy()
                F_i = min(F_i, 1.0)
                
                # Generate CR (Gaussian)
                CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0, 1)
                
                # current-to-pbest/1 with archive
                p = max(2, int(0.1 * pop_size))
                p_best_idx = np.argsort(fit)[:p]
                x_pbest = pop[np.random.choice(p_best_idx)]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                # r2 from pop + archive
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                combined = [c for c in combined if c != i and c != r1]
                if len(combined) == 0:
                    combined = [c for c in range(pop_size) if c != i]
                r2_idx = np.random.choice(combined)
                if r2_idx < pop_size:
                    x_r2 = pop[r2_idx]
                else:
                    x_r2 = archive[r2_idx - pop_size]
                
                mutant = pop[i] + F_i * (x_pbest - pop[i]) + F_i * (pop[r1] - x_r2)
                
                # Bounce-back boundary handling
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i][d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i][d]) / 2
                
                # Binomial crossover
                cross_points = np.random.random(dim) < CR_i
                j_rand = np.random.randint(dim)
                cross_points[j_rand] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = eval_func(trial)
                
                if f_trial <= fit[i]:
                    df = fit[i] - f_trial
                    if f_trial < fit[i]:
                        archive.append(pop[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        S_df.append(df)
                    pop[i] = trial
                    fit[i] = f_trial
            
            # Update memory
            if len(S_F) > 0:
                S_df = np.array(S_df)
                w = S_df / (np.sum(S_df) + 1e-30)
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                
                # Lehmer mean for F
                M_F[k] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                # Weighted mean for CR
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
    
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
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
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
            eigen_countdown = 0
        else:
            diag_cov = np.ones(n)
        
        counteval = 0
        stag_count = 0
        prev_median = float('inf')
        
        while elapsed() < target_end:
            if use_full and eigen_countdown <= 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                except:
                    C = np.eye(n)
                    eigvals = np.ones(n)
                    eigvecs = np.eye(n)
                eigen_countdown = max(1, int(1.0/(c1 + cmu_val)/n/10))
            
            if use_full:
                eigen_countdown -= 1
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_full:
                D = np.sqrt(eigvals)
                BD = eigvecs * D[np.newaxis, :]
                for k in range(lam):
                    arx[k] = mean + sigma * (BD @ arz[k])
            else:
                sqrt_d = np.sqrt(np.maximum(diag_cov, 1e-20))
                for k in range(lam):
                    arx[k] = mean + sigma * sqrt_d * arz[k]
            
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.full(lam, float('inf'))
            for k in range(lam):
                if elapsed() >= target_end:
                    return
                arfitness[k] = eval_func(arx[k])
                counteval += 1
            
            arindex = np.argsort(arfitness)
            
            med_fit = np.median(arfitness)
            if med_fit >= prev_median - 1e-12 * abs(prev_median + 1e-30):
                stag_count += 1
            else:
                stag_count = 0
            prev_median = min(prev_median, med_fit)
            
            if stag_count > 10 + 30*n/lam:
                break
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * arx[arindex[:mu]], axis=0)
            
            diff = mean - old_mean
            
            if use_full:
                invD = 1.0 / D
                z = eigvecs.T @ (diff / sigma)
                ps = (1 - cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (eigvecs @ (invD * z))
            else:
                inv_sd = 1.0/np.sqrt(np.maximum(diag_cov, 1e-20))
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * inv_sd * diff / sigma
            
            norm_ps = np.linalg.norm(ps)
            hsig = int(norm_ps / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff) * diff / sigma
            
            if use_full:
                artmp = (arx[arindex[:mu]] - old_mean) / sigma
                rank_mu = np.einsum('i,ij,ik->jk', weights, artmp, artmp)
                C = (1-c1-cmu_val+(1-hsig)*c1*cc*(2-cc))*C + c1*np.outer(pc,pc) + cmu_val*rank_mu
            else:
                artmp = (arx[arindex[:mu]] - old_mean) / sigma
                rank_mu_d = np.sum(weights[:, np.newaxis] * artmp**2, axis=0)
                diag_cov = (1-c1-cmu_val+(1-hsig)*c1*cc*(2-cc))*diag_cov + c1*pc**2 + cmu_val*rank_mu_d
            
            sigma *= np.exp((cs/damps)*(norm_ps/chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5*np.max(ranges))
            
            max_std = sigma * np.max(np.sqrt(eigvals) if use_full else np.sqrt(np.maximum(diag_cov,1e-20)))
            if max_std < 1e-18:
                break
    
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
                if elapsed() >= target_end:
                    return
                for direction in [1, -1]:
                    xt = x.copy()
                    xt[d] = np.clip(x[d] + direction*step[d], lower[d], upper[d])
                    ft = eval_func(xt)
                    if ft < fx:
                        x = xt; fx = ft; improved = True
                        for _ in range(8):
                            if elapsed() >= target_end: return
                            xt2 = x.copy()
                            xt2[d] = np.clip(x[d] + direction*step[d]*2, lower[d], upper[d])
                            ft2 = eval_func(xt2)
                            if ft2 < fx:
                                x = xt2; fx = ft2
                            else: break
                        break
            if not improved:
                step *= 0.5
                if np.max(step/ranges) < 1e-16: break
    
    # --- Execute ---
    if time_left() > 0.5:
        shade_search(0.35)
    
    restart = 0
    while time_left() > max_time * 0.12:
        restart += 1
        if restart == 1 and best_x is not None:
            x0 = best_x.copy()
        elif best_x is not None and np.random.random() < 0.6:
            x0 = best_x + 0.2*ranges*np.random.randn(dim)/np.sqrt(restart)
            x0 = np.clip(x0, lower, upper)
        else:
            x0 = lower + np.random.random(dim)*ranges
        
        s0 = max(0.2*np.mean(ranges)/np.sqrt(restart), 0.005*np.mean(ranges))
        tb = min(time_left()*0.45, max_time*0.2)
        if tb < 0.3: break
        cma_es_search(x0, s0, tb)
    
    if best_x is not None and time_left() > 0.1:
        local_search(best_x.copy(), time_left()*0.95)
    
    return best
