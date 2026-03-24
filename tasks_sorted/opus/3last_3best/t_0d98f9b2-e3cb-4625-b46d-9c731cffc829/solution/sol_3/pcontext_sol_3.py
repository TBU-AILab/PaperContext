#Looking at the results, the best algorithm (score 22.09) uses SHADE-DE with coordinate descent local search. Let me analyze what can be improved:
#
#1. **The SHADE-DE memory update is slightly wrong** - weighted mean for CR should use weights properly
#2. **Need better time allocation** - more time for local refinement
#3. **Multiple restarts of DE with different strategies** could help escape local optima
#4. **Combine DE with a proper CMA-ES that uses full covariance** for moderate dimensions, diagonal for high dimensions
#5. **Add opposition-based learning** for better exploration
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # === Phase 1: SHADE with L-SHADE population reduction ===
    def run_shade(time_budget_frac, pop_init=None):
        nonlocal best, best_x
        time_limit = elapsed() + max_time * time_budget_frac
        
        pop_size_init = min(max(40, 10 * dim), 300)
        pop_size_min = max(4, dim // 2)
        
        if pop_init is not None and len(pop_init) > 0:
            pop_size_init = len(pop_init)
            pop = pop_init.copy()
        else:
            # LHS initialization
            pop = np.zeros((pop_size_init, dim))
            for d in range(dim):
                perm = np.random.permutation(pop_size_init)
                pop[:, d] = (perm + np.random.uniform(0, 1, pop_size_init)) / pop_size_init
            pop = lower + pop * ranges
        
        pop_size = len(pop)
        fit = np.array([evaluate(pop[i]) for i in range(pop_size) if elapsed() < time_limit])
        if len(fit) < pop_size:
            fit = np.append(fit, [float('inf')] * (pop_size - len(fit)))
        
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        mem_k = 0
        
        archive = []
        max_archive = pop_size_init
        
        nfe = pop_size
        max_nfe = pop_size_init * 200  # for L-SHADE reduction
        
        gen = 0
        while elapsed() < time_limit:
            gen += 1
            S_F, S_CR, S_delta = [], [], []
            
            sorted_idx = np.argsort(fit[:pop_size])
            
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            
            for i in range(pop_size):
                if elapsed() >= time_limit:
                    break
                
                ri = np.random.randint(0, H)
                
                # Generate F via Cauchy
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                while F_i <= 0:
                    F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
                F_i = min(F_i, 1.0)
                
                # Generate CR via Normal
                CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
                
                # current-to-pbest/1
                p = max(2, int(max(0.05, 0.2 - 0.15 * nfe / max_nfe) * pop_size))
                pbest_idx = sorted_idx[np.random.randint(0, p)]
                
                # r1 from population
                r1 = np.random.randint(0, pop_size - 1)
                if r1 >= i:
                    r1 += 1
                
                # r2 from population + archive
                union_size = pop_size + len(archive)
                r2 = np.random.randint(0, union_size - 1)
                if r2 >= i and r2 < pop_size:
                    r2 += 1
                if r2 == r1 and r2 < pop_size:
                    r2 = (r2 + 1) % union_size
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - x_r2)
                
                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR_i
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Bounce-back boundary handling
                out_low = trial < lower
                out_high = trial > upper
                trial[out_low] = (lower[out_low] + pop[i][out_low]) / 2
                trial[out_high] = (upper[out_high] + pop[i][out_high]) / 2
                trial = clip(trial)
                
                f_trial = evaluate(trial)
                nfe += 1
                
                if f_trial <= fit[i]:
                    delta = fit[i] - f_trial
                    if f_trial < fit[i]:
                        archive.append(pop[i].copy())
                        if len(archive) > max_archive:
                            archive.pop(np.random.randint(len(archive)))
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        S_delta.append(delta + 1e-30)
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop[:pop_size] = new_pop
            fit[:pop_size] = new_fit
            
            # Update memory with weighted Lehmer mean
            if S_F:
                w = np.array(S_delta)
                w = w / w.sum()
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[mem_k % H] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[mem_k % H] = np.sum(w * scr)
                mem_k += 1
            
            # L-SHADE: reduce population
            new_pop_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * nfe / max_nfe)))
            if new_pop_size < pop_size:
                keep_idx = np.argsort(fit[:pop_size])[:new_pop_size]
                pop = pop[keep_idx].copy()
                fit = fit[keep_idx].copy()
                pop_size = new_pop_size
        
        return pop[:pop_size], fit[:pop_size]

    # Run main SHADE
    run_shade(0.55)
    
    # === Phase 2: Restart SHADE from near-best region ===
    if best_x is not None and elapsed() < max_time * 0.70:
        restart_pop_size = min(max(20, 5 * dim), 150)
        restart_pop = np.zeros((restart_pop_size, dim))
        for i in range(restart_pop_size):
            if np.random.random() < 0.7:
                restart_pop[i] = best_x + 0.05 * ranges * np.random.randn(dim)
            else:
                restart_pop[i] = lower + np.random.random(dim) * ranges
            restart_pop[i] = clip(restart_pop[i])
        remaining_frac = (max_time * 0.78 - elapsed()) / max_time
        if remaining_frac > 0.02:
            run_shade(remaining_frac, restart_pop)

    # === Phase 3: CMA-ES local search ===
    if best_x is not None:
        use_full_cov = dim <= 50
        
        for restart in range(10):
            if elapsed() >= max_time * 0.93:
                break
            
            n = dim
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
            chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            sigma = 0.05 * np.mean(ranges) / (restart + 1)
            mean = best_x.copy() + 0.005 * ranges * np.random.randn(dim) / (restart + 1)
            mean = clip(mean)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            if use_full_cov:
                C = np.eye(n)
                eigeneval = 0
                B = np.eye(n)
                D = np.ones(n)
            else:
                C_diag = np.ones(n)
            
            time_limit = min(elapsed() + max_time * 0.06, max_time * 0.94)
            gen = 0
            
            while elapsed() < time_limit:
                gen += 1
                
                if use_full_cov:
                    # Decompose if needed
                    if gen == 1 or eigeneval >= lam / (c1 + cmu_val) / n / 5:
                        try:
                            D_sq, B = np.linalg.eigh(C)
                            D = np.sqrt(np.maximum(D_sq, 1e-20))
                            eigeneval = 0
                        except:
                            C = np.eye(n)
                            B = np.eye(n)
                            D = np.ones(n)
                    
                    arz = np.random.randn(lam, n)
                    arx = np.zeros((lam, n))
                    for k_i in range(lam):
                        arx[k_i] = mean + sigma * (B @ (D * arz[k_i]))
                        arx[k_i] = clip(arx[k_i])
                else:
                    arz = np.random.randn(lam, n)
                    arx = np.zeros((lam, n))
                    for k_i in range(lam):
                        arx[k_i] = mean + sigma * (np.sqrt(C_diag) * arz[k_i])
                        arx[k_i] = clip(arx[k_i])
                
                fitvals = np.zeros(lam)
                for k_i in range(lam):
                    if elapsed() >= time_limit:
                        break
                    fitvals[k_i] = evaluate(arx[k_i])
                
                if elapsed() >= time_limit:
                    break
                
                idx = np.argsort(fitvals)
                old_mean = mean.copy()
                mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
                
                zmean = (mean - old_mean) / sigma
                
                if use_full_cov:
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ zmean)
                else:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (zmean / np.sqrt(C_diag))
                
                hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen)) / chiN < 1.4 + 2.0/(n+1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * zmean
                
                if use_full_cov:
                    artmp = (arx[idx[:mu]] - old_mean) / sigma
                    C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                    for k_i in range(mu):
                        C += cmu_val * weights[k_i] * np.outer(artmp[k_i], artmp[k_i])
                    eigeneval += lam
                else:
                    C_diag = (1 - c1 - cmu_val) * C_diag + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag)
                    for k_i in range(mu):
                        diff = (arx[idx[k_i]] - old_mean) / sigma
                        C_diag += cmu_val * weights[k_i] * diff**2
                    C_diag = np.maximum(C_diag, 1e-20)
                
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-20, np.max(ranges))
                
                if sigma < 1e-16:
                    break

    # === Phase 4: Fine-grained coordinate descent ===
    if best_x is not None:
        x = best_x.copy()
        fx = best
        step = 0.005 * ranges
        
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    return best
                for sign in [1, -1]:
                    xt = x.copy()
                    xt[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    ft = evaluate(xt)
                    if ft < fx:
                        x, fx = xt, ft
                
