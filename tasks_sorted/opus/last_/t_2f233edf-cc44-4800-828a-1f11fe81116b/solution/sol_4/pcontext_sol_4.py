#Looking at the previous code, I can see several issues:
#
#1. **Double evaluation bug** in L-SHADE initialization - fitness is computed twice for non-focused, non-pop_init cases
#2. **Time allocation is suboptimal** - CMA-ES gets too little time for refinement
#3. **L-SHADE lacks current-to-pbest/1 with weighted recombination improvements**
#4. **No Nelder-Mead or pattern search for final polishing**
#5. **Restart diversity is too conservative**
#
#Let me redesign with cleaner code and better strategies:
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def lhs_init(n):
        pop = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            pop[:, j] = (perm + np.random.uniform(0, 1, n)) / n
        return lower + pop * ranges

    # ---- CMA-ES with restarts ----
    def cma_es(x0, sigma0, time_budget):
        nonlocal best, best_params
        if time_budget <= 0.1:
            return
        t_start = elapsed()
        
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
        cmu_cov = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        count_eval = 0
        no_improve_gen = 0
        local_best = best
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.05:
            solutions = []
            f_vals = []
            for _ in range(lam):
                if time_left() <= 0.03:
                    return
                z = np.random.randn(n)
                x = mean + sigma * (B @ (D * z))
                x = clip(x)
                f = eval_func(x)
                count_eval += 1
                solutions.append(x)
                f_vals.append(f)
            
            idx = np.argsort(f_vals)
            old_mean = mean.copy()
            
            mean = np.zeros(n)
            for i in range(mu):
                mean += weights[i] * solutions[idx[i]]
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            
            norm_ps = np.linalg.norm(ps)
            hsig = int(norm_ps / np.sqrt(1 - (1-cs)**(2*(count_eval/lam+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = np.zeros((n, mu))
            for i in range(mu):
                artmp[:, i] = (solutions[idx[i]] - old_mean) / max(sigma, 1e-30)
            
            C = (1 - c1 - cmu_cov) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_cov * (artmp * weights) @ artmp.T
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2.0 * np.max(ranges))
            
            if count_eval - eigeneval > lam / (c1 + cmu_cov + 1e-30) / n / 10:
                eigeneval = count_eval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
                    sigma = sigma0 * 0.5
            
            if best < local_best - 1e-12:
                no_improve_gen = 0
                local_best = best
            else:
                no_improve_gen += 1
            
            if sigma * np.max(D) < 1e-15 or no_improve_gen > 30 + 5*n:
                break

    # ---- L-SHADE ----
    def run_lshade(time_frac=0.5, pop_size_init=None, focused=False):
        nonlocal best, best_params
        if time_left() <= 0.2:
            return
        
        budget_time = time_left() * time_frac
        t_start = elapsed()
        
        if pop_size_init is None:
            pop_size_init = min(max(18 * dim, 50), 300)
        pop_size = pop_size_init
        N_init = pop_size_init
        N_min = 4
        
        if focused and best_params is not None:
            population = np.zeros((pop_size, dim))
            population[0] = best_params.copy()
            for i in range(1, pop_size):
                scale = np.random.uniform(0.005, 0.2)
                population[i] = clip(best_params + np.random.randn(dim) * ranges * scale)
        else:
            population = lhs_init(pop_size)
            # Opposition-based initialization
            opp_pop = lower + upper - population
            combined = np.vstack([population, opp_pop])
            combined_f = np.array([eval_func(combined[j]) for j in range(len(combined)) if time_left() > 0.1])
            if len(combined_f) < len(combined):
                combined_f = np.append(combined_f, [float('inf')] * (len(combined) - len(combined_f)))
            idx = np.argsort(combined_f)[:pop_size]
            population = combined[idx]
            fitness = combined_f[idx]
        
        if focused and best_params is not None:
            fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
        
        if time_left() <= 0:
            return
        
        H = min(100, max(6 * dim, 20))
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k_idx = 0
        archive = []
        max_archive = pop_size
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.1:
            S_F, S_CR, delta_f = [], [], []
            sort_idx = np.argsort(fitness)
            p_best_size = max(2, int(0.11 * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if time_left() <= 0.05:
                    return
                
                ri = np.random.randint(H)
                
                Fi = -1
                attempts = 0
                while Fi <= 0 and attempts < 20:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    attempts += 1
                Fi = np.clip(Fi, 0.01, 1.0)
                
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0, 1)
                
                pi = sort_idx[np.random.randint(p_best_size)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(union_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(union_size)
                
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - xr2)
                
                for j in range(dim):
                    if mutant[j] < lower[j]:
                        mutant[j] = (lower[j] + population[i][j]) / 2
                    elif mutant[j] > upper[j]:
                        mutant[j] = (upper[j] + population[i][j]) / 2
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = (np.random.random(dim) < CRi) | (np.arange(dim) == j_rand)
                trial[mask] = mutant[mask]
                
                trial_f = eval_func(trial)
                
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    
                    new_pop[i] = trial
                    new_fit[i] = trial_f
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_f))
                elif trial_f == fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            if len(S_F) > 0:
                w = np.array(delta_f)
                ws = w.sum()
                if ws > 0:
                    w = w / ws
                else:
                    w = np.ones(len(w)) / len(w)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                
                M_F[k_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                if np.max(scr) <= 0:
                    M_CR[k_idx] = -1
                else:
                    M_CR[k_idx] = np.sum(w * scr)
                k_idx = (k_idx + 1) % H
            
            ratio = min(1.0, (elapsed() - t_start) / max(budget_time, 1e-10))
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                si = np.argsort(fitness)
                population = population[si[:new_pop_size]]
                fitness = fitness[si[:new_pop_size]]
                pop_size = new_pop_size
                max_archive = pop_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))

    # ---- Nelder-Mead for local polishing ----
    def nelder_mead(x0, time_budget, initial_scale=0.05):
        nonlocal best, best_params
        if time_budget <= 0.1 or dim > 40:
            return
        t_start = elapsed()
        n = dim
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += initial_scale * ranges[i]
            simplex[i+1] = clip(simplex[i+1])
        
        f_simplex = np.array([eval_func(simplex[i]) for i in range(n+1)])
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.05:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n] = xe
                    f_simplex[n] = fe
                else:
                    simplex[n] = xr
                    f_simplex[n] = fr
            elif fr < f_simplex[n-1]:
                simplex[n] = xr
                f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[n] = xc
                        f_simplex[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                            if time_left() <= 0.05:
                                return
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[n]))
                    fc = eval_func(xc)
                    if fc < f_simplex[n]:
                        simplex[n] = xc
                        f_simplex[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                            if time_left() <= 0.05:
                                return
            
            # Check convergence
            if np.max(np.abs(f_simplex - f_simplex[0])) < 1e-14:
                break

    # ========== Main Pipeline ==========
    
    # Phase 1: L-SHADE global exploration (45% of time)
    run_lshade(time_frac=0.45)
    
    # Phase 2: CMA-ES from best (25% of remaining)
    if best_params is not None and time_left() > 0.5:
        sigma0 = 0.2 * np.mean(ranges)
        cma_es(best_params.copy(), sigma0, time_left() * 0.35)
    
    # Phase 3: Focused L-SHADE around best
    if time_left() > 0.5:
        run_lshade(pop_size_init=min(max(8*dim, 20), 80), time_frac=0.4, focused=True)
    
    # Phase 4: Nelder-Mead polishing
    if best_params is not None and time_left() > 0.5 and dim <= 40:
        nelder_mead(best_params.copy(), time_left() * 0.4, initial_scale=0.02)
    
    # Phase 5: CMA-ES refinement restarts
    restart_count = 0
    while time_left() > 0.3 and best_params is not None:
        restart_count += 1
        sigma0 = 0.03 * np.mean(ranges) / (restart_count ** 0.4)
        sigma0 = max(sigma0, 1e-5 * np.mean(ranges))
        perturbation = np.random.randn(dim) * ranges * 0.005 / (restart_count**0.3)
        x0 = clip(best_params + perturbation)
        cma_es(x0, sigma0, min(time_left() * 0.5, 3.0))
    
    return best
