#Looking at the previous algorithm, I need to analyze what could be improved. The score of 21.52 suggests the algorithm is getting stuck in local optima. Key issues I see:
#
#1. **L-SHADE population re-initialization is missing** - the second L-SHADE call reuses no knowledge of the best region
#2. **CMA-ES has numerical stability issues** and can waste time
#3. **Time allocation is rigid** - phases may not get enough time
#4. **No diversity mechanism** between restarts
#5. **Missing opposition-based learning and other exploration tricks**
#
#Let me redesign with a more robust approach:
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Latin Hypercube Sampling
    def lhs_init(n):
        pop = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            pop[:, j] = (perm + np.random.uniform(0, 1, n)) / n
        return lower + pop * ranges

    # Opposition-based point
    def opposition(x):
        return lower + upper - x

    # ---- CMA-ES ----
    def cma_es(x0, sigma0, max_evals=5000, time_frac=None):
        nonlocal best, best_params
        t_start = elapsed()
        budget_time = time_frac if time_frac else time_left()
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 6)
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
        no_improve = 0
        prev_best = best
        
        while count_eval < max_evals and (elapsed() - t_start) < budget_time and time_left() > 0.1:
            solutions = []
            f_vals = []
            for _ in range(lam):
                if time_left() <= 0.05:
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
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            
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
            
            # Check stagnation
            if best < prev_best - 1e-10:
                no_improve = 0
                prev_best = best
            else:
                no_improve += 1
            
            if sigma * np.max(D) < 1e-14 or no_improve > 50 + 10*n:
                break

    # ---- L-SHADE ----
    def run_lshade(pop_init=None, pop_size_init=None, time_frac=0.5, focused=False):
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
        
        if pop_init is not None:
            population = pop_init.copy()
            pop_size = len(population)
            N_init = pop_size
        elif focused and best_params is not None:
            # Initialize around best
            population = np.zeros((pop_size, dim))
            population[0] = best_params.copy()
            for i in range(1, pop_size):
                scale = np.random.uniform(0.01, 0.3)
                population[i] = clip(best_params + np.random.randn(dim) * ranges * scale)
        else:
            population = lhs_init(pop_size)
            # Add opposition
            n_opp = min(pop_size // 4, pop_size)
            for i in range(n_opp):
                opp = opposition(population[i])
                population = np.vstack([population, opp.reshape(1, -1)])
            # Evaluate all and keep best pop_size
            all_f = np.array([eval_func(population[j]) for j in range(len(population))])
            if time_left() <= 0:
                return
            idx = np.argsort(all_f)[:pop_size]
            population = population[idx]
            all_f = all_f[idx]
            fitness = all_f
        
        if not (focused and best_params is not None) and pop_init is None:
            pass
        else:
            fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
        
        if time_left() <= 0:
            return
        
        H = min(100, 6 * dim)
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k_idx = 0
        archive = []
        max_archive = pop_size
        
        gen = 0
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.1:
            gen += 1
            
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
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi >= 1:
                        Fi = 1.0
                        break
                
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0, 1)
                
                pi = sort_idx[np.random.randint(p_best_size)]
                
                r1 = np.random.randint(pop_size)
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                
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
                    if len(archive) < max_archive:
                        archive.append(population[i].copy())
                    elif len(archive) > 0:
                        archive[np.random.randint(len(archive))] = population[i].copy()
                    
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

    # ========== Main multi-strategy pipeline ==========
    
    # Phase 1: L-SHADE global exploration (50% of time)
    run_lshade(time_frac=0.50)
    
    # Phase 2: CMA-ES from best (20% of time)
    if best_params is not None and time_left() > 0.5:
        sigma0 = 0.15 * np.mean(ranges)
        cma_es(best_params.copy(), sigma0, max_evals=10000, time_frac=time_left() * 0.4)
    
    # Phase 3: Focused L-SHADE around best (15% of time)
    if time_left() > 0.5:
        run_lshade(pop_size_init=min(max(10*dim, 20), 100), time_frac=0.5, focused=True)
    
    # Phase 4: CMA-ES refinement cycles with restarts
    restart_count = 0
    while time_left() > 0.3 and best_params is not None:
        restart_count += 1
        sigma0 = 0.05 * np.mean(ranges) / (restart_count ** 0.5)
        sigma0 = max(sigma0, 1e-4 * np.mean(ranges))
        # Slight perturbation for diversity
        x0 = clip(best_params + np.random.randn(dim) * ranges * 0.01 / (restart_count**0.3))
        cma_es(x0, sigma0, max_evals=max(200, dim*30), time_frac=min(time_left()*0.4, 2.0))
    
    return best
#
#Key improvements:
#1. **Opposition-based learning** in initial population to better cover search space
#2. **Focused L-SHADE phase** that initializes population around best solution found
#3. **Better stagnation detection** in CMA-ES with restart capability
#4. **Adaptive CMA-ES restart sigma** that decreases with restart count for progressive refinement
#5. **Fixed fitness initialization bug** - previous code had issues with second L-SHADE not knowing search region
#6. **Better time budget allocation** - more balanced phases
#7. **Improved numerical safeguards** throughout
