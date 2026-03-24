#Looking at the previous algorithm, I'll make several improvements:
#
#1. **L-SHADE with linear population size reduction** for better convergence
#2. **CMA-ES as a complementary optimizer** for local refinement
#3. **Multi-restart strategy** with intelligent budget allocation
#4. **Improved local search with adaptive step sizes**
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

    # --- CMA-ES local search ---
    def cma_es_local(x0, sigma0=0.1, max_evals=2000):
        if time_left() <= 0:
            return
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
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        local_evals = 0
        
        while local_evals < max_evals and time_left() > 0:
            # Sample
            solutions = []
            f_vals = []
            for _ in range(lam):
                if time_left() <= 0:
                    return
                z = np.random.randn(n)
                x = mean + sigma * (B @ (D * z))
                x = clip(x)
                f = eval_func(x)
                local_evals += 1
                solutions.append(x)
                f_vals.append(f)
            
            # Sort
            idx = np.argsort(f_vals)
            old_mean = mean.copy()
            
            # Recombination
            mean = np.zeros(n)
            for i in range(mu):
                mean += weights[i] * solutions[idx[i]]
            
            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(local_evals/lam+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Covariance update
            artmp = np.zeros((n, mu))
            for i in range(mu):
                artmp[:, i] = (solutions[idx[i]] - old_mean) / sigma
            
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp * weights) @ artmp.T
            
            # Step-size update
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            # Update B, D
            if local_evals - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = local_evals
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
            
            # Convergence
            if sigma * np.max(D) < 1e-12:
                break

    # --- L-SHADE ---
    def run_lshade(pop_size_init, max_time_frac=0.6):
        nonlocal best, best_params
        if time_left() <= 0:
            return
        
        budget_time = time_left() * max_time_frac
        t_start = elapsed()
        
        pop_size = pop_size_init
        N_init = pop_size_init
        N_min = 4
        
        # LHS init
        population = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size)
            population[:, j] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
        population = lower + population * ranges
        
        fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
        
        H = 100
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k_idx = 0
        archive = []
        max_archive = pop_size
        
        gen = 0
        max_gen_est = max(100, int(budget_time * 50))  # rough estimate
        
        while (elapsed() - t_start) < budget_time and time_left() > 0:
            gen += 1
            
            S_F, S_CR, delta_f = [], [], []
            sort_idx = np.argsort(fitness)
            p_best_size = max(2, int(0.11 * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if time_left() <= 0:
                    return
                
                ri = np.random.randint(H)
                
                # Cauchy for F
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi >= 1:
                        Fi = 1.0
                        break
                
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0, 1)
                
                # For terminal dims, sometimes set CR=0
                if M_CR[ri] < 0:
                    CRi = 0
                
                pi = sort_idx[np.random.randint(p_best_size)]
                
                # r1
                r1 = np.random.randint(pop_size)
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                
                # r2 from pop+archive
                union_size = pop_size + len(archive)
                r2 = np.random.randint(union_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(union_size)
                
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                # current-to-pbest/1
                mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - xr2)
                
                # Bounce-back
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
                    delta_f.append(fitness[i] - trial_f)
                elif trial_f == fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            
            # Update memory
            if len(S_F) > 0:
                w = np.array(delta_f)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                
                M_F[k_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k_idx] = np.sum(w * scr)
                k_idx = (k_idx + 1) % H
            
            # Linear population size reduction
            ratio = min(1.0, (elapsed() - t_start) / max(budget_time, 1e-10))
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                sort_idx = np.argsort(fitness)
                population = population[sort_idx[:new_pop_size]]
                fitness = fitness[sort_idx[:new_pop_size]]
                pop_size = new_pop_size
                
                # Trim archive
                max_archive = pop_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))

    # --- Main strategy ---
    # Phase 1: L-SHADE exploration
    init_pop = min(max(18 * dim, 50), 200)
    run_lshade(init_pop, max_time_frac=0.55)
    
    # Phase 2: CMA-ES refinement from best
    if best_params is not None and time_left() > 1:
        sigma_init = 0.15 * np.mean(ranges)
        cma_budget = max(500, int(dim * 100))
        cma_es_local(best_params, sigma0=sigma_init, max_evals=cma_budget)
    
    # Phase 3: Another L-SHADE with smaller population around best
    if time_left() > 1 and best_params is not None:
        run_lshade(min(max(8 * dim, 20), 80), max_time_frac=0.6)
    
    # Phase 4: Final CMA-ES polish
    if best_params is not None and time_left() > 0.5:
        cma_es_local(best_params, sigma0=0.01 * np.mean(ranges), max_evals=max(200, dim * 50))
    
    # Phase 5: Very fine local search with random restarts
    while time_left() > 0.2 and best_params is not None:
        perturbed = best_params + np.random.randn(dim) * ranges * 0.001
        cma_es_local(clip(perturbed), sigma0=0.005 * np.mean(ranges), max_evals=max(100, dim * 20))
    
    return best
#
#Key improvements:
#1. **L-SHADE with linear population reduction** - proven state-of-the-art DE variant that gradually reduces population for better convergence
#2. **CMA-ES local search** - covariance matrix adaptation for powerful local optimization with learned correlations
#3. **Multi-phase strategy** - exploration (L-SHADE) → refinement (CMA-ES) → re-exploration → polish
#4. **Better F parameter sampling** - resampling negative F values instead of clipping to 0
#5. **Adaptive sigma in CMA-ES** with proper eigendecomposition updates
#6. **Multiple refinement passes** with decreasing step sizes for final precision
