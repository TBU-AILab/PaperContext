#Looking at the results, algorithm 1 (score 28.52) performed much better than algorithm 2 (157.82). Let me improve upon algorithm 1 by:
#
#1. Adding a more sophisticated DE with SHADE-style adaptation
#2. Better CMA-ES with proper eigendecomposition frequency
#3. Adding a Nelder-Mead local search phase
#4. Better time allocation and restart strategy
#5. Multi-start from diverse initial points
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
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    evals = [0]
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(30 * dim, 200), 2000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.10):
            break
        init_fitness[i] = eval_func(init_pop[i])
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: SHADE (Success-History based Adaptive DE) ---
    def shade_phase(time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        pop_size = min(max(8 * dim, 50), 300)
        H = 100  # history size
        
        # Initialize population from best LHS + random
        n_elite = min(pop_size // 3, np.sum(init_fitness < float('inf')))
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
        
        for i in range(n_elite, pop_size):
            if not time_ok() or elapsed() > deadline:
                return
            fit[i] = eval_func(pop[i])
        
        # SHADE memory
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        archive = []
        max_archive = pop_size
        
        gen = 0
        while time_ok() and elapsed() < deadline:
            gen += 1
            S_F = []
            S_CR = []
            S_df = []
            
            # Generate F and CR for each individual
            r_idx = np.random.randint(0, H, pop_size)
            F_vals = np.zeros(pop_size)
            CR_vals = np.zeros(pop_size)
            
            for i in range(pop_size):
                # Cauchy for F
                while True:
                    Fi = M_F[r_idx[i]] + 0.1 * np.random.standard_cauchy()
                    if Fi > 0:
                        break
                F_vals[i] = min(Fi, 1.0)
                
                # Normal for CR
                CRi = np.clip(M_CR[r_idx[i]] + 0.1 * np.random.randn(), 0, 1)
                CR_vals[i] = CRi
            
            for i in range(pop_size):
                if not time_ok() or elapsed() > deadline:
                    return
                
                Fi = F_vals[i]
                CRi = CR_vals[i]
                
                # current-to-pbest/1 with archive
                p = max(2, int(pop_size * max(0.05, 0.2 - 0.15 * gen / (gen + 100))))
                pbest_idx = np.argsort(fit)[:p]
                xpbest = pop[pbest_idx[np.random.randint(p)]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                # r2 from pop + archive
                combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                if i in combined:
                    combined_filtered = [c for c in combined if c != i]
                else:
                    combined_filtered = combined
                if r1 in combined_filtered:
                    combined_filtered = [c for c in combined_filtered if c != r1]
                
                if len(combined_filtered) == 0:
                    continue
                r2_idx = np.random.choice(combined_filtered)
                
                if r2_idx < pop_size:
                    xr2 = pop[r2_idx]
                else:
                    xr2 = archive[r2_idx - pop_size]
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Bounce-back for bounds
                for d_i in range(dim):
                    if mutant[d_i] < lower[d_i]:
                        mutant[d_i] = (lower[d_i] + pop[i, d_i]) / 2
                    elif mutant[d_i] > upper[d_i]:
                        mutant[d_i] = (upper[d_i] + pop[i, d_i]) / 2
                
                # Binomial crossover
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[jrand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = eval_func(trial)
                if f_trial < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(fit[i] - f_trial)
                    
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    
                    pop[i] = trial
                    fit[i] = f_trial
                elif f_trial == fit[i]:
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
    def cma_es_run(x0, sigma0, time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        
        mean = x0.copy()
        sigma = sigma0
        
        gen = 0
        no_improve = 0
        prev_best = best
        
        while time_ok() and elapsed() < deadline:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                # Bounce-back
                for d_i in range(n):
                    if arx[k, d_i] < lower[d_i]:
                        arx[k, d_i] = lower[d_i] + abs(arx[k, d_i] - lower[d_i]) % ranges[d_i]
                    elif arx[k, d_i] > upper[d_i]:
                        arx[k, d_i] = upper[d_i] - abs(arx[k, d_i] - upper[d_i]) % ranges[d_i]
                arx[k] = clip(arx[k])
            
            fitnesses = np.zeros(lam)
            for k in range(lam):
                if not time_ok() or elapsed() > deadline:
                    return
                fitnesses[k] = eval_func(arx[k])
            
            idx = np.argsort(fitnesses)
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / sigma
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            
            C = ((1 - c1 - cmu_val) * C +
                 c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                 cmu_val * np.sum(weights[:, None, None] * (artmp[:, :, None] * artmp[:, None, :]), axis=0))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            
            # Eigendecomposition
            if gen % max(1, int(1.0 / (c1 + cmu_val + 1e-20) / n / 10)) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            if best < prev_best - 1e-10:
                no_improve = 0
                prev_best = best
            else:
                no_improve += 1
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges) or no_improve > 40 + 10*n:
                break
    
    # Time allocation
    rem = max_time - elapsed()
    shade_budget = rem * 0.40
    shade_phase(shade_budget)
    
    # CMA-ES restarts (IPOP-style)
    restart = 0
    while time_ok(0.92):
        rem = max_time * 0.97 - elapsed()
        if rem < 0.2:
            break
        n_left = max(1, min(6, int(rem / max(0.5, rem / 4))))
        t_budget = rem / n_left * 0.85
        
        if restart == 0:
            x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim) * ranges
            sig = 0.25 * np.max(ranges)
        elif restart % 3 == 0:
            x0 = lower + np.random.random(dim) * ranges
            sig = 0.5 * np.max(ranges)
        else:
            x0 = best_params + 0.15 * ranges * np.random.randn(dim)
            x0 = clip(x0)
            sig = 0.15 * np.max(ranges) * (1 + restart * 0.3)
        
        cma_es_run(x0, min(sig, np.max(ranges)), t_budget)
        restart += 1
    
    # --- Local refinement: Nelder-Mead ---
    if time_ok(0.92) and best_params is not None:
        n = dim
        scale = 0.02 * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1, i] += scale[i] if np.random.random() < 0.5 else -scale[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_s = np.array([eval_func(simplex[i]) for i in range(n + 1) if time_ok(0.97)])
        if len(f_s) < n + 1:
            return best
        
        while time_ok(0.97):
            order = np.argsort(f_s)
            simplex = simplex[order]
            f_s = f_s[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + 1.0 * (centroid - simplex[-1]))
            fr = eval_func(xr)
            
            if f_s[0] <= fr < f_s[-2]:
                simplex[-1] = xr; f_s[-1] = fr
            elif fr < f_s[0]:
                xe = clip(centroid + 2.0 * (xr - centroid))
                if not time_ok(0.97): break
                fe = eval_func(xe)
                if fe < fr: simplex[-1] = xe; f_s[-1] = fe
                else: simplex[-1] = xr; f_s[-1] = fr
            else:
                xc = clip(centroid + 0.5 * (simplex[-1] - centroid))
                if not time_ok(0.97): break
                fc = eval_func(xc)
                if fc < f_s[-1]: simplex[-1] = xc; f_s[-1] = fc
                else:
                    for i in range(1, n + 1):
                        if not time_ok(0.97): return best
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i])
                        f_s[i] = eval_func(simplex[i])
            
            if np.max(np.abs(f_s - f_s[0])) < 1e-15:
                break
    
    return best
