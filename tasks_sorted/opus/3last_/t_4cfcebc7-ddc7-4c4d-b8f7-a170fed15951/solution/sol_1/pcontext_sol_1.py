#Looking at the previous algorithm, I can see it combines Latin Hypercube Sampling, Nelder-Mead, and Differential Evolution. Let me improve it with a more sophisticated approach:
#
#1. Better DE variant (SHADE - Success-History based Adaptive DE)
#2. More restarts of local search from diverse starting points
#3. Better time management
#4. CMA-ES-like adaptation for local search
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
    
    def remaining():
        return max_time - elapsed()
    
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
    
    # --- Phase 1: Sobol-like initialization via Latin Hypercube ---
    n_init = min(max(30 * dim, 200), 1000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if remaining() < max_time * 0.05:
            return best
        init_fitness[i] = eval_func(init_pop[i])
    
    sorted_idx = np.argsort(init_fitness)
    
    # --- Phase 2: SHADE (Success-History Adaptive DE) ---
    def shade():
        nonlocal best, best_params
        
        pop_size = min(max(8 * dim, 50), 300)
        n_elite_init = min(pop_size // 3, len(sorted_idx))
        
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite_init):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite_init, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
            if remaining() < max_time * 0.05:
                return
            fit[i] = eval_func(pop[i])
        
        # SHADE memory
        H = 100
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        
        p_min = max(2, pop_size // 20)
        p_max = max(2, pop_size // 5)
        
        archive = []
        archive_max = pop_size
        
        gen = 0
        while remaining() > max_time * 0.12:
            gen += 1
            S_F = []
            S_CR = []
            S_delta = []
            
            # Sort population for current-to-pbest
            fit_order = np.argsort(fit)
            
            for i in range(pop_size):
                if remaining() < max_time * 0.12:
                    return
                
                # Pick random memory index
                ri = np.random.randint(H)
                mu_F = M_F[ri]
                mu_CR = M_CR[ri]
                
                # Generate F from Cauchy
                Fi = mu_F + 0.1 * np.random.standard_cauchy()
                while Fi <= 0:
                    Fi = mu_F + 0.1 * np.random.standard_cauchy()
                Fi = min(Fi, 1.0)
                
                # Generate CR from Normal
                CRi = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
                
                # current-to-pbest/1
                p = np.random.randint(p_min, p_max + 1)
                pbest_idx = fit_order[np.random.randint(p)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                # r2 from pop + archive
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - xr2)
                
                # Bounce-back clipping
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + pop[i, d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + pop[i, d]) / 2
                
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                
                f_trial = eval_func(trial)
                
                if f_trial <= fit[i]:
                    delta = fit[i] - f_trial
                    if delta > 0:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_delta.append(delta)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    pop[i] = trial
                    fit[i] = f_trial
            
            # Update memory
            if len(S_F) > 0:
                S_delta = np.array(S_delta)
                w = S_delta / S_delta.sum()
                S_F = np.array(S_F)
                S_CR = np.array(S_CR)
                # Lehmer mean for F
                M_F[k] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                # Weighted mean for CR
                M_CR[k] = np.sum(w * S_CR)
                k = (k + 1) % H
    
    shade()
    
    # --- Phase 3: Local search (Nelder-Mead) from best ---
    def nelder_mead(x0, scale=0.05):
        nonlocal best, best_params
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            p = x0.copy()
            p[i] += ranges[i] * scale * (1 if np.random.random() > 0.5 else -1)
            simplex[i + 1] = clip(p)
        
        f_s = np.array([eval_func(simplex[i]) for i in range(n + 1) if remaining() > max_time * 0.01])
        if len(f_s) < n + 1:
            return
        
        for _ in range(5000):
            if remaining() < max_time * 0.01:
                return
            order = np.argsort(f_s)
            simplex = simplex[order]
            f_s = f_s[order]
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-14:
                return
            
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(2 * c - simplex[-1])
            fr = eval_func(xr)
            
            if fr < f_s[0]:
                xe = clip(c + 2 * (xr - c))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1], f_s[-1] = xe, fe
                else:
                    simplex[-1], f_s[-1] = xr, fr
            elif fr < f_s[-2]:
                simplex[-1], f_s[-1] = xr, fr
            else:
                if fr < f_s[-1]:
                    xc = clip(c + 0.5 * (xr - c))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1], f_s[-1] = xc, fc; continue
                else:
                    xc = clip(c - 0.5 * (c - simplex[-1]))
                    fc = eval_func(xc)
                    if fc < f_s[-1]:
                        simplex[-1], f_s[-1] = xc, fc; continue
                for i in range(1, n + 1):
                    if remaining() < max_time * 0.01: return
                    simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    f_s[i] = eval_func(simplex[i])
    
    if best_params is not None and remaining() > 0.1:
        nelder_mead(best_params, scale=0.02)
    
    if best_params is not None and remaining() > 0.1:
        nelder_mead(best_params, scale=0.005)
    
    return best
#
#Key improvements:
#1. **SHADE algorithm** - state-of-the-art adaptive DE with success-history based parameter adaptation (Lehmer mean for F, weighted mean for CR)
#2. **Archive mechanism** - stores replaced solutions for better diversity in mutation
#3. **Bounce-back clipping** instead of simple clip, maintaining diversity near boundaries
#4. **Better initialization** with more samples
#5. **Multiple local search rounds** with decreasing scale for fine-tuning
#6. **Current-to-pbest/1** mutation strategy which is more effective than current-to-best/1
