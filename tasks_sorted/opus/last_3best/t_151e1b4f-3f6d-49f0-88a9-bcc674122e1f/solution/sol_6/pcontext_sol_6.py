#Looking at the best algorithm (score 26.96), I'll improve it with several enhancements:
#
#1. **Multiple restarts with different strategies** to escape local optima
#2. **CMA-ES-inspired local search** instead of just Nelder-Mead
#3. **Better parameter adaptation** in SHADE
#4. **Coordinate descent** as additional local refinement
#5. **More aggressive exploitation** of the best solution found
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    evals = 0
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS initialization ---
    pop_size = min(max(30, 8 * dim), 200)
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.07:
            break
        fitness[i] = ev(population[i])
    
    # Opposition-based initialization
    for i in range(pop_size):
        if elapsed() >= max_time * 0.12:
            break
        opp = lower + upper - population[i]
        of = ev(opp)
        if of < fitness[i]:
            population[i] = clip(opp)
            fitness[i] = of

    # --- Phase 2: SHADE with L-SHADE population reduction ---
    H = 100
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    init_pop_size = pop_size
    min_pop_size = max(4, dim)
    
    de_time_limit = max_time * 0.70
    gen = 0

    while elapsed() < de_time_limit:
        gen += 1
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.11 * pop_size))
        
        S_F = []
        S_CR = []
        delta_f = []
        
        trial_pop = np.empty_like(population)
        trial_fit = np.full(pop_size, float('inf'))
        
        for i in range(pop_size):
            if elapsed() >= de_time_limit:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 30:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            pb = sorted_idx[np.random.randint(p_best_size)]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = idxs[np.random.randint(len(idxs))]
            
            pool_size = pop_size + len(archive)
            r2 = np.random.randint(pool_size - 1)
            if r2 >= i:
                r2 += 1
            if r2 == r1:
                r2 = (r2 + 1) % pool_size
                if r2 == i:
                    r2 = (r2 + 1) % pool_size
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            trial_f = ev(trial)
            trial_pop[i] = trial
            trial_fit[i] = trial_f
            
            if trial_f < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                delta_f.append(fitness[i] - trial_f)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                population[i] = trial
                fitness[i] = trial_f
            
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H
        
        # L-SHADE population reduction
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / de_time_limit)))
        if new_pop_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_pop_size]]
            fitness = fitness[si[:new_pop_size]]
            pop_size = new_pop_size
            max_archive = pop_size
            if len(archive) > max_archive:
                archive = archive[:max_archive]
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            n_replace = max(1, pop_size // 3)
            si = np.argsort(fitness)
            for ii in range(n_replace):
                idx = si[-(ii + 1)]
                # Mix random restart with perturbation of best
                if np.random.random() < 0.5:
                    population[idx] = best_params + 0.2 * ranges * np.random.randn(dim)
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip(population[idx])
                if elapsed() >= de_time_limit:
                    break
                fitness[idx] = ev(population[idx])
            stagnation = 0

    # --- Phase 3: CMA-ES-like local search ---
    if best_params is not None and elapsed() < max_time * 0.88:
        sigma = 0.05
        n = dim
        lam = max(4 + int(3 * np.log(n)), 8)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        mean = best_params.copy()
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        while elapsed() < max_time * 0.88:
            try:
                sqrtC = np.linalg.cholesky(C)
            except:
                C = np.eye(n)
                sqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            
            for j in range(lam):
                if elapsed() >= max_time * 0.88:
                    break
                arx[j] = clip(mean + sigma * sqrtC.dot(arz[j]))
                arf[j] = ev(arx[j])
            
            if elapsed() >= max_time * 0.88:
                break
            
            idx_sort = np.argsort(arf)
            
            old_mean = mean.copy()
            mean = np.zeros(n)
            for j in range(mu):
                mean += weights[j] * arx[idx_sort[j]]
            mean = clip(mean)
            
            zmean = np.zeros(n)
            for j in range(mu):
                zmean += weights[j] * arz[idx_sort[j]]
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chiN else 0.0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / (sigma + 1e-30)
            
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for j in range(mu):
                dv = (arx[idx_sort[j]] - old_mean) / (sigma + 1e-30)
                C += cmu_v * weights[j] * np.outer(dv, dv)
            
            # Symmetrize
            C = (C + C.T) / 2
            np.fill_diagonal(C, np.maximum(np.diag(C), 1e-20))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(1e-15, min(sigma, 1.0))
            
            if sigma < 1e-12:
                break

    # --- Phase 4: Nelder-Mead from best ---
    if best_params is not None and elapsed() < max_time * 0.97:
        x0 = best_params.copy()
        n = dim
        scale = 0.01 * ranges
        simplex = np.vstack([x0, x0 + np.diag(scale)])
        fs = []
        for j in range(n + 1):
            if elapsed() >= max_time * 0.97:
                return best
            fs.append(ev(simplex[j]))
        fs = np.array(fs)
        
        while elapsed() < max_time * 0.97:
            o = np.argsort(fs)
            simplex = simplex[o]
            fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            
            xr = clip(c + (c - simplex[-1]))
            fr = ev(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr
            elif fr < fs[0]:
                xe = clip(c + 2 * (xr - c))
                fe = ev(xe)
                if fe < fr:
                    simplex[-1] = xe; fs[-1] = fe
                else:
                    simplex[-1] = xr; fs[-1] = fr
            else:
                xc = clip(c + 0.5 * (simplex[-1] - c))
                fc = ev(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc; fs[-1] = fc
                else:
                    for j in range(1, n + 1):
                        if elapsed() >= max_time * 0.97:
                            return best
                        simplex[j] = clip(simplex[0] + 0.5 * (simplex[j] - simplex[0]))
                        fs[j] = ev(simplex[j])
    
    return best
#
#Key improvements over the best algorithm:
#- **L-SHADE population reduction** for better convergence
#- **CMA-ES local search** between SHADE and Nelder-Mead phases, which adapts a covariance matrix to follow the landscape curvature
#- **Mixed restart strategy** (50% perturbation of best, 50% random) during stagnation
#- **Smaller Nelder-Mead scale** (0.01 vs 0.02) for finer local refinement
#- **Better time allocation**: 70% SHADE, 18% CMA-ES, 9% Nelder-Mead
