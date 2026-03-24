#Looking at the current best algorithm (score 12.05), I'll analyze what can be improved:
#
#1. **CMA-ES had a bug**: `gen` counter from SHADE was used in CMA-ES hsig calculation
#2. **Better time allocation**: More time for CMA-ES which is excellent for local refinement
#3. **Multiple CMA-ES restarts with decreasing sigma** (IPOP-CMA-ES style)
#4. **Coordinate descent** as final polishing phase
#5. **Better stagnation handling** in SHADE
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def ev(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
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
        if elapsed() >= max_time * 0.06:
            break
        fitness[i] = ev(population[i])
    
    # Opposition-based initialization
    for i in range(pop_size):
        if elapsed() >= max_time * 0.10:
            break
        opp = lower + upper - population[i]
        of = ev(opp)
        if of < fitness[i]:
            population[i] = clip(opp)
            fitness[i] = of

    # --- Phase 2: L-SHADE ---
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
    
    de_time_limit = max_time * 0.55
    gen = 0

    while elapsed() < de_time_limit:
        gen += 1
        sorted_idx = np.argsort(fitness)
        p_best_size = max(2, int(0.11 * pop_size))
        
        S_F = []
        S_CR = []
        delta_f = []
        
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
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = idxs[np.random.randint(len(idxs))]
            
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined = [j for j in combined if j != i and j != r1]
            if not combined:
                combined = [j for j in range(pop_size) if j != i]
            r2v = combined[np.random.randint(len(combined))]
            x_r2 = population[r2v] if r2v < pop_size else archive[r2v - pop_size]
            
            mutant = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - x_r2)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)
            
            trial_f = ev(trial)
            
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
        
        if stagnation > 15:
            n_replace = max(1, pop_size // 3)
            si = np.argsort(fitness)
            for ii in range(n_replace):
                idx = si[-(ii + 1)]
                if np.random.random() < 0.6:
                    population[idx] = best_params + 0.15 * ranges * np.random.randn(dim)
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip(population[idx])
                if elapsed() >= de_time_limit:
                    break
                fitness[idx] = ev(population[idx])
            stagnation = 0

    # --- Phase 3: CMA-ES with restarts ---
    cma_time_limit = max_time * 0.90
    cma_run = 0
    
    while best_params is not None and elapsed() < cma_time_limit:
        cma_run += 1
        n = dim
        
        if cma_run == 1:
            sigma0 = 0.05
            mean = best_params.copy()
        elif cma_run == 2:
            sigma0 = 0.1
            mean = best_params.copy()
        else:
            sigma0 = 0.02
            mean = best_params.copy()
        
        sigma = sigma0
        lam = max(4 + int(3 * np.log(n)), 10)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        eigeneval = 0
        cma_gen = 0
        no_improve = 0
        cma_prev_best = best
        
        while elapsed() < cma_time_limit:
            cma_gen += 1
            
            if cma_gen % (max(1, n // 2)) == 0 or cma_gen == 1:
                try:
                    C = (C + C.T) / 2
                    D, B = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-20)
                    sqrtC = B @ np.diag(np.sqrt(D)) @ B.T
                    invsqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                except:
                    C = np.eye(n)
                    sqrtC = np.eye(n)
                    invsqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            
            for j in range(lam):
                if elapsed() >= cma_time_limit:
                    break
                arx[j] = clip(mean + sigma * (sqrtC @ arz[j]))
                arf[j] = ev(arx[j])
            
            if elapsed() >= cma_time_limit:
                break
            
            idx_sort = np.argsort(arf)
            
            old_mean = mean.copy()
            mean = np.zeros(n)
            for j in range(mu):
                mean += weights[j] * arx[idx_sort[j]]
            mean = clip(mean)
            
            diff = mean - old_mean
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff / (sigma + 1e-30)
            
            ps_norm = np.linalg.norm(ps)
            hsig = 1.0 if ps_norm / np.sqrt(1 - (1 - cs)**(2 * cma_gen)) < (1.4 + 2 / (n + 1)) * chiN else 0.0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / (sigma + 1e-30)
            
            artmp = np.zeros((mu, n))
            for j in range(mu):
                artmp[j] = (arx[idx_sort[j]] - old_mean) / (sigma + 1e-30)
            
            C = (1 - c1 - cmu_v) * C
            C += c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for j in range(mu):
                C += cmu_v * weights[j] * np.outer(artmp[j], artmp[j])
            
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = max(1e-16, min(sigma, max(ranges)))
            
            if abs(cma_prev_best - best) < 1e-15:
                no_improve += 1
            else:
                no_improve = 0
            cma_prev_best = best
            
            if sigma < 1e-13 or no_improve > 30 + 10 * n:
                break
            
            cond = np.max(D) / (np.min(D) + 1e-30) if 'D' in dir() else 1
            if cond > 1e14:
                break

    # --- Phase 4: Coordinate descent polishing ---
    if best_params is not None and elapsed() < max_time * 0.95:
        x = best_params.copy()
        fx = best
        step_sizes = 0.001 * ranges
        
        for _ in range(5):
            if elapsed() >= max_time * 0.95:
                break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = clip(np.array([x[d] + sign * step_sizes[d]]))[0]
                    fn = ev(xn)
                    if fn < fx:
                        # Line search along this direction
                        while elapsed() < max_time * 0.95:
                            step_sizes[d] *= 1.5
                            xnn = x.copy()
                            xnn[d] = clip(np.array([xn[d] + sign * step_sizes[d]]))[0]
                            fnn = ev(xnn)
                            if fnn < fn:
                                xn = xnn; fn = fnn
                            else:
                                break
                        x = xn; fx = fn; improved = True
                        break
            if not improved:
                step_sizes *= 0.5

    # --- Phase 5: Final Nelder-Mead ---
    if best_params is not None and elapsed() < max_time * 0.99:
        x0 = best_params.copy()
        n = dim
        scale = 0.005 * ranges
        simplex = np.vstack([x0, x0 + np.diag(scale)])
        fs = []
        for j in range(n + 1):
            if elapsed() >= max_time * 0.99:
                return best
            fs.append(ev(simplex[j]))
        fs = np.array(fs)
        
        while elapsed() < max_time * 0.99:
            o = np.argsort(fs)
            simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c + (c - simplex[-1])); fr = ev(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr
            elif fr < fs[0]:
                xe = clip(c + 2 * (xr - c)); fe = ev(xe)
                if fe < fr: simplex[-1] = xe; fs[-1] = fe
                else
