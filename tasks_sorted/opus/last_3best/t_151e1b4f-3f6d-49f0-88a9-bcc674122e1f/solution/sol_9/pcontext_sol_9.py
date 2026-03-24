#Looking at the results, algorithm 1 (SHADE + CMA-ES + Nelder-Mead) achieved the best score of 12.05. I'll build on that approach with improvements:
#
#1. **Better CMA-ES implementation** with proper eigendecomposition updates
#2. **Multi-restart strategy** for escaping local optima
#3. **Improved time allocation** and parameter tuning
#4. **Powell's method** as additional local search
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
            
            idxs = [j for j in range(pop_size) if j != i]
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
        
        if stagnation > 20:
            n_replace = max(1, pop_size // 3)
            si = np.argsort(fitness)
            for ii in range(n_replace):
                idx = si[-(ii + 1)]
                if np.random.random() < 0.5:
                    population[idx] = best_params + 0.15 * ranges * np.random.randn(dim)
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip(population[idx])
                if elapsed() >= de_time_limit:
                    break
                fitness[idx] = ev(population[idx])
            stagnation = 0

    # --- Phase 3: CMA-ES local search with restarts ---
    def run_cmaes(x_start, sigma_init, time_limit):
        n = dim
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
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        mean = x_start.copy()
        sigma = sigma_init
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        count_eval = 0
        
        while elapsed() < time_limit:
            # Update eigen decomposition periodically
            if count_eval - eigeneval > lam / (c1 + cmu_v) / n / 10:
                eigeneval = count_eval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            
            for j in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                arx[j] = clip(mean + sigma * (B @ (D * z)))
                arf[j] = ev(arx[j])
                count_eval += 1
            
            idx_sort = np.argsort(arf)
            old_mean = mean.copy()
            
            mean = np.zeros(n)
            for j in range(mu):
                mean += weights[j] * arx[idx_sort[j]]
            mean = clip(mean)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / (sigma + 1e-30)
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * count_eval / lam)) < (1.4 + 2 / (n + 1)) * chiN else 0.0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / (sigma + 1e-30)
            
            artmp = np.zeros((mu, n))
            for j in range(mu):
                artmp[j] = (arx[idx_sort[j]] - old_mean) / (sigma + 1e-30)
            
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for j in range(mu):
                C += cmu_v * weights[j] * np.outer(artmp[j], artmp[j])
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(1e-15, min(sigma, max(ranges)))
            
            if sigma < 1e-13:
                break
            if np.max(D) > 1e7 * np.min(D):
                break
    
    # Run CMA-ES from best found solution
    if best_params is not None:
        run_cmaes(best_params, 0.1 * np.mean(ranges), max_time * 0.82)
    
    # Second CMA-ES restart with smaller sigma
    if best_params is not None and elapsed() < max_time * 0.90:
        run_cmaes(best_params, 0.02 * np.mean(ranges), max_time * 0.90)

    # --- Phase 4: Powell-like coordinate search ---
    if best_params is not None and elapsed() < max_time * 0.98:
        current_x = best_params.copy()
        current_f = best
        step = ranges * 0.005
        
        while elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                best_delta = 0
                best_fd = current_f
                for sign in [1, -1]:
                    trial = current_x.copy()
                    trial[d] += sign * step[d]
                    trial_f = ev(trial)
                    if trial_f < best_fd:
                        best_fd = trial_f
                        best_delta = sign * step[d]
                if best_delta != 0:
                    current_x[d] += best_delta
                    current_x = clip(current_x)
                    current_f = best_fd
                    improved = True
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-13:
                    break
    
    return best
