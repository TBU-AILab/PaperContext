#Looking at the previous algorithm, I see it's already quite sophisticated with SHADE-DE, CMA-ES, and Nelder-Mead. Let me analyze potential weaknesses:
#
#1. **Time budget allocation might be suboptimal** - too much time on DE, not enough on CMA-ES
#2. **CMA-ES implementation could have numerical issues** 
#3. **Population sizing and restart strategies could be improved**
#4. **Missing opposition-based learning and other diversity mechanisms**
#5. **The local search could use Powell's method or pattern search for better convergence**
#
#Let me redesign with a tighter, more robust approach:
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
        return max_time * 0.99 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # ---- Adaptive pattern search for local refinement ----
    def pattern_search(x0, step, max_ps_time, min_step=1e-15):
        nonlocal best, best_params
        ps_start = elapsed()
        x = x0.copy()
        fx = evaluate(x)
        directions = np.eye(dim)
        
        while step > min_step and (elapsed() - ps_start) < max_ps_time and remaining() > 0:
            improved = False
            for i in range(dim):
                if remaining() <= 0:
                    return x, fx
                for sign in [1.0, -1.0]:
                    trial = x.copy()
                    trial += sign * step * directions[i] * ranges
                    trial = clip(trial)
                    ft = evaluate(trial)
                    if ft < fx:
                        fx = ft
                        x = trial
                        improved = True
                        break
            if not improved:
                step *= 0.5
        return x, fx

    # ---- CMA-ES with restarts ----
    def cmaes_search(x0, sigma0, max_cma_time):
        nonlocal best, best_params
        cma_start = elapsed()
        n = dim
        mean = x0.copy()
        sigma = sigma0
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
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        eigen_update = 0
        
        gen = 0
        best_cma = float('inf')
        stag = 0
        
        while (elapsed() - cma_start) < max_cma_time and remaining() > 0:
            gen += 1
            
            if gen > 1 and eigen_update >= max(1, int(1.0/(c1 + cmu_val)/n/10)):
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-30)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                    eigen_update = 0
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
                    sigma *= 0.5
            eigen_update += 1
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
            
            fit = np.zeros(lam)
            for k in range(lam):
                if remaining() <= 0:
                    return
                fit[k] = evaluate(arx[k])
            
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            zmean = np.sum(weights[:, None] * arz[idx[:mu]], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*gen)) / chiN < 1.4 + 2.0/(n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_val * (weights[:, None] * artmp).T @ artmp
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-20)
            sigma = min(sigma, np.max(ranges) * 2)
            
            if fit[0] < best_cma - 1e-12:
                best_cma = fit[0]
                stag = 0
            else:
                stag += 1
            
            if sigma < 1e-16 or stag > 100 + 30*n:
                break
            if np.max(D) > 1e7 * np.min(D):
                break

    # ---- SHADE-ILS DE ----
    def shade_de(population, fitness, max_de_time):
        nonlocal best, best_params
        de_start = elapsed()
        pop_size = len(population)
        H = 100
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.9)
        hist_pos = 0
        archive = []
        archive_max = pop_size
        stag = 0
        prev_b = best
        
        while (elapsed() - de_start) < max_de_time and remaining() > 0:
            SF, SCR, S_delta = [], [], []
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if remaining() <= 0 or (elapsed() - de_start) >= max_de_time:
                    idx2 = np.argsort(new_fit)
                    return new_pop[idx2], new_fit[idx2]
                
                ri = np.random.randint(0, H)
                while True:
                    Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
                    if Fi > 0: break
                Fi = min(Fi, 1.0)
                CRi = np.clip(np.random.normal(MCR[ri], 0.1), 0.0, 1.0)
                
                p = max(2, int(np.random.uniform(0.05, 0.20) * pop_size))
                pbest_idx = np.random.randint(0, p)
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                combined = pop_size + len(archive)
                r2 = np.random.randint(0, combined)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, combined)
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
                
                for d in range(dim):
                    if mutant[d] < lower[d]:
                        mutant[d] = (lower[d] + population[i][d]) / 2
                    elif mutant[d] > upper[d]:
                        mutant[d] = (upper[d] + population[i][d]) / 2
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = evaluate(trial)
                if f_trial <= new_fit[i]:
                    if f_trial < new_fit[i]:
                        SF.append(Fi)
                        SCR.append(CRi)
                        S_delta.append(abs(new_fit[i] - f_trial))
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            if SF and sum(S_delta) > 0:
                w = np.array(S_delta); w /= w.sum()
                sf_a = np.array(SF); scr_a = np.array(SCR)
                MF[hist_pos] = np.sum(w * sf_a**2) / (np.sum(w * sf_a) + 1e-30)
                MCR[hist_pos] = np.sum(w * scr_a)
                hist_pos = (hist_pos + 1) % H
            
            population = new_pop
            fitness = new_fit
            idx2 = np.argsort(fitness)
            population = population[idx2]
            fitness = fitness[idx2]
            
            if abs(best - prev_b) < 1e-14:
                stag += 1
            else:
                stag = 0; prev_b = best
            
            if stag > 15 + dim:
                n_replace = pop_size // 2
                for k in range(n_replace):
                    if remaining() <= 0: break
                    ri2 = pop_size - 1 - k
                    if np.random.random() < 0.3:
                        population[ri2] = lower + np.random.random(dim) * ranges
                    else:
                        population[ri2] = best_params + np.random.randn(dim) * ranges * np.random.uniform(0.01, 0.2)
                    population[ri2] = clip(population[ri2])
                    fitness[ri2] = evaluate(population[ri2])
                idx2 = np.argsort(fitness)
                population = population[idx2]
                fitness = fitness[idx2]
                stag = 0
                archive.clear()
        
        idx2 = np.argsort(fitness)
        return population[idx2], fitness[idx2]

    # ==== MAIN FLOW ====
    
    # Phase 1: LHS initial sampling
    pop_size = min(200, max(30, 15 * dim))
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    # Add opposition-based points
    opp = lower + upper - pop
    pop = np.vstack([pop, opp])
    
    fit = np.array([evaluate(pop[i]) for i in range(len(pop)) if remaining() > 0])
    if len(fit) < len(pop):
        pop = pop[:len(fit)]
    
    idx = np.argsort(fit)
    pop = pop[idx[:pop_size]]
    fit = fit[idx[:pop_size]]
    
    # Phase 2: SHADE DE
    if remaining() > max_time * 0.5:
        de_time = remaining() - max_time * 0.5
        pop, fit = shade_de(pop, fit, de_time)
    
    # Phase 3: CMA-ES restarts
    restart_count = 0
    while remaining() > max_time * 0.12:
        restart_count += 1
        cma_time = min(remaining() - max_time * 0.10, remaining() * 0.5)
        if cma_time < 0.5: break
        
        if restart_count == 1:
            x0 = best_params.copy()
            s0 = np.max(ranges) * 0.15
        elif restart_count <= 3:
            x0 = best_params + np.random.randn(dim) * ranges * 0.05
            x0 = clip(x0)
            s0 = np.max(ranges) * 0.05
        else:
            x0 = lower + np.random.random(dim) * ranges
            s0 = np.max(ranges) * 0.3
        
        cmaes_search(x0, s0, cma_time)
    
    # Phase 4: Pattern search polish
    if best_params is not None and remaining() > 0.5:
        pattern_search(best_params.copy(), 0.01, remaining() * 0.5)
    
    if best_params is not None and remaining() > 0.2:
        pattern_search(best_params.copy(), 0.001, remaining())
    
    return best
#
#Key improvements:
#1. **Opposition-based learning** in initialization doubles coverage
#2. **Pattern search** for local refinement (more robust than Nelder-Mead in high dimensions)
#3. **Better time allocation**: less DE, more CMA-ES restarts
#4. **Multiple CMA-ES restarts** with varying sigma and starting points
#5. **Improved SHADE parameter adaptation** (higher initial CR)
#6. **Better bounce-back** using midpoint reflection
#7. **Cleaner eigendecomposition scheduling** in CMA-ES
