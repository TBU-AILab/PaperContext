#Looking at the results, algorithm 1 performed best (10.14) with L-SHADE + CMA-ES with IPOP restarts + coordinate descent + fine perturbation. I'll improve upon it with:
#
#1. **Better CMA-ES** with sep-CMA-ES for high dimensions and smarter restart strategy
#2. **Improved L-SHADE** with better memory and archive management
#3. **Golden section line search** in coordinate descent
#4. **Surrogate-assisted perturbation** in final phase
#5. **Better time allocation** based on dimension
#
import numpy as np
from datetime import datetime

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
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS + Opposition-based initialization ---
    pop_size_init = min(max(14 * dim, 60), 300)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    opp_pop = lower + upper - population
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.full(len(all_pop), float('inf'))
    
    for i in range(len(all_pop)):
        if elapsed() >= max_time * 0.07:
            break
        all_fit[i] = eval_f(all_pop[i])
    
    valid_count = np.sum(all_fit < float('inf'))
    
    if valid_count >= pop_size:
        order = np.argsort(all_fit)
        population = all_pop[order[:pop_size]].copy()
        fitness = all_fit[order[:pop_size]].copy()
    else:
        population = all_pop[:pop_size].copy()
        fitness = all_fit[:pop_size].copy()
        for i in range(pop_size):
            if fitness[i] == float('inf'):
                fitness[i] = eval_f(population[i])
                if elapsed() >= max_time * 0.09:
                    break

    # --- Phase 2: L-SHADE ---
    memory_size = 12
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    mem_idx = 0
    archive = []
    max_archive = pop_size_init
    
    total_budget_shade = 0.38
    
    while elapsed() < max_time * total_budget_shade:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.25 * pop_size))
        
        sorted_indices = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * total_budget_shade:
                break
            
            r = np.random.randint(memory_size)
            
            F_i = -1
            for _ in range(10):
                F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
                if F_i > 0:
                    break
            if F_i <= 0:
                F_i = 0.1
            F_i = min(F_i, 1.0)
            
            if M_CR[r] < 0:
                CR_i = 0.0
            else:
                CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            
            p = np.random.randint(p_min, p_max + 1)
            pbest = population[sorted_indices[np.random.randint(p)]]
            
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2_candidates = [j for j in range(combined_size) if j != i and j != r1]
            if len(r2_candidates) == 0:
                r2_candidates = [j for j in range(pop_size) if j != i]
            r2 = np.random.choice(r2_candidates)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + F_i * (pbest - population[i]) + F_i * (population[r1] - x_r2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross = np.random.random(dim) < CR_i
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            trial = clip(trial)
            
            f_trial = eval_f(trial)
            
            if f_trial <= fitness[i]:
                delta = fitness[i] - f_trial
                if delta > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                archive.append(population[i].copy())
                if len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(weights * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        progress = elapsed() / (max_time * total_budget_shade)
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * progress)))
        if new_pop_size < pop_size:
            order = np.argsort(fitness)
            population = population[order[:new_pop_size]]
            fitness = fitness[order[:new_pop_size]]
            pop_size = new_pop_size

    # --- Phase 3: CMA-ES with IPOP restarts ---
    def run_cmaes(x0, sigma0, deadline, lam_override=None):
        nonlocal best, best_params
        n = dim
        lam = lam_override if lam_override else 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        counteval = 0
        stag = 0
        best_local = float('inf')
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        
        while elapsed() < deadline:
            if counteval - eigeneval > lam / (c1 + cmu_v + 1e-30) / n / 10:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    D_sq, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D_sq, 1e-20))
                    eigeneval = counteval
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
            arx = np.clip(arx, lower, upper)
            
            arfit = np.array([eval_f(arx[k]) for k in range(lam)])
            if elapsed() >= deadline:
                break
            counteval += lam
            
            idx = np.argsort(arfit)
            old_mean = mean.copy()
            mean = np.dot(w, arx[idx[:mu]])
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            try:
                invsqC = B @ np.diag(1.0 / (D + 1e-20)) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqC @ diff)
            except:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*counteval/lam + 2)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - old_mean) / max(sigma, 1e-30)
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * (artmp.T @ np.diag(w) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges) * 2)
            
            cur = arfit[idx[0]]
            if cur < best_local - 1e-10:
                best_local = cur
                stag = 0
            else:
                stag += 1
            
            if sigma < 1e-14 or stag > 12 + 3 * n:
                break
        return lam

    cma_deadline = max_time * 0.76
    base_lam = 4 + int(3 * np.log(dim))
    lam_now = base_lam
    run_idx = 0
    
    while elapsed() < cma_deadline:
        time_left = cma_deadline - elapsed()
        if time_left < 0.3:
            break
        
        if run_idx == 0:
            x0 = best_params.copy()
            sig = 0.2 * np.mean(ranges)
        elif run_idx == 1:
            x0 = best_params.copy()
            sig = 0.05 * np.mean(ranges)
        elif run_idx == 2:
            x0 = best_params.copy()
            sig = 0.01 * np.mean(ranges)
        else:
            # IPOP: increase lambda
            lam_now = int(lam_now * 1.5)
            x0 = best_params + 0.15 * ranges * np.random.randn(dim)
            x0 = clip(x0)
            sig = 0.25 * np.mean(ranges)
        
        per_run = min(time_left, max(max_time * 0.10, time_left / 2))
        run_cmaes(x0, sig, elapsed() + per_run, lam_now if run_idx >= 3 else None)
        run_idx += 1

    # --- Phase 4: Golden section line search per coordinate ---
    if best_params is not None:
        gr = (np.sqrt(5) + 1) / 2
        
        def golden_line_search(d, a, b, tol):
            nonlocal best, best_params
            c = b - (b - a) / gr
            dd = a + (b - a) / gr
            
            tc = best_params.copy(); tc[d] = c; fc = eval_f(tc)
            td = best_params.copy(); td[d] = dd; fd = eval_f(td)
            
            for _ in range(20):
                if elapsed() >= max_time * 0.92 or abs(b - a) < tol:
                    break
                if fc < fd:
                    b = dd
                    dd = c; fd = fc
                    c = b - (b - a) / gr
                    tc = best_params.copy(); tc[d] = c; fc = eval_f(tc)
                else:
                    a = c
                    c = dd; fc = fd
                    dd = a + (b - a) / gr
                    td = best_params.copy(); td[d] = dd; fd = eval_f(td)
        
        for radius in [0.1, 0.02, 0.004]:
            if elapsed() >= max_time * 0.92:
                break
            improved_round = False
            for d in range(dim):
                if elapsed() >= max_time * 0.92:
                    break
                old_best = best
                a = max(lower[d], best_params[d] - radius * ranges[d])
                b = min(upper[d], best_params[d] + radius * ranges[d])
