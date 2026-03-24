#Looking at the results, algorithm 2 performed best (11.43) with L-SHADE + CMA-ES + coordinate descent. I'll analyze what worked and improve:
#
#1. **Better CMA-ES with restarts and proper sigma adaptation**
#2. **Improved L-SHADE with better time allocation**
#3. **Seperable CMA-ES variant for high dimensions**
#4. **Multi-start local search with pattern search (Hooke-Jeeves)**
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
    memory_size = 10
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    mem_idx = 0
    archive = []
    max_archive = pop_size_init
    
    total_budget_shade = 0.40
    
    while elapsed() < max_time * total_budget_shade:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.25 * pop_size))
        
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
            pbest_idx = np.argsort(fitness)[:p]
            pbest = population[np.random.choice(pbest_idx)]
            
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

    # Collect diverse restart points
    top_k = min(5, pop_size)
    restart_points = population[np.argsort(fitness)[:top_k]].copy()

    # --- Phase 3: CMA-ES with IPOP restarts ---
    def run_cmaes(x0, sigma0, deadline):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        counteval = 0
        best_f_local = float('inf')
        stag = 0
        
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        
        while elapsed() < deadline:
            if counteval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
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
            
            arfitness = np.zeros(lam)
            for k in range(lam):
                if elapsed() >= deadline:
                    return
                arfitness[k] = eval_f(arx[k])
            counteval += lam
            
            idx = np.argsort(arfitness)
            old_mean = mean.copy()
            mean = np.dot(w, arx[idx[:mu]])
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            
            try:
                invsqrtC = B @ np.diag(1.0 / (D + 1e-20)) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            except:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff
            
            ps_norm = np.linalg.norm(ps)
            hsig = float(ps_norm / np.sqrt(1 - (1 - cs)**(2*counteval/lam + 2)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - old_mean) / max(sigma, 1e-30)
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(w) @ artmp)
            
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges) * 2)
            
            cur_best = arfitness[idx[0]]
            if cur_best < best_f_local - 1e-10:
                best_f_local = cur_best
                stag = 0
            else:
                stag += 1
            
            if sigma < 1e-15 or stag > 30 + 10 * n:
                break

    # Run CMA-ES from best and restart points
    cma_deadline = max_time * 0.78
    n_runs = min(4, top_k + 1)
    
    # First run from global best
    if elapsed() < cma_deadline and best_params is not None:
        time_per = (cma_deadline - elapsed()) / n_runs
        run_cmaes(best_params.copy(), 0.15 * np.mean(ranges), elapsed() + time_per)
    
    # Additional runs from diverse points with increasing population (IPOP-like)
    for r in range(1, n_runs):
        if elapsed() >= cma_deadline:
            break
        time_per = (cma_deadline - elapsed()) / (n_runs - r)
        x0 = restart_points[r % len(restart_points)] if r < len(restart_points) else best_params.copy()
        sig = 0.08 * np.mean(ranges) * (0.6 ** (r-1))
        run_cmaes(x0, sig, elapsed() + time_per)
    
    # One more from global best with tiny sigma
    if elapsed() < cma_deadline:
        run_cmaes(best_params.copy(), 0.02 * np.mean(ranges), cma_deadline)

    # --- Phase 4: Hooke-Jeeves pattern search ---
    if best_params is not None:
        step = 0.05 * ranges.copy()
        min_step = 1e-7 * ranges
        
        while elapsed() < max_time * 0.93 and np.any(step > min_step):
            improved_any = False
            base = best_params.copy()
            base_val = best
            
            for d in range(dim):
                if elapsed() >= max_time * 0.93:
                    break
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f_trial = eval_f(trial)
                    if f_trial < best:
                        improved_any = True
                        # Accelerate
                        factor = 2.0
                        for _ in range(10):
                            trial2 = best_params.copy()
                            trial2[d] = np.clip(trial2[d] + direction * step[d] * factor, lower[d], upper[d])
                            f2 = eval_f(trial2)
                            if f2 >= best:
                                break
                            factor *= 2.0
                        break
            
            if not improved_any:
                step *= 0.5
            
            # Pattern move
            if improved_any and elapsed() < max_time * 0.93:
                pattern = best_params - base
                if np.linalg.norm(pattern) > 1e-20:
                    for scale in [2.0, 1.5, 1.0]:
                        trial = clip(best_params + scale
