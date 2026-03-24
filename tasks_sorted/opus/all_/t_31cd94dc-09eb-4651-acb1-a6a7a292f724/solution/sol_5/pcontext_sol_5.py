#Looking at the results, algorithm 1 performed best (13.93) with L-SHADE + CMA-ES + coordinate descent. I'll improve upon it with:
#
#1. **Better CMA-ES** with restarts from multiple promising points
#2. **Improved L-SHADE** with better parameter settings
#3. **More sophisticated local search** with adaptive step sizes and pattern search
#4. **Better time budget allocation**
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
    
    eval_count = [0]
    
    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
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
        if elapsed() >= max_time * 0.08:
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
                if elapsed() >= max_time * 0.10:
                    break

    # Store top solutions for later restarts
    top_k = min(5, pop_size)
    top_solutions = population[np.argsort(fitness)[:top_k]].copy()

    # --- Phase 2: L-SHADE ---
    memory_size = 10
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    mem_idx = 0
    archive = []
    max_archive = pop_size_init
    
    gen_count = 0
    total_budget_shade = 0.45
    
    while elapsed() < max_time * total_budget_shade:
        gen_count += 1
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

    # Update top solutions
    order = np.argsort(fitness)
    for i in range(min(top_k, pop_size)):
        if fitness[order[i]] < eval_f(top_solutions[i % top_k]):
            top_solutions[i % top_k] = population[order[i]].copy()

    # --- Phase 3: CMA-ES with restarts from best points ---
    def run_cmaes(x0, sigma0, deadline):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
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
        stag_count = 0
        
        while elapsed() < deadline:
            try:
                D_sq, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_sq, 1e-20))
            except:
                C = np.eye(n)
                D = np.ones(n)
                B = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.array([eval_f(arx[k]) for k in range(lam)])
            if elapsed() >= deadline:
                break
            counteval += lam
            
            idx = np.argsort(arfitness)
            old_mean = mean.copy()
            mean = np.dot(weights, arx[idx[:mu]])
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            try:
                invsqrtC = B @ np.diag(1.0 / (D + 1e-20)) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            except:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff
            
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*counteval/lam + 2)) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - old_mean) / (sigma + 1e-30)
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges))
            
            cur_best = arfitness[idx[0]]
            if cur_best < best_f_local - 1e-10:
                best_f_local = cur_best
                stag_count = 0
            else:
                stag_count += 1
            
            if sigma < 1e-14 or stag_count > 20 + 10 * n:
                break

    # Run CMA-ES from best solution and top solutions
    cma_budget = 0.75
    n_cma_runs = min(3, top_k)
    time_per_run = (max_time * cma_budget - elapsed()) / max(n_cma_runs, 1)
    
    if elapsed() < max_time * cma_budget:
        # First run from global best with medium sigma
        run_cmaes(best_params.copy(), 0.1 * np.mean(ranges), min(elapsed() + time_per_run, max_time * cma_budget))
    
    for r in range(1, n_cma_runs):
        if elapsed() >= max_time * cma_budget:
            break
        sig = 0.05 * np.mean(ranges) * (0.5 ** r)
        run_cmaes(best_params.copy(), sig, min(elapsed() + time_per_run, max_time * cma_budget))

    # --- Phase 4: Adaptive coordinate descent with pattern search ---
    if best_params is not None:
        step = 0.1 * ranges.copy()
        min_step = 1e-6 * ranges
        
        while elapsed() < max_time * 0.94 and np.any(step > min_step):
            improved_any = False
            for d in range(dim):
                if elapsed() >= max_time * 0.94:
                    break
                current_val = best
                for direction in [1, -1]:
                    trial = best_params.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f_trial = eval_f(trial)
                    if f_trial < current_val:
                        improved_any = True
                        # Accelerate in this direction
                        for acc in range(8):
                            trial2 = best_params.copy()
                            trial2[d] = np.clip(trial2[d] + direction * step[d] * (2 ** acc), lower[d], upper[d])
                            f2 = eval_f(trial2)
                            if f2 >= best:
                                break
                        break
            if not improved_any:
                step *= 0.5

    # --- Phase 5: Fine-grained random perturbation ---
    if best_params is not None:
        scales = [0.001, 0.0003, 0.0001]
        si = 0
        while elapsed() < max_time * 0.998:
            s = scales[si % len(scales)]
            trial = best_params + s * ranges * np.random.randn(dim)
            eval_f(trial)
            si += 1
    
    return best
