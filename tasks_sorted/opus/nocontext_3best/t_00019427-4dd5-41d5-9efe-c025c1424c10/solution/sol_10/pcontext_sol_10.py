#Looking at the results, algorithm 1 (SHADE + multi-scale Nelder-Mead) performed best at 4.26. I'll build upon it with key improvements:
#
#1. **CMA-ES as the main local search** instead of Nelder-Mead - much better for correlated landscapes
#2. **Better SHADE with improved parameter adaptation**
#3. **Smarter time allocation and restart strategy**
#4. **Opposition-based learning for diversity**
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
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size = min(max(30, 8 * dim), 200)
    
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.uniform(0, 1, pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness_vals = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness_vals[i] = evaluate(population[i])

    # Opposition-based population
    opp_pop = lower + upper - population
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        f_opp = evaluate(opp_pop[i])
        if f_opp < fitness_vals[i]:
            population[i] = opp_pop[i].copy()
            fitness_vals[i] = f_opp

    top_solutions = []

    # --- Phase 2: SHADE ---
    memory_size = 30
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    memory_idx = 0
    archive = []
    max_archive = pop_size
    
    stagnation_count = 0
    prev_best = best
    gen = 0
    initial_pop_size = pop_size
    min_pop_size = max(5, dim)
    
    de_time_budget = 0.45
    
    while elapsed() < max_time * de_time_budget:
        gen += 1
        S_F = []
        S_CR = []
        S_delta = []
        
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.20 * pop_size))
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_budget:
                break
            
            r = np.random.randint(memory_size)
            F_i = -1
            while F_i <= 0:
                F_i = np.random.standard_cauchy() * 0.1 + M_F[r]
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            
            p = np.random.randint(p_min, p_max + 1)
            sorted_idx = np.argsort(fitness_vals[:pop_size])
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2_idx = i
            while r2_idx == i or r2_idx == r1:
                r2_idx = np.random.randint(combined_size)
            x_r2 = population[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = population[i] + F_i * (population[pbest_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            cross_points = np.random.random(dim) < CR_i
            cross_points[np.random.randint(dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2.0
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2.0
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness_vals[i]:
                delta = fitness_vals[i] - f_trial
                if delta > 0:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(delta)
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness_vals[i] = f_trial
        
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[memory_idx] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-30)
            M_CR[memory_idx] = np.sum(weights * scr)
            memory_idx = (memory_idx + 1) % memory_size
        
        if abs(prev_best - best) < 1e-14:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        if gen % 5 == 0 and pop_size > min_pop_size:
            new_size = max(min_pop_size, int(initial_pop_size - (initial_pop_size - min_pop_size) * elapsed() / (max_time * de_time_budget)))
            if new_size < pop_size:
                sorted_idx = np.argsort(fitness_vals[:pop_size])
                population = population[sorted_idx[:new_size]]
                fitness_vals = fitness_vals[sorted_idx[:new_size]]
                pop_size = new_size
        
        if stagnation_count > 8:
            sorted_idx = np.argsort(fitness_vals[:pop_size])
            half = pop_size // 2
            for j in sorted_idx[half:]:
                if elapsed() >= max_time * de_time_budget:
                    break
                if np.random.random() < 0.5:
                    scale = 0.15 * ranges * np.random.random()
                    population[j] = best_params + np.random.randn(dim) * scale
                else:
                    population[j] = lower + np.random.random(dim) * ranges
                population[j] = clip(population[j])
                fitness_vals[j] = evaluate(population[j])
            stagnation_count = 0

    sorted_idx = np.argsort(fitness_vals[:pop_size])
    for idx in sorted_idx[:min(8, pop_size)]:
        top_solutions.append((population[idx].copy(), fitness_vals[idx]))

    # --- Phase 3: CMA-ES local search ---
    def run_cmaes(x0, sigma0, time_limit):
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
        cmu_val = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen_count = 0
        while elapsed() < time_limit:
            gen_count += 1
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fit = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= time_limit:
                    return
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
                fit[k] = evaluate(arx[k])
            
            idx_sort = np.argsort(fit)
            arx = arx[idx_sort]
            arz = arz[idx_sort]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu], axis=0)
            
            zmean = np.sum(weights[:, None] * arz[:mu], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen_count+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if gen_count % (1 + int(1/(10*n*(c1+cmu_val+1e-30)))) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
                    invsqrtC = np.eye(n)
            
            if sigma * np.max(D) < 1e-12 * np.max(ranges):
                break
            
            if fit[0] == fit[min(lam-1, mu)]:
                sigma *= np.exp(0.2 + cs/damps)
    
    # Run CMA-ES from best and diverse top solutions
    cma_time_each = (max_time * 0.92 - elapsed()) / max(3, 1)
    
    if best_params is not None:
        run_cmaes(best_params, 0.15 * np.mean(ranges), elapsed() + cma_time_each)
        run_cmaes(best_params, 0.03 * np.mean(ranges), elapsed() + cma_time_each)
    
    for sol, sol_fit in top_solutions:
        if elapsed() >= max_time * 0.92:
            break
        if np.linalg.norm(sol - best_params) > 0.01 * np.linalg.norm(ranges):
            run_cmaes(sol, 0.05 * np.mean(ranges), elapsed() + cma_time_each * 0.5)
    
    # Final refinement with CMA-ES at small sigma
    if elapsed() < max_time * 0.96:
        run_cmaes(best_params, 0.005 * np.mean(ranges), max_time * 0.97)

    # --- Phase 4: Pattern search ---
    step = 0.001 * ranges
    while elapsed() < max_time * 0.99:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.99:
                break
            for sign in [1, -1]:
                trial = best_params.copy()
                trial[d] += sign * step[d]
                f_trial = evaluate(trial)
                if f_trial < best:
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if np.max(step / ranges) < 1e-13:
                break

    while elapsed() < max_time * 0.998:
        s = np.random.exponential(0.001)
        evaluate(best_params + np.random.randn(dim) * s * ranges)
    
    return best
