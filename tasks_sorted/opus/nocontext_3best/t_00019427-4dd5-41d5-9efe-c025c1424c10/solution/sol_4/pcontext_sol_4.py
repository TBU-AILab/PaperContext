#Looking at the results, algorithm 1 scored best (4.25). I'll build on its strengths while adding key improvements:
#
#1. **Better SHADE with L-SHADE population reduction** throughout
#2. **Smarter local search scheduling** - adaptive time allocation
#3. **CMA-ES-inspired local search** alongside Nelder-Mead
#4. **Improved pattern search** with adaptive step sizes
#5. **Better diversity management** with opposition-based learning
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
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS Initialization ---
    pop_size = min(max(30, 8 * dim), 200)
    initial_pop_size = pop_size
    min_pop_size = max(5, dim)
    
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

    # Opposition-based learning on initial population
    opp_population = lower + upper - population
    for i in range(pop_size):
        if elapsed() >= max_time * 0.10:
            break
        f_opp = evaluate(opp_population[i])
        if f_opp < fitness_vals[i]:
            population[i] = clip(opp_population[i])
            fitness_vals[i] = f_opp

    top_solutions = []

    # --- Phase 2: L-SHADE ---
    memory_size = 30
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    memory_idx = 0
    archive = []
    max_archive = pop_size
    
    stagnation_count = 0
    prev_best = best
    gen = 0
    
    de_time_budget = 0.50
    
    while elapsed() < max_time * de_time_budget:
        gen += 1
        S_F = []
        S_CR = []
        S_delta = []
        
        p_min = max(2, int(0.05 * pop_size))
        p_max = max(2, int(0.20 * pop_size))
        
        trial_pop = np.copy(population)
        trial_fitness = np.copy(fitness_vals)
        
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
                trial_pop[i] = trial
                trial_fitness[i] = f_trial
        
        population = trial_pop
        fitness_vals = trial_fitness
        
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
        
        # LPSR
        if gen % 3 == 0 and pop_size > min_pop_size:
            ratio = elapsed() / (max_time * de_time_budget)
            new_size = max(min_pop_size, int(round(initial_pop_size + (min_pop_size - initial_pop_size) * ratio)))
            if new_size < pop_size:
                sorted_idx = np.argsort(fitness_vals[:pop_size])
                population = population[sorted_idx[:new_size]]
                fitness_vals = fitness_vals[sorted_idx[:new_size]]
                pop_size = new_size
                max_archive = new_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))
        
        if stagnation_count > 8:
            sorted_idx = np.argsort(fitness_vals[:pop_size])
            half = pop_size // 2
            for j in sorted_idx[half:]:
                if elapsed() >= max_time * de_time_budget:
                    break
                if np.random.random() < 0.5:
                    scale = 0.12 * ranges * np.random.random()
                    population[j] = best_params + np.random.randn(dim) * scale
                else:
                    population[j] = lower + np.random.random(dim) * ranges
                population[j] = clip(population[j])
                fitness_vals[j] = evaluate(population[j])
            stagnation_count = 0

    # Collect diverse top solutions
    sorted_idx = np.argsort(fitness_vals[:pop_size])
    for idx in sorted_idx[:min(8, pop_size)]:
        top_solutions.append((population[idx].copy(), fitness_vals[idx]))

    # --- Phase 3: Simplified CMA-ES local search ---
    def cma_local(start_point, sigma0, time_limit):
        nonlocal best, best_params
        n = dim
        mean = start_point.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        lam = max(4 + int(3 * np.log(n)), 8)
        mu = lam // 2
        weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_cma = weights_raw / weights_raw.sum()
        mueff = 1.0 / np.sum(weights_cma**2)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        for iteration in range(500):
            if elapsed() >= time_limit:
                return
            
            try:
                sqrtC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                sqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fits = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= time_limit:
                    return
                arx[k] = mean + sigma * (sqrtC @ arz[k])
                arx[k] = clip(arx[k])
                fits[k] = evaluate(arx[k])
            
            idx_sort = np.argsort(fits)
            
            old_mean = mean.copy()
            mean = np.zeros(n)
            zmean = np.zeros(n)
            for k in range(mu):
                mean += weights_cma[k] * arx[idx_sort[k]]
                zmean += weights_cma[k] * arz[idx_sort[k]]
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) / chiN < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (1 / sigma) * np.array([arx[idx_sort[k]] - old_mean for k in range(mu)])
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for k in range(mu):
                C += cmu_val * weights_cma[k] * np.outer(artmp[k], artmp[k])
            
            # Ensure symmetry
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-12 * np.eye(n)
            
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))
            
            if sigma < 1e-14 * np.max(ranges):
                break
    
    # Run CMA-ES if dim is manageable
    if dim <= 50:
        cma_local(best_params, 0.1 * np.mean(ranges), max_time * 0.68)
        cma_local(best_params, 0.02 * np.mean(ranges), max_time * 0.78)
    
    # --- Phase 4: Multi-scale Nelder-Mead ---
    def nelder_mead(start_point, start_fit, scale_factor, time_limit_abs):
        nonlocal best, best_params
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        f_simplex = np.zeros(n + 1)
        simplex[0] = start_point.copy()
        f_simplex[0] = start_fit
        
        for i in range(n):
            if elapsed() >= time_limit_abs:
                return
            simplex[i+1] = start_point.copy()
            simplex[i+1][i] += scale[i] if abs(scale[i]) > 1e-15 else 0.01
            simplex[i+1] = clip(simplex[i+1])
            f_simplex[i+1] = evaluate(simplex[i+1])
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        no_improve = 0
        
        for _ in range(20000):
            if elapsed() >= time_limit_abs:
                return
            order = np.argsort(f_simplex)
            simplex, f_simplex = simplex[order], f_simplex[order]
            
            old_best_f = f_simplex[0]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + alpha*(centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma*(xr - centroid))
                fe = evaluate(xe)
                simplex[-1], f_simplex[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                candidate = xr if fr < f_simplex[-1] else simplex[-1]
                xc = clip(centroid + rho*(candidate - centroid))
                fc = evaluate(xc)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for j in range(1, n+1):
                        simplex[j] = clip(simplex[0] + sigma*(simplex[j]-simplex[
