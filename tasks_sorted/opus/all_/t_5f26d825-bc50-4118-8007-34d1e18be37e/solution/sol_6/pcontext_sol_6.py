#Looking at the results, Algorithm 2 performed best (6.62) with its L-SHADE + CMA-ES + Nelder-Mead + coordinate descent pipeline. Let me analyze and improve:
#
#1. The multi-phase approach works well
#2. CMA-ES restarts help escape local optima
#3. More aggressive exploration early on could help
#4. Better integration between phases and smarter time allocation
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Stratified sampling + opposition ---
    n_init = min(max(80, 30 * dim), 1200)
    population = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        population[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    opp_pop = lower + upper - population
    all_cands = np.vstack([population, opp_pop])
    np.random.shuffle(all_cands)
    
    fit_list = []
    for i in range(len(all_cands)):
        if elapsed() >= max_time * 0.06:
            break
        fit_list.append(evaluate(all_cands[i]))
    
    if len(fit_list) == 0:
        x0 = np.array([(l+u)/2 for l,u in bounds])
        return evaluate(x0)
    
    eval_cands = all_cands[:len(fit_list)]
    eval_fits = np.array(fit_list)
    
    pop_size = min(max(20, 6 * dim), 120)
    sorted_idx = np.argsort(eval_fits)[:pop_size]
    population = eval_cands[sorted_idx].copy()
    fitness = eval_fits[sorted_idx].copy()
    
    # Store top solutions for later restarts
    top_k = min(10, len(eval_fits))
    top_solutions = eval_cands[np.argsort(eval_fits)[:top_k]].copy()
    
    # --- Phase 2: L-SHADE ---
    memory_size = 50
    memory_F = np.full(memory_size, 0.5)
    memory_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size * 3
    
    min_pop_size = max(4, dim // 3)
    initial_pop_size = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    de_end_time = 0.48
    
    while elapsed() < max_time * de_end_time:
        generation += 1
        progress = min(1.0, elapsed() / (max_time * de_end_time))
        
        target_pop = max(min_pop_size, int(initial_pop_size - (initial_pop_size - min_pop_size) * progress))
        
        success_F = []
        success_CR = []
        success_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        cur_pop = len(population)
        
        p_min = max(2, int(0.05 * cur_pop))
        p_max = max(2, int(0.20 * cur_pop))
        p = max(p_min, int(p_max - (p_max - p_min) * progress))
        
        for i in range(cur_pop):
            if elapsed() >= max_time * de_end_time:
                break
            
            ri = np.random.randint(memory_size)
            F_i = -1
            attempts = 0
            while F_i <= 0 and attempts < 10:
                F_i = np.random.standard_cauchy() * 0.1 + memory_F[ri]
                attempts += 1
            F_i = np.clip(F_i, 0.1, 1.0)
            
            CR_i = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            if progress > 0.6:
                CR_i *= (1.0 - 0.4 * (progress - 0.6) / 0.4)
            
            idxs = list(range(cur_pop))
            idxs.remove(i)
            
            p_best_idx = np.random.randint(p)
            r1 = np.random.choice(idxs)
            idxs2 = [j for j in idxs if j != r1]
            
            use_archive = len(archive) > 0 and np.random.random() < 0.5 and len(idxs2) > 0
            if use_archive:
                arc_idx = np.random.randint(len(archive))
                diff2 = population[r1] - archive[arc_idx]
            else:
                if len(idxs2) > 0:
                    r2 = np.random.choice(idxs2)
                    diff2 = population[r1] - population[r2]
                else:
                    diff2 = np.random.randn(dim) * 0.001 * ranges
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * diff2
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR_i
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = clip(trial)
            
            f_trial = evaluate(trial)
            
            if f_trial < fitness[i]:
                delta = fitness[i] - f_trial
                success_F.append(F_i)
                success_CR.append(CR_i)
                success_delta.append(delta)
                
                if len(archive) < archive_max:
                    archive.append(population[i].copy())
                elif len(archive) > 0:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                
                new_population[i] = trial
                new_fitness[i] = f_trial
        
        population = new_population
        fitness = new_fitness
        
        if len(success_F) > 0:
            weights = np.array(success_delta)
            w_sum = np.sum(weights)
            if w_sum > 0:
                weights = weights / w_sum
                sf = np.array(success_F)
                scr = np.array(success_CR)
                denom = np.sum(weights * sf)
                if denom > 1e-30:
                    memory_F[mem_idx] = np.sum(weights * sf**2) / denom
                memory_CR[mem_idx] = np.sum(weights * scr)
                mem_idx = (mem_idx + 1) % memory_size
        
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if len(population) > target_pop:
            population = population[:target_pop]
            fitness = fitness[:target_pop]
        
        if best >= prev_best - 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 15:
            n_replace = max(1, len(population) // 3)
            for i in range(len(population) - n_replace, len(population)):
                if np.random.random() < 0.2:
                    population[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
                else:
                    sigma = 0.03 * ranges * max(0.001, 1.0 - progress)
                    population[i] = best_x + sigma * np.random.randn(dim)
                population[i] = clip(population[i])
                fitness[i] = evaluate(population[i])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: CMA-ES restarts ---
    def run_cmaes(x0, sigma0, end_time_frac):
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_p = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        D = np.ones(n)
        B = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        counteval = 0
        best_local = float('inf')
        no_improve = 0
        
        while elapsed() < max_time * end_time_frac:
            if counteval - eigeneval > lam / (c1 + cmu_p + 1e-30) / n / 10:
                eigeneval = counteval
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
            
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= max_time * end_time_frac:
                    return
                z = np.random.randn(n)
                arx[k] = mean + sigma * (B @ (D * z))
                arx[k] = clip(arx[k])
                arfitness[k] = evaluate(arx[k])
                counteval += 1
            
            if elapsed() >= max_time * end_time_frac:
                return
            
            arindex = np.argsort(arfitness)
            selected = arx[arindex[:mu]]
            old_mean = mean.copy()
            mean = selected.T @ w
            
            diff = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu_p) * C + c1 * (np.outer(pc, pc) + (1-hsig) * cc*(2-cc) * C) + cmu_p * (artmp.T @ np.diag(w) @ artmp)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-15 * np.max(ranges), 0.5 * np.max(ranges))
            
            gen_best = arfitness[arindex[0]]
            if gen_best < best_local - 1e-12:
                best_local = gen_best
                no_improve = 0
            else:
                no_improve += 1
            
            if sigma * np.max(D) < 1e-13 * np.max(ranges):
                break
            if no_improve > 50 + 10 * n:
                break
    
    # CMA-ES run 1: from best with medium sigma
    if best_x is not None and elapsed() < max_time * 0.65:
        run_cmaes(best_x.copy(), 0.05 * np.max(ranges), 0.63)
    
    # CMA-ES run 2: from best with smaller sigma
    if best_x is not None and elapsed() < max_time * 0.76:
        run_cmaes(best_x.copy(), 0.015 * np.max(ranges), 0.74)
    
    # CMA-ES run 3: from a different top solution
    if elapsed() < max_time * 0.82:
        if len(top_solutions) > 1:
            alt_x = top_solutions[np.random.randint(1, len(top_solutions))]
            run_cmaes(alt_x.copy(), 0.03 * np.max(ranges), 0.81)
        elif best_x is not None:
            perturbed = best_x + 0.1 * ranges * np.random.randn(dim)
            run_cmaes(clip(perturbed), 0.03 * np.max(ranges), 0.81)
    
    # CMA-ES run 4: tiny sigma polish
    if best_x is not None and elapsed() < max_time * 0.88:
        run_cmaes(best_x.copy(), 0.005 * np.max(ranges), 0.87)
    
    # --- Phase 4: Nelder-Mead ---
    if best_x is not None and elapsed() < max_time * 0.93:
        n = dim
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        init_scale = 0.003 * ranges
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += init_scale[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.array([evaluate(simplex[i]) if elapsed() < max_time * 0.93 else float('inf') for i in range(n + 1)])
        
        for iteration in range(500):
            if elapsed() >= max_time * 0.93:
                break
            
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + alpha_nm * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma_nm * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = clip(centroid + rho_nm * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if elapsed() >= max_time * 0.93: break
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
                else:
                    xc = clip(centroid - rho_nm * (centroid - simplex[-1]))
                    fc = evaluate(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if elapsed() >= max_time * 0.93: break
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = evaluate(simplex[i])
            
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15 * np.max(ranges):
                break
    
    # --- Phase 5: Multi-scale coordinate descent ---
    if best_x is not None:
        for scale in [0.05, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
            if elapsed() >= max_time * 0.97:
                break
            step = scale * ranges
            for _ in range(4):
                if elapsed() >= max_time * 0.97:
                    break
                improved = False
                for j in range(dim):
                    if elapsed() >= max_time * 0.97:
                        break
                    current_best = best
                    for direction in [1, -1]:
                        trial = best_x.copy()
                        trial[j] += direction * step[j]
                        f_trial = evaluate(trial)
                        if f_trial < current_best:
                            improved = True
                            step[j] *= 1.5
                            break
                    else:
                        step[j] *= 0.5
                if not improved:
                    break
    
    # --- Phase 6: Golden section on each coordinate ---
    if best_x is not None and elapsed() < max_time * 0.993:
        gr = (np.sqrt(5) - 1) / 2
        for j in range(dim):
            if elapsed() >= max_time * 0.993:
                break
            radius = 0.0005 * ranges[j]
            a = max(lower[j], best_x[j] - radius)
            b = min(upper[j], best_x[j] + radius)
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            trial_c = best_x.copy(); trial_c[j] = c
            trial_d = best_x.copy(); trial_d[j] = d
            fc = evaluate(trial_c)
            fd = evaluate(trial_d)
            for _ in range(12):
                if elapsed() >= max_time * 0.993:
                    break
                if fc < fd:
                    b = d; d = c; fd = fc
                    c = b - gr * (b - a)
                    trial_c = best_x.copy(); trial_c[j] = c
                    fc = evaluate(trial_c)
                else:
                    a = c; c = d; fc = fd
                    d = a + gr * (b - a)
                    trial_d = best_x.copy(); trial_d[j] = d
                    fd = evaluate(trial_d)
    
    # --- Phase 7: Final perturbations ---
    while elapsed() < max_time * 0.998 and best_x is not None:
        sigma = 0.00005 * ranges * np.random.random()
        evaluate(best_x + sigma * np.random.randn(dim))
    
    return best
