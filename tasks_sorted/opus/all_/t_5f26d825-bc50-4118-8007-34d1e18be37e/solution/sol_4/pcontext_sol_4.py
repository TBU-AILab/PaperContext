#Looking at the results, Algorithm 1 performed best (10.74). Let me analyze what made it successful and improve upon it:
#
#1. SHADE with population reduction worked well
#2. Multi-scale local search was effective
#3. Good time management
#
#Key improvements I'll make:
#- Better CMA-ES as a secondary optimizer
#- Improved SHADE with rank-based mutation
#- Nelder-Mead simplex for local search
#- Better restart strategy with diverse initial points
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
    
    evals = [0]
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # --- Phase 1: Sobol-like stratified sampling + opposition ---
    n_init = min(max(50, 20 * dim), 800)
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
        if elapsed() >= max_time * 0.08:
            break
        fit_list.append(evaluate(all_cands[i]))
    
    if len(fit_list) == 0:
        x0 = np.array([(l+u)/2 for l,u in bounds])
        return evaluate(x0)
    
    eval_cands = all_cands[:len(fit_list)]
    eval_fits = np.array(fit_list)
    
    pop_size = min(max(25, 7 * dim), 150)
    sorted_idx = np.argsort(eval_fits)[:pop_size]
    population = eval_cands[sorted_idx].copy()
    fitness = eval_fits[sorted_idx].copy()
    
    # --- Phase 2: L-SHADE (SHADE with Linear Pop Size Reduction) ---
    memory_size = 30
    memory_F = np.full(memory_size, 0.5)
    memory_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []
    archive_max = pop_size * 2
    
    min_pop_size = max(4, dim // 2)
    initial_pop_size = pop_size
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    de_end_time = 0.72
    
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
        
        # Compute rank-based weights for pbest selection
        p_min = max(2, int(0.05 * cur_pop))
        p_max = max(2, int(0.25 * cur_pop))
        p = max(p_min, int(p_max - (p_max - p_min) * progress))
        
        for i in range(cur_pop):
            if elapsed() >= max_time * de_end_time:
                break
            
            ri = np.random.randint(memory_size)
            # Cauchy for F
            F_i = -1
            attempts = 0
            while F_i <= 0 and attempts < 10:
                F_i = np.random.standard_cauchy() * 0.1 + memory_F[ri]
                attempts += 1
            F_i = np.clip(F_i, 0.1, 1.0)
            
            CR_i = np.clip(np.random.normal(memory_CR[ri], 0.1), 0.0, 1.0)
            # For late stages, reduce CR to encourage coordinate-wise changes
            if progress > 0.7:
                CR_i *= (1.0 - 0.5 * (progress - 0.7) / 0.3)
            
            idxs = list(range(cur_pop))
            idxs.remove(i)
            
            # current-to-pbest/1 with archive
            p_best_idx = np.random.randint(p)  # population is sorted
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
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR_i
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            # Bounce-back clipping
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
            elif f_trial == fitness[i]:
                # Accept equal with small probability for diversity
                if np.random.random() < 0.01:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
        
        population = new_population
        fitness = new_fitness
        
        # Update memory (weighted Lehmer mean)
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
        
        # Sort and reduce
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
        
        if stagnation > 20:
            n_replace = max(1, len(population) // 3)
            for i in range(len(population) - n_replace, len(population)):
                if np.random.random() < 0.3:
                    population[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
                else:
                    sigma = 0.02 * ranges * max(0.001, 1.0 - progress)
                    population[i] = best_x + sigma * np.random.randn(dim)
                population[i] = clip(population[i])
                fitness[i] = evaluate(population[i])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Simplified CMA-ES from best ---
    if best_x is not None and elapsed() < max_time * 0.88:
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_p = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = best_x.copy()
        sigma = 0.02 * np.max(ranges)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        eigeneval = 0
        counteval = 0
        
        while elapsed() < max_time * 0.88:
            # Eigen decomposition
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
            elif counteval == 0:
                D = np.ones(n)
                B = np.eye(n)
                invsqrtC = np.eye(n)
            
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if elapsed() >= max_time * 0.88:
                    break
                z = np.random.randn(n)
                arx[k] = mean + sigma * (B @ (D * z))
                arx[k] = clip(arx[k])
                arfitness[k] = evaluate(arx[k])
                counteval += 1
            
            if elapsed() >= max_time * 0.88:
                break
            
            arindex = np.argsort(arfitness)
            selected = arx[arindex[:mu]]
            old_mean = mean.copy()
            mean = selected.T @ weights
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu_p) * C + c1 * (np.outer(pc, pc) + (1-hsig) * cc*(2-cc) * C) + cmu_p * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-15 * np.max(ranges), 0.5 * np.max(ranges))
            
            if sigma * np.max(D) < 1e-13 * np.max(ranges):
                break
    
    # --- Phase 4: Multi-scale coordinate descent ---
    if best_x is not None:
        for scale in [0.05, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]:
            if elapsed() >= max_time * 0.96:
                break
            step = scale * ranges
            for _ in range(5):
                if elapsed() >= max_time * 0.96:
                    break
                improved = False
                for j in range(dim):
                    if elapsed() >= max_time * 0.96:
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
    
    # --- Phase 5: Golden section on each coordinate ---
    if best_x is not None and elapsed() < max_time * 0.99:
        gr = (np.sqrt(5) - 1) / 2
        for j in range(dim):
            if elapsed() >= max_time * 0.99:
                break
            radius = 0.001 * ranges[j]
            a = max(lower[j], best_x[j] - radius)
            b = min(upper[j], best_x[j] + radius)
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            trial_c = best_x.copy(); trial_c[j] = c
            trial_d = best_x.copy(); trial_d[j] = d
            fc = evaluate(trial_c)
            fd = evaluate(trial_d)
            for _ in range(8):
                if elapsed() >= max_time * 0.99:
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
    
    # --- Phase 6: Final random perturbations ---
    while elapsed() < max_time * 0.998 and best_x is not None:
        sigma = 0.0002 * ranges * np.random.random()
        evaluate(best_x + sigma * np.random.randn(dim))
    
    return best
