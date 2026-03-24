#Looking at the results, algorithm #1 (score 13.38) is the best. I'll improve it with:
#
#1. **Better SHADE parameters and population sizing**
#2. **Proper CMA-ES with covariance matrix** (diagonal for high dim)
#3. **Multi-restart CMA-ES** with different starting points from DE population
#4. **Adaptive local search** with coordinate-wise refinement
#5. **Better time allocation** - more time for CMA-ES refinement
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
    
    def evaluate(x):
        nonlocal best, best_params
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    pop_size_init = min(max(40, 15 * dim), 350)
    pop_size = pop_size_init
    min_pop_size = max(5, dim // 2)
    
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * ranges
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Phase 2: L-SHADE ---
    memory_size = 25
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    mem_idx = 0
    
    archive = []
    archive_max = pop_size_init
    
    stagnation = 0
    prev_best = best
    generation = 0
    t_de_start = elapsed()
    t_de_end = max_time * 0.45
    
    while elapsed() < t_de_end:
        generation += 1
        
        S_F = []
        S_CR = []
        S_delta = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= t_de_end:
                break
            
            ri = np.random.randint(0, memory_size)
            mu_F = M_F[ri]
            mu_CR = M_CR[ri]
            
            F_i = mu_F + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_F + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            
            CR_i = np.clip(mu_CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            p = max(2, int(pop_size * np.random.uniform(0.05, 0.2)))
            p_best_idx = np.random.randint(0, p)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(0, combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, combined_size)
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + F_i * (population[p_best_idx] - population[i]) + F_i * (population[r1] - x_r2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + population[i][d]) / 2
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + population[i][d]) / 2
            
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial_fitness = evaluate(trial)
            
            if trial_fitness <= fitness[i]:
                if trial_fitness < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(0, len(archive)))
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_delta.append(abs(fitness[i] - trial_fitness))
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        
        population = new_population
        fitness = new_fitness
        
        if S_F:
            weights = np.array(S_delta)
            weights = weights / (weights.sum() + 1e-30)
            mean_F = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            mean_CR = np.sum(weights * np.array(S_CR))
            M_F[mem_idx] = mean_F
            M_CR[mem_idx] = mean_CR
            mem_idx = (mem_idx + 1) % memory_size
        
        ratio = min(1.0, (elapsed() - t_de_start) / (t_de_end - t_de_start + 1e-10))
        new_pop_size = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * ratio)))
        sorted_idx = np.argsort(fitness)
        if new_pop_size < pop_size:
            population = population[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            pop_size = new_pop_size
        else:
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        if stagnation > 30:
            n_replace = max(1, pop_size // 3)
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * ranges
                fitness[j] = evaluate(population[j])
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0

    # Collect top candidates for restarts
    top_k = min(5, pop_size)
    candidates = [population[i].copy() for i in range(top_k)]
    
    # --- Phase 3: CMA-ES with multiple restarts ---
    def run_cmaes(init_mean, init_sigma, time_limit):
        nonlocal best, best_params
        n = dim
        use_full = (n <= 60)
        sigma = init_sigma
        mean = init_mean.copy()
        lam = max(10, 4 + int(3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        if use_full:
            C = np.eye(n)
        else:
            diag_C = np.ones(n)
        
        gen_cma = 0
        no_improve = 0
        local_best = best
        
        while elapsed() < time_limit:
            gen_cma += 1
            
            if use_full:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                    BD = B * D
                    inv_sqrt_C = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n)
                    BD = np.eye(n)
                    inv_sqrt_C = np.eye(n)
                    D = np.ones(n)
                    B = np.eye(n)
            
            samples = []
            f_s = []
            ys = []
            
            for _ in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                if use_full:
                    y = BD @ z
                else:
                    y = np.sqrt(diag_C) * z
                x = clip(mean + sigma * y)
                f = evaluate(x)
                samples.append(x)
                f_s.append(f)
                ys.append(y)
            
            if len(samples) < mu:
                return
            
            idx_s = np.argsort(f_s)
            sel = np.array([samples[idx_s[j]] for j in range(mu)])
            y_sel = np.array([ys[idx_s[j]] for j in range(mu)])
            
            old_mean = mean.copy()
            mean = clip(np.dot(w, sel))
            
            y_w = np.dot(w, y_sel)
            
            if use_full:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ y_w)
            else:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w / np.sqrt(diag_C + 1e-30))
            
            ps_norm = np.linalg.norm(ps)
            h_sig = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_cma)) < (1.4 + 2 / (n + 1)) * chiN else 0.0
            
            pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * y_w
            
            if use_full:
                rank_one = np.outer(pc, pc) + (1 - h_sig) * cc * (2 - cc) * C
                rank_mu = np.zeros((n, n))
                for j in range(mu):
                    rank_mu += w[j] * np.outer(y_sel[j], y_sel[j])
                C = (1 - c1 - cmu_val) * C + c1 * rank_one + cmu_val * rank_mu
                C = (C + C.T) / 2
                # Ensure positive definiteness
                eigv = np.linalg.eigvalsh(C)
                if eigv[0] < 1e-15:
                    C += (1e-15 - eigv[0]) * np.eye(n)
            else:
                diag_C = (1 - c1 - cmu_val) * diag_C + c1 * (pc ** 2 + (1 - h_sig) * cc * (2 - cc) * diag_C) + cmu_val * np.sum(w[:, None] * y_sel ** 2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp((c_sigma / d_sigma) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-15, 2.0)
            
            if best < local_best - 1e-12:
                local_best = best
                no_improve = 0
            else:
                no_improve += 1
            
            if sigma < 1e-13 or no_improve > 30 + 5 * n:
                return
    
    # Run CMA-ES from best
    if best_params is not None:
        run_cmaes(best_params.copy(), 0.1, max_time * 0.70)
    
    # Run CMA-ES from other top candidates
    for ci, cand in enumerate(candidates[1:]):
        if elapsed() >= max_time * 0.85:
            break
        run_cmaes(cand.copy(), 0.15, min(max_time * (0.70 + 0.05 * (ci + 1)), max_time * 0.85))
    
    # Another CMA-ES from best with smaller sigma
    if best_params is not None and elapsed() < max_time * 0.92:
        run_cmaes(best_params.copy(), 0.02, max_time * 0.92)
    
    # --- Phase 4: Coordinate-wise + random local search ---
    if best_params is not None:
        scale = 0.001
        no_imp = 0
        while elapsed() < max_time * 0.97:
            # Random perturbation
            candidate = clip(best_params + scale * ranges * np.random.randn(dim))
            old_b = best
            evaluate(candidate)
            if best < old_b:
                no_imp = 0
            else:
                no_imp += 1
            if no_imp > 40:
                scale *= 0.5
                no_imp = 0
            if scale < 1e-10:
                break
        
        # Coordinate descent
        step = 0.0005
        while elapsed() < max_time * 0.998:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.998:
                    break
                for sign in [1, -1]:
                    cand = best_params.copy()
                    cand[d] = np.clip(cand[d] + sign * step * ranges[d], lower[d], upper[d])
                    old_b = best
                    evaluate(cand)
                    if best < old_b:
                        improved = True
                        break
            if not improved:
                step *= 0.5
            if step < 1e-12:
                break
    
    return best
