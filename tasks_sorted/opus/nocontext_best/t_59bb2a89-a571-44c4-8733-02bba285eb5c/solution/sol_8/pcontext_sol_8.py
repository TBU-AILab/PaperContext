#Looking at the current algorithm, I see an L-SHADE with local search. Let me analyze potential improvements:
#
#1. **Better initialization** with opposition-based learning
#2. **Improved DE strategy** with current-to-pbest/1 and eigenvector-based crossover
#3. **More aggressive local search** with CMA-ES-like covariance adaptation
#4. **Better time allocation** and restart strategy
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
    
    def time_left():
        return max_time - elapsed()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    evals = 0
    def eval_func(x):
        nonlocal best, best_x, evals
        x = clip_to_bounds(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # --- Phase 1: Opposition-based LHS initialization ---
    init_pop_size = min(max(20, 8 * dim), 200)
    pop_size = init_pop_size
    min_pop_size = 4

    # LHS
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + np.random.random(pop_size)) / pop_size
    population = lower + population * ranges
    
    fitness = np.array([eval_func(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.90])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
        pop_size = len(fitness)
    
    # Opposition-based population
    if elapsed() < max_time * 0.15:
        opp_pop = lower + upper - population
        opp_fitness = np.array([eval_func(opp_pop[i]) for i in range(pop_size) if elapsed() < max_time * 0.15])
        n_opp = len(opp_fitness)
        if n_opp > 0:
            combined_pop = np.vstack([population, opp_pop[:n_opp]])
            combined_fit = np.concatenate([fitness, opp_fitness])
            sidx = np.argsort(combined_fit)[:pop_size]
            population = combined_pop[sidx]
            fitness = combined_fit[sidx]

    # --- Phase 2: L-SHADE with eigenvector crossover ---
    H = 5
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    max_archive = pop_size
    
    no_improve_count = 0
    prev_best = best
    generation = 0
    
    # Track history for covariance estimation
    success_diffs = []
    
    de_time_budget = max_time * 0.78
    
    while elapsed() < de_time_budget:
        generation += 1
        
        sorted_idx = np.argsort(fitness)
        
        # Adaptive p value
        p_rate = max(0.05, 0.25 - 0.20 * (elapsed() / de_time_budget))
        p_best_size = max(2, int(p_rate * pop_size))
        
        S_F = []
        S_CR = []
        S_delta = []
        
        # Compute eigenvector rotation if enough successes
        use_eigen = len(success_diffs) >= dim and np.random.random() < 0.3
        if use_eigen:
            try:
                diff_mat = np.array(success_diffs[-min(len(success_diffs), 5*dim):])
                cov = np.cov(diff_mat.T) + 1e-20 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                rotation = eigenvectors
            except:
                use_eigen = False
        
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Generate all F, CR at once
        for i in range(pop_size):
            if elapsed() >= de_time_budget:
                break
            
            ri = np.random.randint(0, H)
            
            while True:
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F > 0:
                    break
            F = min(F, 1.0)
            
            if M_CR[ri] < 0:
                CR = 0.0
            else:
                CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
            
            p_best_idx = sorted_idx[np.random.randint(0, p_best_size)]
            
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            
            combined_size = pop_size + len(archive)
            r2 = i
            att = 0
            while (r2 == i or r2 == r1) and att < 25:
                r2 = np.random.randint(0, combined_size)
                att += 1
            
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            # Mutation: current-to-pbest/1
            diff1 = population[p_best_idx] - population[i]
            diff2 = population[r1] - x_r2
            mutant = population[i] + F * diff1 + F * diff2
            
            # Crossover - eigenvector or binomial
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            
            if use_eigen and np.random.random() < 0.5:
                # Eigenvector crossover
                x_rot = rotation.T @ population[i]
                m_rot = rotation.T @ mutant
                mask = np.random.random(dim) < CR
                mask[j_rand] = True
                t_rot = x_rot.copy()
                t_rot[mask] = m_rot[mask]
                trial = rotation @ t_rot
            else:
                mask = np.random.random(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]
            
            # Bounce-back
            out_low = trial < lower
            out_high = trial > upper
            if np.any(out_low):
                trial[out_low] = (lower[out_low] + population[i][out_low]) / 2.0
            if np.any(out_high):
                trial[out_high] = (upper[out_high] + population[i][out_high]) / 2.0
            trial = clip_to_bounds(trial)
            
            trial_f = eval_func(trial)
            
            if trial_f <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_f
                if trial_f < fitness[i]:
                    delta = fitness[i] - trial_f
                    S_F.append(F)
                    S_CR.append(CR)
                    S_delta.append(delta)
                    success_diffs.append(trial - population[i])
                    if len(success_diffs) > 10 * dim:
                        success_diffs = success_diffs[-5*dim:]
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
        
        population = new_population
        fitness = new_fitness
        
        # Update memory
        if len(S_F) > 0:
            weights = np.array(S_delta)
            weights = weights / (np.sum(weights) + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(weights * sf ** 2) / (np.sum(weights * sf) + 1e-30)
            M_CR[k] = np.sum(weights * scr)
            k = (k + 1) % H
        
        if best < prev_best - 1e-15:
            no_improve_count = 0
            prev_best = best
        else:
            no_improve_count += 1
        
        # Population reduction
        fraction = elapsed() / de_time_budget
        new_pop_size = max(min_pop_size, int(round(init_pop_size + (min_pop_size - init_pop_size) * fraction)))
        
        if new_pop_size < pop_size:
            sidx = np.argsort(fitness)[:new_pop_size]
            population = population[sidx]
            fitness = fitness[sidx]
            pop_size = new_pop_size
            max_archive = pop_size
            while len(archive) > max_archive:
                archive.pop(np.random.randint(len(archive)))
        
        # Stagnation restart
        if no_improve_count > 40 and pop_size > min_pop_size + 2:
            no_improve_count = 0
            sidx = np.argsort(fitness)
            n_restart = max(1, pop_size // 3)
            for kk in range(n_restart):
                idx = sidx[pop_size - 1 - kk]
                if np.random.random() < 0.3:
                    population[idx] = best_x + np.random.randn(dim) * ranges * 0.05
                elif np.random.random() < 0.5:
                    population[idx] = best_x + np.random.randn(dim) * ranges * 0.2
                else:
                    population[idx] = lower + np.random.random(dim) * ranges
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = eval_func(population[idx])

    # --- Phase 3: CMA-ES-like local search ---
    if best_x is not None and time_left() > max_time * 0.02:
        sigma = 0.01
        x_mean = best_x.copy()
        f_mean = best
        C = np.eye(dim)
        lam = max(4, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_cma = weights_cma / np.sum(weights_cma)
        mu_eff = 1.0 / np.sum(weights_cma ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        cma_gen = 0
        while time_left() > max_time * 0.01:
            cma_gen += 1
            try:
                eigenvalues_c, B = np.linalg.eigh(C)
                eigenvalues_c = np.maximum(eigenvalues_c, 1e-20)
                D = np.sqrt(eigenvalues_c)
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
            except:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                invsqrtC = np.eye(dim)
            
            arz = np.random.randn(lam, dim)
            arx = np.zeros((lam, dim))
            arf = np.zeros(lam)
            
            for j in range(lam):
                if time_left() < max_time * 0.005:
                    return best
                arx[j] = x_mean + sigma * (B @ (D * arz[j]))
                arx[j] = clip_to_bounds(arx[j])
                arf[j] = eval_func(arx[j])
            
            idx_sort = np.argsort(arf)
            arx = arx[idx_sort]
            arz = arz[idx_sort]
            arf = arf[idx_sort]
            
            old_mean = x_mean.copy()
            x_mean = np.sum(weights_cma[:, None] * arx[:mu], axis=0)
            
            z_mean = np.sum(weights_cma[:, None] * arz[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrtC @ (x_mean - old_mean) / sigma)
            
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - old_mean) / sigma
            
            artmp = (arx[:mu] - old_mean) / sigma
            C = (1 - c1 - c_mu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * (artmp.T @ np.diag(weights_cma) @ artmp)
            
            sigma *= np.exp(min(0.6, (c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1)))
            sigma = max(sigma, 1e-20)
            
            if sigma * np.max(D) < 1e-15 * np.max(ranges):
                break
            if sigma * np.max(D) > 2 * np.max(ranges):
                sigma = 0.01
                C = np.eye(dim)
                x_mean = best_x.copy()
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

    # --- Phase 4: Final coordinate descent polish ---
    if best_x is not None and time_left() > max_time * 0.005:
        x = best_x.copy()
        fx = best
        step = 0.001 * ranges.copy()
        
        while time_left() > max_time * 0.002:
            improved = False
            for j in range(dim):
                if time_left() < max_time * 0.002:
                    return best
                for direction in [1, -1]:
                    trial = x.copy()
                    trial[j] = np.clip(trial[j] + direction * step[j], lower[j], upper[j])
                    ft = eval_func(trial)
                    if ft < fx:
                        fx = ft
                        x = trial
                        step[j] *= 1.5
                        improved = True
                        break
                else:
                    step[j] *= 0.5
            if not improved and np.max(step / ranges) < 1e-15:
                break
    
    return best
