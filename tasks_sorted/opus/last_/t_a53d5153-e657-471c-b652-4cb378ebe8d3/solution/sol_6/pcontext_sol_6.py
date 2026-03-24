#Looking at the previous algorithm, I see several potential improvements:
#
#1. The CMA-ES restart strategy could be more sophisticated with better budget allocation
#2. The IPOP multiplier logic has issues (lam_multiplier only increases for restart%3==0 cases)
#3. Adding a differential evolution phase for better global exploration
#4. Better Nelder-Mead initialization and convergence detection
#5. More aggressive local search near the end
#
#Let me redesign with a cleaner multi-strategy approach:
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
        return max(0, max_time - elapsed())
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # === Phase 1: Latin Hypercube Sampling ===
    n_init = min(max(20 * dim, 200), 800)
    population = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        population[:, d] = (perm + np.random.uniform(0, 1, n_init)) / n_init
    population = lower + population * ranges
    
    fitness = np.empty(n_init)
    actual_init = 0
    for i in range(n_init):
        if elapsed() >= max_time * 0.07:
            break
        fitness[i] = eval_f(population[i])
        actual_init = i + 1
    
    if actual_init == 0:
        return best
    
    population = population[:actual_init]
    fitness = fitness[:actual_init]
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # === Phase 2: DE/current-to-pbest/1 (SHADE-like) ===
    def run_de(pop, fit, deadline):
        nonlocal best, best_params
        NP = len(pop)
        if NP < 6:
            return pop, fit
        
        p_best_rate = 0.1
        archive = []
        M_CR = np.full(NP, 0.5)
        M_F = np.full(NP, 0.5)
        k = 0
        
        gen = 0
        while elapsed() < deadline:
            gen += 1
            S_CR, S_F, S_df = [], [], []
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            r_i = np.random.randint(0, NP, NP)
            CR_vals = np.clip(np.random.normal(M_CR[r_i], 0.1), 0, 1)
            F_vals = np.clip(np.random.standard_cauchy(NP) * 0.1 + M_F[r_i], 0, 2)
            
            p_best_size = max(2, int(NP * p_best_rate))
            
            for i in range(NP):
                if elapsed() >= deadline:
                    return new_pop, new_fit
                
                # Mutation: current-to-pbest/1
                p_idx = np.random.randint(0, p_best_size)
                p_best_sorted = np.argsort(fit)
                x_pbest = pop[p_best_sorted[p_idx]]
                
                candidates = list(range(NP))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                
                combined = list(range(NP)) + list(range(len(archive)))
                combined = [c for c in combined if c != i and c != r1]
                if len(combined) == 0:
                    combined = [r1]
                r2_idx = np.random.choice(combined)
                if r2_idx < NP:
                    x_r2 = pop[r2_idx]
                else:
                    x_r2 = archive[r2_idx - NP]
                
                F = F_vals[i]
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (pop[r1] - x_r2)
                
                # Crossover
                CR = CR_vals[i]
                j_rand = np.random.randint(dim)
                mask = (np.random.rand(dim) < CR)
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Bounce-back
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + pop[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + pop[i][d]) / 2
                trial = np.clip(trial, lower, upper)
                
                f_trial = eval_f(trial)
                
                if f_trial <= fit[i]:
                    delta_f = fit[i] - f_trial
                    if f_trial < fit[i]:
                        archive.append(pop[i].copy())
                        S_CR.append(CR)
                        S_F.append(F)
                        S_df.append(delta_f)
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit
            
            # Trim archive
            while len(archive) > NP:
                archive.pop(np.random.randint(len(archive)))
            
            # Update memories
            if len(S_CR) > 0:
                S_df_arr = np.array(S_df)
                w = S_df_arr / (S_df_arr.sum() + 1e-30)
                S_CR_arr = np.array(S_CR)
                S_F_arr = np.array(S_F)
                
                mean_CR = np.sum(w * S_CR_arr) if np.sum(S_CR_arr) > 0 else M_CR[k]
                mean_F = np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
                
                M_CR[k] = mean_CR
                M_F[k] = mean_F
                k = (k + 1) % NP
            
            # Convergence check
            if np.std(fit) < 1e-14 * (abs(np.mean(fit)) + 1e-30):
                break
        
        return pop, fit

    de_pop_size = min(max(6 * dim, 40), 200)
    if actual_init >= de_pop_size:
        de_pop = population[:de_pop_size].copy()
        de_fit = fitness[:de_pop_size].copy()
    else:
        extra = de_pop_size - actual_init
        extra_pop = lower + np.random.rand(extra, dim) * ranges
        de_pop = np.vstack([population, extra_pop])
        de_fit = np.empty(de_pop_size)
        de_fit[:actual_init] = fitness
        for i in range(actual_init, de_pop_size):
            if elapsed() >= max_time * 0.1:
                de_pop = de_pop[:i]
                de_fit = de_fit[:i]
                break
            de_fit[i] = eval_f(de_pop[i])

    if len(de_fit) >= 6:
        de_deadline = max_time * 0.40
        de_pop, de_fit = run_de(de_pop, de_fit, de_deadline)

    # === Phase 3: CMA-ES with IPOP restarts ===
    def run_cmaes(init_mean, init_sigma, deadline, lam_scale=1):
        nonlocal best, best_params
        n = dim
        lam = int(max(4 + int(3 * np.log(n)), 6) * lam_scale)
        lam = max(lam, 6)
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n+1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + mu_eff))
        damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = np.clip(init_mean.copy(), lower, upper)
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 100)
        
        if use_full:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            invsqrtC = np.eye(n)
            eigen_counter = 0
            eigen_interval = max(1, int(1 / (10*n*(c1 + cmu) + 1e-30)))
        else:
            diag_C = np.ones(n)
        
        stale = 0
        best_gen_f = float('inf')
        f_history = []
        gen = 0
        
        while elapsed() < deadline:
            gen += 1
            offspring = np.empty((lam, n))
            f_off = np.empty(lam)
            
            for i in range(lam):
                if elapsed() >= deadline:
                    return
                z = np.random.randn(n)
                if use_full:
                    offspring[i] = mean + sigma * (eigenvectors @ (eigenvalues * z))
                else:
                    offspring[i] = mean + sigma * np.sqrt(np.maximum(diag_C, 1e-20)) * z
                offspring[i] = np.clip(offspring[i], lower, upper)
                f_off[i] = eval_f(offspring[i])
            
            order = np.argsort(f_off)
            selected = offspring[order[:mu]]
            
            old_mean = mean.copy()
            mean = np.dot(weights, selected)
            mean = np.clip(mean, lower, upper)
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ diff)
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * diff / np.sqrt(np.maximum(diag_C, 1e-20))
            
            ps_norm = np.linalg.norm(ps)
            gen_thresh = max(1 - (1-cs)**(2*gen), 1e-30)
            hs = 1 if ps_norm / np.sqrt(gen_thresh) < (1.4 + 2/(n+1)) * chi_n else 0
            
            pc = (1-cc)*pc + hs * np.sqrt(cc*(2-cc)*mu_eff) * diff
            
            if use_full:
                artmp = ((selected - old_mean) / max(sigma, 1e-30)).T
                C = (1 - c1 - cmu + (1-hs)*c1*cc*(2-cc)) * C + c1 * np.outer(pc, pc) + cmu * (artmp @ np.diag(weights) @ artmp.T)
                C = np.triu(C) + np.triu(C, 1).T
                
                eigen_counter += 1
                if eigen_counter >= eigen_interval:
                    eigen_counter = 0
                    try:
                        eig_vals, eigenvectors = np.linalg.eigh(C)
                        eig_vals = np.maximum(eig_vals, 1e-20)
                        eigenvalues = np.sqrt(eig_vals)
                        invsqrtC = eigenvectors @ np.diag(1.0/eigenvalues) @ eigenvectors.T
                    except:
                        C = np.eye(n)
                        eigenvalues = np.ones(n)
                        eigenvectors = np.eye(n)
                        invsqrtC = np.eye(n)
            else:
                artmp = (selected - old_mean) / max(sigma, 1e-30)
                diag_C = (1 - c1 - cmu + (1-hs)*c1*cc*(2-cc)) * diag_C + \
                         c1 * pc**2 + cmu * np.sum(weights[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp(np.clip((cs/damps) * (ps_norm/chi_n - 1), -0.5, 0.5))
            sigma = np.clip(sigma, 1e-17 * np.mean(ranges), 2.0 * np.mean(ranges))
            
            cur_best = f_off[order[0]]
            f_history.append(cur_best)
            
            if cur_best < best_gen_f - 1e-15:
                best_gen_f = cur_best
                stale = 0
            else:
                stale += 1
            
            if use_full:
                max_std = sigma * np.max(eigenvalues)
            else:
                max_std = sigma * np.max(np.sqrt(diag_C))
            
            if max_std < 1e-14 * np.mean(ranges):
                return
            if stale > 10 + 30 * n // lam:
                return
            if len(f_history) > 30:
                recent = f_history[-30:]
                if max(recent) - min(recent) < 1e-14 * (abs(best) + 1e-30):
                    return

    # IPOP restart schedule
    lam_mult = 1.0
    restart = 0
    
    while elapsed() < max_time * 0.88:
        if remaining() < 0.3:
            break
        
        if restart == 0:
            sp = best_params.copy()
            sig = 0.15 * np.mean(ranges)
            lm = 1
        elif restart == 1:
            sp = best_params.copy()
            sig = 0.03 * np.mean(ranges)
            lm = 1
        elif restart % 4 == 0:
            # Full random restart with larger population
            sp = lower + np.random.rand(dim) * ranges
            lam_mult = min(lam_mult * 2, 10)
            sig = 0.3 * np.mean(ranges)
            lm = lam_mult
        elif restart % 4 == 2:
            # Random restart from one of top DE solutions
            if len(de_fit) > 3:
                top_idx = np.argsort(de_fit)[:max(3, len(de_fit)//5)]
                chosen = de_pop[np.random.choice(top_idx)]
                sp = chosen + 0.05 * ranges * np.random.randn(dim)
                sp = np.clip(sp, lower, upper)
            else:
                sp = best_params + 0.15 * ranges * np.random.randn(dim)
                sp = np.clip(sp, lower, upper)
            sig = 0.1 * np.mean(ranges)
            lm = 1
        else:
            sp = best_params + 0.08 * ranges * np.random.randn(dim) / (1 + restart * 0.1)
            sp = np.clip(sp, lower, upper)
            sig = 0.08 * np.mean(ranges) / (1 + restart * 0.15)
            lm = 1
        
        time_budget = max(remaining() * 0.25, 0.5)
        deadline = elapsed() + time_budget
        deadline = min(deadline, max_time * 0.88)
        
        run_cmaes(sp, sig, deadline, lm)
        restart += 1

    # === Phase 4: Nelder-Mead local refinement ===
    if best_params is not None and elapsed() < max_time * 0.97:
        n = dim
        scale = 0.001 * ranges
        simplex = np.empty((n+1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i]
            simplex[i+1] = np.clip(simplex[i+1], lower, upper)
        
        f_simplex = np.empty(n+1)
        for i in range(n+1):
            if elapsed() >= max_time * 0.97:
                break
            f_simplex[i] = eval_f(simplex[i])
        else:
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            while elapsed() < max_time * 0.995:
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lower, upper)
                fr = eval_f(xr)
                
                if fr < f_simplex[0]:
                    xe = np.clip(centroid + gamma * (xr - centroid), lower, upper)
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1], f_simplex[-1] = xe, fe
                    else:
                        simplex[-1], f_simplex[-1] = xr, fr
                elif fr < f_simplex[-2]:
                    simplex[-1], f_simplex[-1] = xr, fr
                else:
                    if fr < f_simplex[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                        fc = eval_f(xc)
                        if fc <= fr:
                            simplex[-1], f_simplex[-1] = xc, fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                                if elapsed() >= max_time * 0.995:
                                    break
                    else:
                        xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                        fc = eval_f(xc)
                        if fc < f_simplex[-1]:
                            simplex[-1], f_simplex[-1] = xc, fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_nm*(simplex[i]-simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                                if elapsed() >= max_time * 0.995:
                                    break

    # === Phase 5: Final random micro-perturbations ===
    for sc in [1e-4, 1e-5, 1e-6]:
        while elapsed() < max_time * 0.999:
            candidate = best_params + sc * ranges * np.random.randn(dim)
            candidate = np.clip(candidate, lower, upper)
            f_c = eval_f(candidate)
            if f_c < best - 1e-15:
                break  # Found improvement, will continue in outer loop
        else:
            continue
        break
    
    # Any remaining time
    while elapsed() < max_time * 0.999:
        candidate = best_params + 1e-6 * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best
