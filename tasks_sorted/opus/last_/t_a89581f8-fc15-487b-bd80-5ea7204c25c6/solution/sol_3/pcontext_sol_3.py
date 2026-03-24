#Looking at the previous algorithm, I'll make several improvements:
#
#1. **Better population sizing and initialization** with multiple strategies
#2. **Improved SHADE with linear population size reduction (L-SHADE)**
#3. **Better CMA-ES with restarts (IPOP-CMA-ES style)**
#4. **Multi-start strategy to avoid local optima**
#5. **Coordinate-wise local search for fine-tuning**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_pos = None
    evals = 0
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_pos, evals
        evals += 1
        v = func(clip(x))
        if v < best:
            best = v
            best_pos = x.copy()
        return v
    
    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = (perm[i] + np.random.rand()) / n
        return lower + samples * ranges
    
    def opposition(pop):
        return lower + upper - pop
    
    # --- L-SHADE with improvements ---
    def run_lshade(pop_init, fit_init, budget_time, min_pop=4):
        nonlocal best, best_pos
        t_start = elapsed()
        
        pop_size_init = len(pop_init)
        pop_size = pop_size_init
        population = pop_init.copy()
        fitness = fit_init.copy()
        
        H = 100
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k_idx = 0
        archive = []
        archive_max = pop_size_init
        
        gen = 0
        max_gen_est = max(1, int(budget_time * 50))  # rough estimate
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3 and pop_size >= min_pop:
            gen += 1
            S_F, S_CR, delta_f = [], [], []
            
            sorted_idx = np.argsort(fitness[:pop_size])
            
            new_pop = []
            new_fit = []
            
            for i in range(pop_size):
                if time_left() < 0.2:
                    return population[:pop_size], fitness[:pop_size]
                
                ri = np.random.randint(H)
                
                # Cauchy for F
                Fi = -1
                attempts = 0
                while Fi <= 0 and attempts < 10:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    attempts += 1
                if Fi <= 0:
                    Fi = 0.01
                Fi = min(Fi, 1.0)
                
                # Normal for CR
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0, 1)
                
                # p-best
                p_val = max(2/pop_size, 0.05 + 0.15 * (1 - gen/max_gen_est))
                p_num = max(2, int(p_val * pop_size))
                pbest_idx = sorted_idx[np.random.randint(min(p_num, pop_size))]
                
                # Mutation: current-to-pbest/1 with archive
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                pool_size = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pool_size)
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
                
                # Binomial crossover
                cross = np.random.rand(dim) < CRi
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, population[i])
                
                # Boundary handling: midpoint
                mask_lo = trial < lower
                mask_hi = trial > upper
                trial[mask_lo] = (lower[mask_lo] + population[i][mask_lo]) / 2
                trial[mask_hi] = (upper[mask_hi] + population[i][mask_hi]) / 2
                trial = clip(trial)
                
                trial_fit = eval_f(trial)
                
                if trial_fit <= fitness[i]:
                    if trial_fit < fitness[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - trial_fit))
                        if len(archive) < archive_max:
                            archive.append(population[i].copy())
                        elif archive_max > 0:
                            archive[np.random.randint(archive_max)] = population[i].copy()
                    new_pop.append(trial)
                    new_fit.append(trial_fit)
                else:
                    new_pop.append(population[i].copy())
                    new_fit.append(fitness[i])
            
            population_all = np.array(new_pop)
            fitness_all = np.array(new_fit)
            
            # Update memory
            if len(S_F) > 0:
                w = np.array(delta_f)
                ws = w.sum()
                if ws > 0:
                    w = w / ws
                else:
                    w = np.ones(len(w)) / len(w)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                M_F[k_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                mean_cr = np.sum(w * scr)
                if max(scr) == 0:
                    M_CR[k_idx] = -1
                else:
                    M_CR[k_idx] = mean_cr
                k_idx = (k_idx + 1) % H
            
            # Linear population size reduction
            new_pop_size = max(min_pop, int(round(pop_size_init - (pop_size_init - min_pop) * gen / max_gen_est)))
            new_pop_size = max(min_pop, min(new_pop_size, pop_size))
            
            if new_pop_size < pop_size:
                order = np.argsort(fitness_all)
                population = population_all[order[:new_pop_size]].copy()
                fitness = fitness_all[order[:new_pop_size]].copy()
                pop_size = new_pop_size
                while len(archive) > pop_size:
                    archive.pop(np.random.randint(len(archive)))
                archive_max = pop_size
            else:
                population = population_all
                fitness = fitness_all
        
        return population[:pop_size], fitness[:pop_size]
    
    # --- CMA-ES ---
    def run_cmaes(x0, sigma0, budget_time):
        nonlocal best, best_pos
        t_start = elapsed()
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_ = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = clip(x0.copy())
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_sep = (n > 40)
        
        if use_sep:
            C_diag = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            invsqrtC = np.eye(n)
            eigeneval = 0
        
        counteval = 0
        no_improve = 0
        prev_best = best
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.3:
            arx = np.zeros((lam, n))
            arfitness = np.zeros(lam)
            
            for k in range(lam):
                if time_left() < 0.2:
                    return
                z = np.random.randn(n)
                if use_sep:
                    arx[k] = clip(mean + sigma * np.sqrt(C_diag) * z)
                else:
                    arx[k] = clip(mean + sigma * (B @ (D * z)))
                arfitness[k] = eval_f(arx[k])
                counteval += 1
            
            idx = np.argsort(arfitness)
            
            old_mean = mean.copy()
            mean = np.zeros(n)
            for k in range(mu):
                mean += weights[k] * arx[idx[k]]
            mean = clip(mean)
            
            diff = (mean - old_mean) / (sigma + 1e-30)
            
            if use_sep:
                invsqrt_diag = 1.0 / np.sqrt(C_diag + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * invsqrt_diag * diff
            else:
                ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ diff)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*counteval/lam)) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * diff
            
            if use_sep:
                artmp = np.zeros((mu, n))
                for k in range(mu):
                    artmp[k] = (arx[idx[k]] - old_mean) / (sigma + 1e-30)
                C_diag = ((1 - c1 - cmu_) * C_diag + 
                          c1 * (pc**2 + (1-hsig)*cc*(2-cc)*C_diag) + 
                          cmu_ * np.sum(weights[:, None] * artmp**2, axis=0))
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                artmp = np.zeros((mu, n))
                for k in range(mu):
                    artmp[k] = (arx[idx[k]] - old_mean) / (sigma + 1e-30)
                C = (1 - c1 - cmu_) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
                for k in range(mu):
                    C += cmu_ * weights[k] * np.outer(artmp[k], artmp[k])
                
                if counteval - eigeneval > lam / (c1 + cmu_) / n / 10:
                    eigeneval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        D_sq, B = np.linalg.eigh(C)
                        D_sq = np.maximum(D_sq, 1e-20)
                        D = np.sqrt(D_sq)
                        invsqrtC = B @ np.diag(1.0/D) @ B.T
                    except:
                        B = np.eye(n)
                        D = np.ones(n)
                        C = np.eye(n)
                        invsqrtC = np.eye(n)
            
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            if best < prev_best - 1e-12:
                no_improve = 0
                prev_best = best
            else:
                no_improve += 1
            
            if sigma < 1e-14:
                break
            if no_improve > 50 + 10 * dim:
                break
            if not use_sep and np.max(D) > 1e7 * np.min(D):
                break
    
    # --- Coordinate descent local search ---
    def coord_search(x0, budget_time):
        nonlocal best, best_pos
        t_start = elapsed()
        x = clip(x0.copy())
        fx = eval_f(x)
        
        step_sizes = 0.01 * ranges.copy()
        
        for iteration in range(200):
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            improved = False
            for d in range(dim):
                if time_left() < 0.2:
                    return
                
                # Try positive step
                x_trial = x.copy()
                x_trial[d] = min(upper[d], x[d] + step_sizes[d])
                f_trial = eval_f(x_trial)
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    step_sizes[d] *= 1.2
                    improved = True
                    continue
                
                # Try negative step
                x_trial = x.copy()
                x_trial[d] = max(lower[d], x[d] - step_sizes[d])
                f_trial = eval_f(x_trial)
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    step_sizes[d] *= 1.2
                    improved = True
                    continue
                
                step_sizes[d] *= 0.5
            
            if not improved and np.max(step_sizes / ranges) < 1e-12:
                break
    
    # --- Golden section line search ---
    def golden_section_search(x_base, direction, budget_time):
        nonlocal best, best_pos
        t_start = elapsed()
        
        gr = (np.sqrt(5) + 1) / 2
        
        # Find valid range along direction
        a = 0.0
        b = 1.0
        
        # Scale direction so b=1 is reasonable
        max_step = np.min(np.where(direction > 0, (upper - x_base) / (direction + 1e-30),
                                     np.where(direction < 0, (lower - x_base) / (direction - 1e-30), 1e10)))
        max_step = min(max_step, np.max(ranges))
        if max_step <= 0:
            return
        
        b = max_step
        
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        
        fc = eval_f(clip(x_base + c * direction))
        fd = eval_f(clip(x_base + d * direction))
        
        for _ in range(30):
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            if abs(b - a) < 1e-12:
                break
            
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = eval_f(clip(x_base + c * direction))
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = eval_f(clip(x_base + d * direction))
    
    # --- Nelder-Mead ---
    def nelder_mead(x0, budget_time, scale=0.05):
        nonlocal best, best_pos
        t_start = elapsed()
        n = dim
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = clip(x0.copy())
        for i in range(n):
            simplex[i+1] = x0.copy()
            delta = scale * ranges[i]
            if delta < 1e-15:
                delta = 0.01
            simplex[i+1][i] += delta * (1 if np.random.rand() > 0.5 else -1)
            simplex[i+1] = clip(simplex[i+1])
        
        f_vals = np.array([eval_f(simplex[i]) for i in range(n+1)])
        
        for _ in range(10000):
            if (elapsed() - t_start) > budget_time or time_left() < 0.2:
                break
            
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < f_vals[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1] = xe; f_vals[-1] = fe
                else:
                    simplex[-1] = xr; f_vals[-1] = fr
            elif fr < f_vals[-2]:
                simplex[-1] = xr; f_vals[-1] = fr
            else:
                if fr < f_vals[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1] = xc; f_vals[-1] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = eval_f(xc)
                    if fc < f_vals[-1]:
                        simplex[-1] = xc; f_vals[-1] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_vals[i] = eval_f(simplex[i])
            
            if np.max(f_vals) - np.min(f_vals) < 1e-16:
                break
            if np.max(np.abs(simplex[-1] - simplex[0])) < 1e-15:
                break
    
    # =================== MAIN ORCHESTRATION ===================
    
    pop_size = min(max(30, 10 * dim), 300)
    
    # Phase 1: Initial sampling with opposition
    pop = lhs_sample(pop_size)
    fit = np.array([eval_f(p) for p in pop])
    
    if time_left() > 1.0:
        opp_pop = opposition(pop)
        opp_fit = np.array([eval_f(p) for p in opp_pop])
        
        combined = np.vstack([pop, opp_pop])
        combined_fit = np.concatenate([fit, opp_fit])
        order = np.argsort(combined_fit)[:pop_size]
        pop = combined[order]
        fit = combined_fit[order]
    
    # Phase 2: L-SHADE
    lshade_time = max_time * 0.35
    if time_left() > 1.0:
        pop, fit = run_lshade(pop, fit, lshade_time)
    
    # Phase 3: CMA-ES from best
    if time_left() > 1.5 and best_pos is not None:
        sigma0 = 0.3 * np.mean(ranges)
        run_cmaes(best_pos.copy(), sigma0, min(max_time * 0.20, time_left() * 0.4))
    
    # Phase 4: Multi-start CMA-ES from diverse points
    if time_left() > 2.0 and pop is not None and len(pop) > 0:
        n_starts = min(3, max(1, int(time_left() / 2)))
        sorted_idx = np.argsort(fit)
        for s in range(min(n_starts, len(sorted_idx))):
            if time_left() < 1.5:
                break
            start_pt = pop[sorted_idx[s]].copy()
            # Add some noise
            if s > 0:
                start_pt += 0.1 * ranges * np.random.randn(dim)
                start_pt = clip(start_pt)
            run_cmaes(start_pt, 0.15 * np.mean(ranges), time_left() * 0.3)
    
    # Phase 5: Nelder-Mead refinement
    if time_left() > 1.0 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.4, scale=0.02)
    
    # Phase 6: Coordinate descent fine-tuning
    if time_left() > 0.5 and best_pos is not None:
        coord_search(best_pos.copy(), time_left() * 0.4)
    
    # Phase 7: Final Nelder-Mead polish
    if time_left() > 0.5 and best_pos is not None:
        nelder_mead(best_pos.copy(), time_left() * 0.7, scale=0.002)
    
    # Phase 8: Ultra-fine coordinate descent
    if time_left() > 0.3 and best_pos is not None:
        coord_search(best_pos.copy(), time_left() * 0.8)
    
    return best
