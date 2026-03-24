#Looking at the previous code, I can identify several issues:
#
#1. **L-SHADE initialization bug** - fitness computed for focused case but also in the general path
#2. **Time allocation suboptimal** - phases are rigid rather than adaptive
#3. **Missing diversity mechanisms** - no multi-start exploration
#4. **CMA-ES eigendecomposition frequency too low** for small dimensions
#5. **No coordinate-wise local search** for fine-tuning
#
#Let me redesign with a cleaner, more robust approach:
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
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def lhs_init(n):
        pop = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            pop[:, j] = (perm + np.random.uniform(0, 1, n)) / n
        return lower + pop * ranges

    # ---- CMA-ES ----
    def cma_es(x0, sigma0, time_budget):
        nonlocal best, best_params
        if time_budget <= 0.05:
            return
        t_start = elapsed()
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 8)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_cov = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        C = np.eye(n)
        eigeneval = 0
        count_eval = 0
        no_improve_gen = 0
        local_best = best
        gen = 0
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.03:
            solutions = []
            f_vals = []
            for _ in range(lam):
                if time_left() <= 0.02:
                    return
                z = np.random.randn(n)
                x = mean + sigma * (B @ (D * z))
                x = clip(x)
                f = eval_func(x)
                count_eval += 1
                solutions.append(x)
                f_vals.append(f)
            
            idx = np.argsort(f_vals)
            old_mean = mean.copy()
            
            mean = np.zeros(n)
            for i in range(mu):
                mean += weights[i] * solutions[idx[i]]
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            
            norm_ps = np.linalg.norm(ps)
            hsig = int(norm_ps / np.sqrt(1 - (1-cs)**(2*(count_eval/lam+1))) / chiN < 1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = np.zeros((n, mu))
            for i in range(mu):
                artmp[:, i] = (solutions[idx[i]] - old_mean) / max(sigma, 1e-30)
            
            C = (1 - c1 - cmu_cov) * C + \
                c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                cmu_cov * (artmp * weights) @ artmp.T
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2.0 * np.max(ranges))
            
            gen += 1
            if count_eval - eigeneval > lam / (c1 + cmu_cov + 1e-30) / n / 5:
                eigeneval = count_eval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    invsqrtC = np.eye(n)
                    sigma = sigma0 * 0.5
            
            if best < local_best - 1e-12:
                no_improve_gen = 0
                local_best = best
            else:
                no_improve_gen += 1
            
            cond = np.max(D) / (np.min(D) + 1e-30)
            if sigma * np.max(D) < 1e-15 or no_improve_gen > 50 + 10*n or cond > 1e14:
                break

    # ---- L-SHADE ----
    def run_lshade(time_frac=0.5, pop_size_init=None, focused=False):
        nonlocal best, best_params
        if time_left() <= 0.15:
            return
        
        budget_time = time_left() * time_frac
        t_start = elapsed()
        
        if pop_size_init is None:
            pop_size_init = min(max(18 * dim, 50), 300)
        pop_size = pop_size_init
        N_init = pop_size_init
        N_min = 4
        
        if focused and best_params is not None:
            population = np.zeros((pop_size, dim))
            population[0] = best_params.copy()
            for i in range(1, pop_size):
                scale = np.random.uniform(0.002, 0.15)
                population[i] = clip(best_params + np.random.randn(dim) * ranges * scale)
            fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
        else:
            population = lhs_init(pop_size)
            # Opposition-based initialization
            opp_pop = lower + upper - population
            combined = np.vstack([population, opp_pop])
            combined_f = []
            for j in range(len(combined)):
                if time_left() <= 0.1:
                    break
                combined_f.append(eval_func(combined[j]))
            n_eval = len(combined_f)
            if n_eval < len(combined):
                combined_f.extend([float('inf')] * (len(combined) - n_eval))
            combined_f = np.array(combined_f)
            idx = np.argsort(combined_f)[:pop_size]
            population = combined[idx]
            fitness = combined_f[idx]
        
        if time_left() <= 0:
            return
        
        H = min(100, max(6 * dim, 20))
        M_F = np.full(H, 0.3)
        M_CR = np.full(H, 0.8)
        k_idx = 0
        archive = []
        max_archive = pop_size
        gen = 0
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.08:
            S_F, S_CR, delta_f = [], [], []
            sort_idx = np.argsort(fitness)
            p_best_rate = max(2.0/pop_size, 0.05 + 0.06 * (1.0 - min(1.0, gen/200.0)))
            p_best_size = max(2, int(p_best_rate * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if time_left() <= 0.04:
                    return
                
                ri = np.random.randint(H)
                
                Fi = -1
                for _ in range(20):
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                    if Fi > 0:
                        break
                Fi = np.clip(Fi, 0.01, 1.0)
                
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0, 1)
                
                pi = sort_idx[np.random.randint(p_best_size)]
                
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1 = candidates[np.random.randint(len(candidates))]
                
                union_size = pop_size + len(archive)
                r2 = np.random.randint(union_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(union_size)
                
                xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = population[i] + Fi * (population[pi] - population[i]) + Fi * (population[r1] - xr2)
                
                # Bounce-back boundary repair
                for j in range(dim):
                    if mutant[j] < lower[j]:
                        mutant[j] = lower[j] + np.random.random() * (population[i][j] - lower[j])
                    elif mutant[j] > upper[j]:
                        mutant[j] = upper[j] - np.random.random() * (upper[j] - population[i][j])
                
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                mask = (np.random.random(dim) < CRi) | (np.arange(dim) == j_rand)
                trial[mask] = mutant[mask]
                
                trial_f = eval_func(trial)
                
                if trial_f < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = trial_f
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - trial_f))
                elif trial_f == fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_f
            
            population = new_pop
            fitness = new_fit
            gen += 1
            
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
                if np.max(scr) <= 0:
                    M_CR[k_idx] = -1
                else:
                    M_CR[k_idx] = np.sum(w * scr)
                k_idx = (k_idx + 1) % H
            
            ratio = min(1.0, (elapsed() - t_start) / max(budget_time, 1e-10))
            new_pop_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
            
            if new_pop_size < pop_size:
                si = np.argsort(fitness)
                population = population[si[:new_pop_size]]
                fitness = fitness[si[:new_pop_size]]
                pop_size = new_pop_size
                max_archive = pop_size
                while len(archive) > max_archive:
                    archive.pop(np.random.randint(len(archive)))

    # ---- Nelder-Mead ----
    def nelder_mead(x0, time_budget, initial_scale=0.05):
        nonlocal best, best_params
        if time_budget <= 0.1 or dim > 50:
            return
        t_start = elapsed()
        n = dim
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            delta = initial_scale * ranges[i]
            if delta < 1e-12:
                delta = 1e-6
            simplex[i+1][i] += delta
            simplex[i+1] = clip(simplex[i+1])
        
        f_simplex = np.array([eval_func(simplex[i]) for i in range(n+1)])
        
        stale = 0
        prev_best = np.min(f_simplex)
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.03:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            cur_best = f_simplex[0]
            if cur_best < prev_best - 1e-14:
                stale = 0
                prev_best = cur_best
            else:
                stale += 1
            if stale > 3*n:
                break
            
            centroid = np.mean(simplex[:n], axis=0)
            
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n-1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[n] = xc; f_simplex[n] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                            if time_left() <= 0.03: return
                else:
                    xc = clip(centroid - rho * (centroid - simplex[n]))
                    fc = eval_func(xc)
                    if fc < f_simplex[n]:
                        simplex[n] = xc; f_simplex[n] = fc
                    else:
                        for i in range(1, n+1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                            if time_left() <= 0.03: return
            
            spread = np.max(np.abs(simplex[0] - simplex[-1]))
            if spread < 1e-15:
                break

    # ---- Coordinate-wise local search ----
    def coord_search(x0, time_budget, step_scale=0.01):
        nonlocal best, best_params
        if time_budget <= 0.1:
            return
        t_start = elapsed()
        x = x0.copy()
        fx = eval_func(x)
        step = ranges * step_scale
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.03:
            improved = False
            for j in range(dim):
                if time_left() <= 0.02:
                    return
                # Try positive step
                x_new = x.copy()
                x_new[j] = min(x[j] + step[j], upper[j])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new; fx = f_new; improved = True
                    continue
                # Try negative step
                x_new = x.copy()
                x_new[j] = max(x[j] - step[j], lower[j])
                f_new = eval_func(x_new)
                if f_new < fx:
                    x = x_new; fx = f_new; improved = True
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-14:
                    break

    # ---- Multi-start random probing ----
    def random_probe(n_samples):
        nonlocal best, best_params
        for _ in range(n_samples):
            if time_left() <= 0.05:
                return
            x = np.array([np.random.uniform(l, u) for l, u in bounds])
            eval_func(x)

    # ========== Main Pipeline ==========
    
    # Quick random sampling to get initial best
    random_probe(min(50 * dim, 500))
    
    # Phase 1: L-SHADE global exploration (40% of time)
    run_lshade(time_frac=0.40)
    
    # Phase 2: CMA-ES from best (25% of remaining)
    if best_params is not None and time_left() > 0.5:
        sigma0 = 0.25 * np.mean(ranges)
        cma_es(best_params.copy(), sigma0, time_left() * 0.30)
    
    # Phase 3: Focused L-SHADE around best
    if time_left() > 0.5:
        run_lshade(pop_size_init=min(max(6*dim, 16), 60), time_frac=0.35, focused=True)
    
    # Phase 4: Nelder-Mead polishing
    if best_params is not None and time_left() > 0.5 and dim <= 50:
        nelder_mead(best_params.copy(), time_left() * 0.30, initial_scale=0.01)
    
    # Phase 5: Coordinate search polishing
    if best_params is not None and time_left() > 0.3:
        coord_search(best_params.copy(), time_left() * 0.3, step_scale=0.005)
    
    # Phase 6: CMA-ES refinement restarts with decreasing sigma
    restart_count = 0
    while time_left() > 0.2 and best_params is not None:
        restart_count += 1
        sigma0 = 0.05 * np.mean(ranges) / (restart_count ** 0.5)
        sigma0 = max(sigma0, 1e-6 * np.mean(ranges))
        perturbation = np.random.randn(dim) * ranges * 0.003 / (restart_count**0.3)
        x0 = clip(best_params + perturbation)
        cma_es(x0, sigma0, min(time_left() * 0.4, 2.0))
    
    # Final coordinate search with tiny steps
    if best_params is not None and time_left() > 0.1:
        coord_search(best_params.copy(), time_left() * 0.8, step_scale=0.001)
    
    return best
#
#Key improvements:
#1. **Initial random probing** for better starting points before L-SHADE
#2. **Bounce-back boundary repair** in L-SHADE (randomized, not midpoint)
#3. **Adaptive p-best rate** that decreases over generations
#4. **Coordinate-wise local search** phase for fine-grained polishing
#5. **Stagnation detection** in Nelder-Mead with spread-based convergence
#6. **More aggressive eigendecomposition** frequency in CMA-ES (÷5 instead of ÷10)
#7. **Better time allocation** with more phases and final coordinate search
#8. **Condition number monitoring** in CMA-ES to detect degeneration
