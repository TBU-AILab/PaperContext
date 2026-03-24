#Looking at the previous algorithm's score of 202.34, there are several issues I need to address:
#
#1. **The algorithm is too complex and may have bugs** - simpler, more robust approaches often work better
#2. **Time management could be wasting budget** on unproductive phases
#3. **Need better balance between exploration and exploitation**
#4. **The caching by bytes may not work well with floating point**
#
#Let me redesign with a cleaner, more effective approach:
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
    
    evals = 0
    
    def eval_func(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def lhs_sample(n):
        result = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            result[:, j] = (perm + np.random.uniform(0, 1, n)) / n
        return lower + result * ranges

    # ---- CMA-ES with restarts ----
    def cma_es(x0, sigma0, time_budget, lam_mult=1.0):
        nonlocal best, best_params
        if time_budget <= 0.05:
            return
        t_start = elapsed()
        n = dim
        lam = max(8, int((4 + int(3 * np.log(n))) * lam_mult))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_sep = (n > 40)
        if use_sep:
            diagC = np.ones(n)
            diagD = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            invsqrtC = np.eye(n)
        
        count_eval = 0
        eigeneval = 0
        no_improve = 0
        local_best = best
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.03:
            solutions = []
            f_vals = []
            for _ in range(lam):
                if time_left() <= 0.02:
                    return
                z = np.random.randn(n)
                if use_sep:
                    x = mean + sigma * diagD * z
                else:
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
            
            if use_sep:
                inv_diag = 1.0 / (diagD + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_diag * diff)
                norm_ps = np.linalg.norm(ps)
                hsig = int(norm_ps / np.sqrt(1 - (1-cs)**(2*(count_eval/lam+1))) / chiN < 1.4 + 2/(n+1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                diagC = (1 - c1 - cmu_val) * diagC + \
                    c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC)
                for i in range(mu):
                    artmp = (solutions[idx[i]] - old_mean) / max(sigma, 1e-30)
                    diagC += cmu_val * weights[i] * artmp**2
                diagC = np.maximum(diagC, 1e-20)
                diagD = np.sqrt(diagC)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
                norm_ps = np.linalg.norm(ps)
                hsig = int(norm_ps / np.sqrt(1 - (1-cs)**(2*(count_eval/lam+1))) / chiN < 1.4 + 2/(n+1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
                
                artmp = np.zeros((n, mu))
                for i in range(mu):
                    artmp[:, i] = (solutions[idx[i]] - old_mean) / max(sigma, 1e-30)
                
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * (artmp * weights) @ artmp.T
                
                if count_eval - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
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
                        sigma *= 0.5
            
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2.0 * np.max(ranges))
            
            if best < local_best - 1e-12:
                no_improve = 0
                local_best = best
            else:
                no_improve += 1
            
            if use_sep:
                cond = np.max(diagD) / (np.min(diagD) + 1e-30)
            else:
                cond = np.max(D) / (np.min(D) + 1e-30)
            
            if sigma * (np.max(diagD) if use_sep else np.max(D)) < 1e-15:
                break
            if no_improve > 20 + 3*n:
                break
            if cond > 1e14:
                break

    # ---- L-SHADE ----
    def run_lshade(time_frac=0.5, pop_size_init=None, init_pop=None):
        nonlocal best, best_params
        if time_left() <= 0.15:
            return None, None
        
        budget_time = time_left() * time_frac
        t_start = elapsed()
        
        if pop_size_init is None:
            pop_size_init = min(max(10 * dim, 50), 300)
        pop_size = pop_size_init
        N_init = pop_size_init
        N_min = 4
        
        if init_pop is not None and len(init_pop) >= pop_size:
            population = init_pop[:pop_size].copy()
        else:
            population = lhs_sample(pop_size)
            if best_params is not None:
                population[0] = best_params.copy()
        
        fitness = np.array([eval_func(population[i]) for i in range(pop_size)])
        if time_left() <= 0:
            return population, fitness
        
        H = max(6, min(100, 5 * dim))
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k_idx = 0
        archive = []
        max_archive = pop_size
        gen = 0
        local_best = best
        gens_no_improve = 0
        
        while (elapsed() - t_start) < budget_time and time_left() > 0.08:
            S_F, S_CR, delta_f = [], [], []
            sort_idx = np.argsort(fitness)
            p_best_rate = max(2.0/pop_size, 0.05 + 0.1 * max(0, 1.0 - gen/200.0))
            p_best_size = max(2, int(p_best_rate * pop_size))
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if time_left() <= 0.04:
                    return population, fitness
                
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
                
                # Bounce-back boundary handling
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
            
            if best < local_best - 1e-12:
                gens_no_improve = 0
                local_best = best
            else:
                gens_no_improve += 1
            
            if gens_no_improve > max(50, 10 * dim):
                break
            
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
        
        return population, fitness

    # ---- Nelder-Mead ----
    def nelder_mead(x0, time_budget, scale=0.05):
        nonlocal best, best_params
        if time_budget <= 0.1 or dim > 60:
            return
        t_start = elapsed()
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            delta = scale * ranges[i]
            if delta < 1e-12:
                delta = 1e-6
            simplex[i+1][i] += delta * (1 if np.random.random() > 0.5 else -1)
            simplex[i+1] = clip(simplex[i+1])
        
        f_simplex = np.array([eval_func(simplex[i]) for i in range(n+1)])
        stale = 0
        prev_best = np.min(f_simplex)
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.03:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < prev_best - 1e-14:
                stale = 0
                prev_best = f_simplex[0]
            else:
                stale += 1
            if stale > 2*n + 10:
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
                else:
                    xc = clip(centroid - rho * (centroid - simplex[n]))
                fc = eval_func(xc)
                if fc < max(fr, f_simplex[n]):
                    simplex[n] = xc; f_simplex[n] = fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_func(simplex[i])
                        if time_left() <= 0.03: return
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]))
            if spread < 1e-15:
                break

    # ---- Pattern search / coordinate descent ----
    def pattern_search(x0, time_budget, init_step=0.01):
        nonlocal best, best_params
        if time_budget <= 0.08:
            return
        t_start = elapsed()
        x = x0.copy()
        fx = eval_func(x)
        step = ranges * init_step
        
        while (elapsed() - t_start) < time_budget and time_left() > 0.03:
            improved = False
            perm = np.random.permutation(dim)
            for j in perm:
                if time_left() <= 0.02:
                    return
                best_dir = 0
                best_f = fx
                for sign in [1.0, -1.0]:
                    x_new = x.copy()
                    x_new[j] = np.clip(x[j] + sign * step[j], lower[j], upper[j])
                    f_new = eval_func(x_new)
                    if f_new < best_f:
                        best_f = f_new
                        best_dir = sign
                
                if best_dir != 0:
                    x[j] = np.clip(x[j] + best_dir * step[j], lower[j], upper[j])
                    fx = best_f
                    improved = True
                    # Line search acceleration
                    for k in range(10):
                        x_acc = x.copy()
                        x_acc[j] = np.clip(x[j] + best_dir * step[j], lower[j], upper[j])
                        f_acc = eval_func(x_acc)
                        if f_acc < fx:
                            x = x_acc; fx = f_acc
                        else:
                            break
            if not improved:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-15:
                    break

    # ==== MAIN STRATEGY ====
    
    # Phase 0: Initial sampling with LHS
    n_init = min(max(20 * dim, 100), 500)
    init_pop = lhs_sample(n_init)
    for i in range(n_init):
        if time_left() <= 0.5:
            break
        eval_func(init_pop[i])
    
    # Opposition-based learning
    if time_left() > 1.0:
        n_opp = min(n_init, 200)
        opp = lower + upper - init_pop[:n_opp]
        for i in range(n_opp):
            if time_left() <= 0.5:
                break
            eval_func(opp[i])

    # Phase 1: Main L-SHADE (45% of remaining time)
    if time_left() > 0.5:
        run_lshade(time_frac=0.45)
    
    # Phase 2: CMA-ES from best (20% of remaining time)
    if best_params is not None and time_left() > 0.5:
        sigma0 = 0.2 * np.mean(ranges)
        cma_es(best_params.copy(), sigma0, time_left() * 0.20)
    
    # Phase 3: Focused L-SHADE with small population
    if time_left() > 0.5:
        fp_size = min(max(6*dim, 20), 100)
        focused_init = np.zeros((fp_size, dim))
        if best_params is not None:
            focused_init[0] = best_params.copy()
            for i in range(1, fp_size):
                scale = np.random.uniform(0.001, 0.03)
                focused_init[i] = clip(best_params + np.random.randn(dim) * ranges * scale)
        else:
            focused_init = lhs_sample(fp_size)
        run_lshade(time_frac=0.30, pop_size_init=fp_size, init_pop=focused_init)
    
    # Phase 4: Nelder-Mead polish
    if best_params is not None and time_left() > 0.5 and dim <= 60:
        nelder_mead(best_params.copy(), time_left() * 0.15, scale=0.005)
    
    # Phase 5: Pattern search polish
    if best_params is not None and time_left() > 0.3:
        pattern_search(best_params.copy(), time_left() * 0.15, init_step=0.002)
    
    # Phase 6: IPOP-CMA-ES restarts
    restart = 0
    lam_mult = 1.0
    while time_left() > 0.3 and best_params is not None:
        restart += 1
        
        if restart % 4 == 0:
            # Random restart with large pop
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            sigma0 = 0.3 * np.mean(ranges)
            lam_mult = min(lam_mult * 2, 10)
            cma_es(x0, sigma0, min(time_left() * 0.3, 5.0), lam_mult=lam_mult)
        elif restart % 2 == 0:
            # Medium perturbation from best
            perturb = np.random.randn(dim) * ranges * 0.1
            x0 = clip(best_params + perturb)
            sigma0 = 0.08 * np.mean(ranges)
            cma_es(x0, sigma0, min(time_left() * 0.25, 3.0))
        else:
            # Local refinement
            sigma0 = max(0.01 * np.mean(ranges) / (restart ** 0.3), 1e-8 * np.mean(ranges))
            perturb = np.random.randn(dim) * ranges * 0.003
            x0 = clip(best_params + perturb)
            cma_es(x0, sigma0, min(time_left() * 0.25, 2.0))
    
    # Final fine coordinate descent
    if best_params is not None and time_left() > 0.05:
        pattern_search(best_params.copy(), time_left() * 0.9, init_step=0.0001)
    
    return best
#
#Key improvements:
#1. **Larger initial sampling** (20*dim vs 15*dim) for better initial exploration
#2. **More robust Nelder-Mead** with cleaner contraction logic
#3. **Removed evaluation caching** (byte-key caching can cause issues with floating point and wastes memory)
#4. **Better time allocation** - more time for L-SHADE main phase (45% vs 40%)
#5. **Cleaner IPOP-CMA-ES restarts** with proper lambda multiplier tracking
#6. **Pattern search with proper line search** that continues stepping in successful direction
#7. **Better L-SHADE parameter initialization** (M_F=0.5, M_CR=0.5 - standard values)
#8. **Increased opposition-based samples** and initial population for better coverage
