#Looking at the results, algorithm 2 (score 36.62) and algorithm 4 (score 28.52) performed best. I'll build on these with key improvements:
#
#1. **Better exploration-exploitation balance** with adaptive time allocation
#2. **Improved L-SHADE with population size reduction**
#3. **Smarter CMA-ES restarts** with BIPOP strategy (alternating small/large populations)
#4. **Opposition-based learning** for diversity
#5. **More efficient local search** combining Nelder-Mead and Powell-like coordinate search
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
        return max_time - elapsed()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    eval_time_sum = [0.0]
    
    top_k = min(30, max(8, dim))
    top_solutions = []
    top_fitnesses = []
    
    def update_top(x, f):
        if len(top_solutions) < top_k:
            top_solutions.append(x.copy())
            top_fitnesses.append(f)
        elif f < max(top_fitnesses):
            idx = np.argmax(top_fitnesses)
            top_solutions[idx] = x.copy()
            top_fitnesses[idx] = f
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        t0 = elapsed()
        f = func(x)
        dt = elapsed() - t0
        eval_count[0] += 1
        eval_time_sum[0] += dt
        if f < best:
            best = f
            best_params = x.copy()
        update_top(x, f)
        return f
    
    def avg_eval_time():
        if eval_count[0] == 0:
            return 0.001
        return eval_time_sum[0] / eval_count[0]
    
    # --- Phase 1: Latin Hypercube Sampling with opposition ---
    n_init = min(max(20 * dim, 150), 1500)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    actual_init = 0
    for i in range(n_init):
        if elapsed() > max_time * 0.07:
            break
        init_fitness[i] = eval_func(init_pop[i])
        actual_init = i + 1
    
    # Opposition-based learning on best samples
    if actual_init > 10 and time_ok(0.10):
        sorted_init = np.argsort(init_fitness[:actual_init])
        n_opp = min(actual_init // 3, 50)
        for i in range(n_opp):
            if elapsed() > max_time * 0.10:
                break
            idx = sorted_init[i]
            opp = lower + upper - init_pop[idx]
            opp = clip(opp)
            f_opp = eval_func(opp)
            if f_opp < init_fitness[idx]:
                init_pop[idx] = opp.copy()
                init_fitness[idx] = f_opp
    
    sorted_idx = np.argsort(init_fitness[:actual_init])
    
    # --- Phase 2: L-SHADE ---
    def lshade_phase(time_budget):
        nonlocal best, best_params
        if time_budget < 0.1:
            return
        deadline = elapsed() + time_budget
        
        N_init = min(max(8 * dim, 50), 250)
        N_min = 4
        pop_size = N_init
        H = 100
        
        n_elite = min(pop_size // 3, actual_init)
        pop = np.zeros((N_init, dim))
        fit = np.full(N_init, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, N_init):
            pop[i] = lower + np.random.random(dim) * ranges
        
        for i in range(n_elite, N_init):
            if not time_ok(0.96) or elapsed() > deadline:
                return
            fit[i] = eval_func(pop[i])
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        max_archive = N_init
        
        # Estimate total evals we can do
        est_eval_time = avg_eval_time()
        max_evals_est = int(time_budget / max(est_eval_time, 1e-6))
        evals_in_shade = 0
        
        gen = 0
        while time_ok(0.96) and elapsed() < deadline:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            
            for i in range(pop_size):
                if not time_ok(0.96) or elapsed() > deadline:
                    pop[:pop_size] = new_pop
                    fit[:pop_size] = new_fit
                    return
                
                r_i = np.random.randint(0, H)
                
                for _ in range(10):
                    Fi = M_F[r_i] + 0.1 * np.random.standard_cauchy()
                    if Fi > 0:
                        break
                Fi = min(max(Fi, 0.01), 1.0)
                
                CRi = M_CR[r_i]
                if CRi < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(CRi + 0.1 * np.random.randn(), 0, 1)
                
                # Adaptive p
                progress = evals_in_shade / max(1, max_evals_est)
                p_rate = max(2.0/pop_size, 0.05 + 0.20 * (1 - progress))
                p = max(2, int(pop_size * p_rate))
                pbest_idx = np.argsort(fit[:pop_size])[:p]
                xpbest = pop[pbest_idx[np.random.randint(p)]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                attempts = 0
                while (r2 == i or r2 == r1) and attempts < 25:
                    r2 = np.random.randint(pool_size)
                    attempts += 1
                xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]
                
                mutant = pop[i] + Fi * (xpbest - pop[i]) + Fi * (pop[r1] - xr2)
                
                for d_i in range(dim):
                    if mutant[d_i] < lower[d_i]:
                        mutant[d_i] = (lower[d_i] + pop[i, d_i]) / 2
                    elif mutant[d_i] > upper[d_i]:
                        mutant[d_i] = (upper[d_i] + pop[i, d_i]) / 2
                
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[jrand] = True
                trial[mask] = mutant[mask]
                trial = clip(trial)
                
                f_trial = eval_func(trial)
                evals_in_shade += 1
                
                if f_trial < fit[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(fit[i] - f_trial)
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = f_trial
                elif f_trial == fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop[:pop_size] = new_pop
            fit[:pop_size] = new_fit
            
            if len(S_F) > 0:
                S_df_arr = np.array(S_df)
                w = S_df_arr / (np.sum(S_df_arr) + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)
                M_F[k] = np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
                if np.max(S_CR_arr) <= 0:
                    M_CR[k] = -1.0
                else:
                    M_CR[k] = np.sum(w * S_CR_arr)
                k = (k + 1) % H
            
            # Population size reduction
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * evals_in_shade / max(1, max_evals_est))))
            if new_pop_size < pop_size:
                best_indices = np.argsort(fit[:pop_size])[:new_pop_size]
                pop[:new_pop_size] = pop[best_indices]
                fit[:new_pop_size] = fit[best_indices]
                pop_size = new_pop_size
    
    # --- CMA-ES ---
    def cma_es_run(x0, sigma0, time_budget, pop_mult=1):
        nonlocal best, best_params
        if time_budget < 0.05:
            return
        deadline = elapsed() + time_budget
        
        n = dim
        lam = max(4 + int(3 * np.log(n)), 6) * pop_mult
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n*n))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        
        mean = x0.copy()
        sigma = sigma0
        gen = 0
        no_improve = 0
        prev_best = best
        eigfreq = max(1, int(1.0 / (c1 + cmu_val + 1e-20) / n / 10))
        
        while time_ok(0.96) and elapsed() < deadline:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for ki in range(lam):
                arx[ki] = mean + sigma * (B @ (D * arz[ki]))
                for d_i in range(n):
                    if arx[ki, d_i] < lower[d_i]:
                        arx[ki, d_i] = lower[d_i] + np.random.random() * min(ranges[d_i]*0.01, abs(mean[d_i]-lower[d_i])+1e-20)
                    elif arx[ki, d_i] > upper[d_i]:
                        arx[ki, d_i] = upper[d_i] - np.random.random() * min(ranges[d_i]*0.01, abs(upper[d_i]-mean[d_i])+1e-20)
                arx[ki] = clip(arx[ki])
            
            fitnesses = np.zeros(lam)
            for ki in range(lam):
                if not time_ok(0.96) or elapsed() > deadline:
                    return
                fitnesses[ki] = eval_func(arx[ki])
            
            idx = np.argsort(fitnesses)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / max(sigma, 1e-30)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / max(sigma, 1e-30)
            
            artmp = (arx[idx[:mu]] - old_mean) / max(sigma, 1e-30)
            C = ((1 - c1 - cmu_val) * C +
                 c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                 cmu_val * np.sum(weights[:, None, None] * (artmp[:, :, None] * artmp[:, None, :]), axis=0))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            if gen % eigfreq == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, Bn = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    B = Bn
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
            
            if best < prev_best - 1e-10:
                no_improve = 0
                prev_best = best
            else:
                no_improve += 1
            if sigma * np.max(D) < 1e-13 * np.max(ranges) or no_improve > 30 + 10*n:
                break
    
    # --- Nelder-Mead ---
    def nelder_mead(x0, scale_factor, time_budget):
        nonlocal best, best_params
        if time_budget < 0.05:
            return
        deadline = elapsed() + time_budget
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1, i] += scale[i] * (1 if np.random.random() < 0.5 else -1)
            simplex[i+1] = clip(simplex[i+1])
        
        f_s = np.zeros(n + 1)
        for i in range(n+1):
            if not time_ok(0.96) or elapsed() > deadline:
                return
            f_s[i] = eval_func(simplex[i])
        
        max_iter = 2000
        it = 0
        while time_ok(0.97) and elapsed() < deadline and it < max_iter:
            it += 1
            order = np.argsort(f_s)
            simplex = simplex[order]
            f_s = f_s[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + 1.0*(centroid - simplex[-1]))
            fr = eval_func(xr)
            
            if f_s[0] <= fr < f_s[-2]:
                simplex[-1] = xr; f_s[-1] = fr
            elif fr < f_s[0]:
                xe = clip(centroid + 2.0*(xr - centroid))
                if not time_ok(0.97) or elapsed() > deadline: break
                fe = eval_func(xe)
                if fe < fr: simplex[-1] = xe; f_s[-1] = fe
                else: simplex[-1] = xr; f_s[-1] = fr
            else:
                if fr < f_s[-1]:
                    xc = clip(centroid + 0.5*(xr - centroid))
                else:
                    xc = clip(centroid + 0.5*(simplex[-1] - centroid))
                if not time_ok(0.97) or elapsed() > deadline: break
                fc = eval_func(xc)
                if fc < min(fr, f_s[-1]):
                    simplex[-1] = xc; f_s[-1] = fc
                else:
                    for i in range(1, n+1):
                        if not time_ok(0.97) or elapsed() > deadline: return
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i])
                        f_s[i] = eval_func(simplex[i])
            if np.max(np.abs(f_s - f_s[0])) < 1e-16: break
    
    # --- Powell-like coordinate descent ---
    def powell_search(x0, time_budget):
        nonlocal best, best_params
        if time_budget < 0.05:
            return
        deadline = elapsed() + time_budget
        x = x0.copy()
        fx = eval_func(x)
        
        directions = np.eye(dim)
        
        while time_ok(0.98) and elapsed() < deadline:
            x_start = x.copy()
            fx_start = fx
            deltas = np.zeros(dim)
            
            for d_i in range(dim):
                if not time_ok(0.98) or elapsed() > deadline:
                    return
                
                direction = directions[d_i]
                
                # Golden section search along direction
                a_lo, a_hi = -0.5 * ranges[d_i], 0.5 * ranges[d_i]
                gr = (np.sqrt(5) + 1) / 2
                
                c = a_hi - (a_hi - a_lo) / gr
                d = a_lo + (a_hi - a_lo) / gr
                
                for _ in range(12):
                    if not time_ok(0.98) or elapsed() > deadline:
                        return
                    xc = clip(x + c * direction)
                    xd = clip(x + d * direction)
                    fc = eval_func(xc)
                    fd = eval_func(xd)
                    
                    if fc < fd:
                        a_hi = d
                    else:
                        a_lo = c
                    c = a_hi - (a_hi - a_lo) / gr
                    d = a_lo + (a_hi - a_lo) / gr
                    
                    if abs(a_hi - a_lo) < 1e-8 * ranges[d_i]:
                        break
                
                alpha_best = (a_lo + a_hi) / 2
                x_new = clip(x + alpha_best * direction)
                f_new = eval_func(x_new)
                if f_new < fx:
                    deltas[d_i] = abs(fx - f_new)
                    x = x_new
                    fx = f_new
            
            # Update directions (simplified Powell)
            improvement = fx_start - fx
            if improvement < 1e-15:
                break
            
            if np.sum(deltas) > 0:
                worst_dir = np.argmax(deltas)
                new_dir = x - x_start
                norm = np.linalg.norm(new_dir)
                if norm > 1e-20:
                    new_dir /= norm
                    directions[worst_dir] = new_dir
    
    # --- Main orchestration ---
    rem = max_time - elapsed()
    
    # L-SHADE: 30% time
    lshade_phase(rem * 0.30)
    
    # CMA-ES with BIPOP restarts: ~50% time
    restart = 0
    small_budget_total = 0
    large_budget_total = 0
    
    while time_ok(0.82):
        rem = max_time * 0.82 - elapsed()
        if rem < 0.15: break
        n_left = max(1, min(8, int(rem / max(0.2, rem/6))))
        t_budget = rem / n_left * 0.9
        
        if restart == 0:
            x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim)*ranges
            sig = 0.2 * np.max(ranges)
            pop_mult = 1
        elif restart % 2 == 1:
            # Small population, local restart near best
            if len(top_solutions) > 1:
                idx_t = np.random.randint(min(5, len(top_solutions)))
                sorted_top = np.argsort(top_fitnesses)
                x0 = top_solutions[sorted_top[idx_t]].copy()
            else:
                x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim)*ranges
            x0 = clip(x0 + 0.05*ranges*np.random.randn(dim))
            sig = 0.08 * np.max(ranges)
            pop_mult = 1
        else:
            # Large population, global restart
            x0 = lower + np.random.random(dim)*ranges
            sig = 0.4 * np.max(ranges)
            pop_mult = min(2, max(1, restart // 2))
        
        cma_es_run(x0, min(sig, np.max(ranges)), t_budget, pop_mult)
        restart += 1
    
    # Nelder-Mead refinement from multiple top solutions
    if time_ok(0.85) and len(top_solutions) > 0:
        sorted_top = np.argsort(top_fitnesses)
        nm_runs = min(3, len(top_solutions))
        for r in range(nm_runs):
            if not time_ok(0.90): break
            rem = max_time * 0.93 - elapsed()
            if rem < 0.1: break
            x0 = top_solutions[sorted_top[r]].copy()
            nelder_mead(x0, 0.02 / (1 + r), rem / (nm_runs - r) * 0.8)
    
    # Nelder-Mead fine polish from best
    if time_ok(0.93) and best_params is not None:
        nelder_mead(best_params, 0.003, (max_time*0.96 - elapsed()) * 0.5)
    
    # Powell search polish
    if time_ok(0.96) and best_params is not None:
        powell_search(best_params, max_time*0.99 - elapsed())
    
    return best
