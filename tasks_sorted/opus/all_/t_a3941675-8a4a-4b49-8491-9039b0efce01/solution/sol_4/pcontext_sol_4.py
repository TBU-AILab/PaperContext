#Looking at the results, algorithm 3 (score 28.52) performed best. Let me analyze what can be improved:
#
#1. **Better SHADE with linear population size reduction (L-SHADE)**
#2. **More aggressive exploitation around best solutions**
#3. **Better CMA-ES restart strategy with warm starting from SHADE population**
#4. **Coordinate-wise local search as final polishing**
#5. **Improved time management**
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
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # Top solutions archive
    top_k = min(20, max(5, dim))
    top_solutions = []
    top_fitnesses = []
    
    def update_top(x, f):
        nonlocal top_solutions, top_fitnesses
        if len(top_solutions) < top_k:
            top_solutions.append(x.copy())
            top_fitnesses.append(f)
        elif f < max(top_fitnesses):
            worst_idx = np.argmax(top_fitnesses)
            top_solutions[worst_idx] = x.copy()
            top_fitnesses[worst_idx] = f
    
    def eval_and_track(x):
        f = eval_func(x)
        update_top(x, f)
        return f
    
    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(25 * dim, 200), 2000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]
    
    init_fitness = np.full(n_init, float('inf'))
    actual_init = 0
    for i in range(n_init):
        if elapsed() > max_time * 0.08:
            break
        init_fitness[i] = eval_and_track(init_pop[i])
        actual_init = i + 1
    
    sorted_idx = np.argsort(init_fitness[:actual_init])
    
    # --- Phase 2: L-SHADE ---
    def lshade_phase(time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        N_init = min(max(8 * dim, 50), 250)
        N_min = 4
        pop_size = N_init
        H = 100
        
        n_elite = min(pop_size // 3, actual_init)
        pop = np.zeros((pop_size, dim))
        fit = np.full(pop_size, float('inf'))
        
        for i in range(n_elite):
            pop[i] = init_pop[sorted_idx[i]].copy()
            fit[i] = init_fitness[sorted_idx[i]]
        for i in range(n_elite, pop_size):
            pop[i] = lower + np.random.random(dim) * ranges
        
        for i in range(n_elite, pop_size):
            if not time_ok() or elapsed() > deadline:
                return pop[:pop_size], fit[:pop_size]
            fit[i] = eval_and_track(pop[i])
        
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        max_archive = N_init
        
        max_evals_estimate = int(time_budget / max(0.001, elapsed() / max(1, eval_count[0])))
        evals_used = 0
        
        gen = 0
        while time_ok(0.96) and elapsed() < deadline:
            gen += 1
            S_F, S_CR, S_df = [], [], []
            
            new_pop = pop[:pop_size].copy()
            new_fit = fit[:pop_size].copy()
            
            for i in range(pop_size):
                if not time_ok(0.96) or elapsed() > deadline:
                    return new_pop[:pop_size], new_fit[:pop_size]
                
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
                
                # current-to-pbest/1
                p_rate = max(2.0/pop_size, 0.05 + 0.15 * (1 - evals_used / max(1, max_evals_estimate)))
                p = max(2, int(pop_size * p_rate))
                pbest_idx = np.argsort(fit[:pop_size])[:p]
                xpbest = pop[pbest_idx[np.random.randint(p)]]
                
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                pool_size = pop_size + len(archive)
                r2 = np.random.randint(pool_size)
                attempts = 0
                while (r2 == i or r2 == r1) and attempts < 20:
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
                
                f_trial = eval_and_track(trial)
                evals_used += 1
                
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
            
            pop[:pop_size] = new_pop[:pop_size]
            fit[:pop_size] = new_fit[:pop_size]
            
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
            
            # Linear population size reduction
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * evals_used / max(1, max_evals_estimate))))
            if new_pop_size < pop_size:
                best_indices = np.argsort(fit[:pop_size])[:new_pop_size]
                pop[:new_pop_size] = pop[best_indices]
                fit[:new_pop_size] = fit[best_indices]
                pop_size = new_pop_size
        
        return pop[:pop_size], fit[:pop_size]
    
    # --- CMA-ES ---
    def cma_es_run(x0, sigma0, time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        
        n = dim
        lam = 4 + int(3 * np.log(n))
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
                fitnesses[ki] = eval_and_track(arx[ki])
            
            idx = np.argsort(fitnesses)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / sigma
            
            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = ((1 - c1 - cmu_val) * C +
                 c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) +
                 cmu_val * np.sum(weights[:, None, None] * (artmp[:, :, None] * artmp[:, None, :]), axis=0))
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * np.max(ranges))
            
            gen += 1
            if gen % max(1, int(1.0 / (c1 + cmu_val + 1e-20) / n / 10)) == 0:
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
            f_s[i] = eval_and_track(simplex[i])
        
        while time_ok(0.96) and elapsed() < deadline:
            order = np.argsort(f_s)
            simplex = simplex[order]
            f_s = f_s[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + 1.0*(centroid - simplex[-1]))
            fr = eval_and_track(xr)
            
            if f_s[0] <= fr < f_s[-2]:
                simplex[-1] = xr; f_s[-1] = fr
            elif fr < f_s[0]:
                xe = clip(centroid + 2.0*(xr - centroid))
                if not time_ok(0.96) or elapsed() > deadline: break
                fe = eval_and_track(xe)
                if fe < fr: simplex[-1] = xe; f_s[-1] = fe
                else: simplex[-1] = xr; f_s[-1] = fr
            else:
                xc = clip(centroid + 0.5*(simplex[-1] - centroid))
                if not time_ok(0.96) or elapsed() > deadline: break
                fc = eval_and_track(xc)
                if fc < f_s[-1]: simplex[-1] = xc; f_s[-1] = fc
                else:
                    for i in range(1, n+1):
                        if not time_ok(0.96) or elapsed() > deadline: return
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i])
                        f_s[i] = eval_and_track(simplex[i])
            if np.max(np.abs(f_s - f_s[0])) < 1e-15: break
    
    # --- Coordinate descent local search ---
    def coord_descent(x0, time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        x = x0.copy()
        fx = eval_and_track(x)
        
        step_sizes = 0.01 * ranges.copy()
        
        while time_ok(0.98) and elapsed() < deadline:
            improved = False
            for d_i in range(dim):
                if not time_ok(0.98) or elapsed() > deadline:
                    return
                
                # Try positive step
                x_new = x.copy()
                x_new[d_i] = min(x[d_i] + step_sizes[d_i], upper[d_i])
                f_new = eval_and_track(x_new)
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    step_sizes[d_i] *= 1.2
                    improved = True
                    continue
                
                # Try negative step
                x_new = x.copy()
                x_new[d_i] = max(x[d_i] - step_sizes[d_i], lower[d_i])
                if not time_ok(0.98) or elapsed() > deadline:
                    return
                f_new = eval_and_track(x_new)
                
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    step_sizes[d_i] *= 1.2
                    improved = True
                else:
                    step_sizes[d_i] *= 0.5
            
            if not improved or np.max(step_sizes) < 1e-14 * np.max(ranges):
                break
    
    # === Main orchestration ===
    rem = max_time - elapsed()
    
    # L-SHADE phase: 35% of remaining time
    shade_pop, shade_fit = lshade_phase(rem * 0.35)
    
    # CMA-ES restarts: ~50% of total time
    restart = 0
    while time_ok(0.85):
        rem = max_time * 0.85 - elapsed()
        if rem < 0.2: break
        n_left = max(1, min(6, int(rem / max(0.3, rem/5))))
        t_budget = rem / n_left * 0.85
        
        if restart == 0:
            x0 = best_params.copy() if best_params is not None else lower + np.random.random(dim)*ranges
            sig = 0.2 * np.max(ranges)
        elif restart % 4 == 0:
            # Full random restart
            x0 = lower + np.random.random(dim)*ranges
            sig = 0.5 * np.max(ranges)
        elif restart % 4 == 1 and len(top_solutions) > 1:
            # Start from a diverse top solution
            idx_t = restart % len(top_solutions)
            x0 = top_solutions[idx_t].copy()
            sig = 0.15 * np.max(ranges)
        else:
            x0 = clip(best_params + 0.1*ranges*np.random.randn(dim))
            sig = 0.1 * np.max(ranges) * (1 + restart*0.2)
        
        cma_es_run(x0, min(sig, np.max(ranges)), t_budget)
        restart += 1
    
    # Nelder-Mead refinement
    if time_ok(0.88) and best_params is not None:
        nelder_mead(best_params, 0.02, (max_time*0.93 - elapsed()) * 0.5)
    if time_ok(0.93) and best_params is not None:
        nelder_mead(best_params, 0.003, (max_time*0.96 - elapsed()) * 0.6)
    
    # Final coordinate descent polish
    if time_ok(0.96) and best_params is not None:
        coord_descent(best_params, max_time*0.99 - elapsed())
    
    return best
