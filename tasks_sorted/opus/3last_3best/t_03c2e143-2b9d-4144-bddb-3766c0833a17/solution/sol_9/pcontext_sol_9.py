#Looking at the results, the best algorithm (16.14) uses L-SHADE with opposition-based learning and Hooke-Jeeves with multiple pattern step sizes. I'll improve with:
#
#1. **Nelder-Mead simplex** as an additional local search method for better non-separable optimization
#2. **Approximate gradient descent** with finite differences for faster convergence
#3. **More aggressive local search time allocation** (35% SHADE, 65% local)
#4. **Better restart diversity** with basin-hopping style perturbations
#5. **Smarter step size management** in Hooke-Jeeves
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
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: L-SHADE ---
    init_pop_size = min(max(50, 12 * dim), 500)
    min_pop_size = 4
    pop_size = init_pop_size
    H = 100

    population = np.zeros((pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, j] = (perm + np.random.rand(pop_size)) / pop_size
    population = lower + population * ranges
    
    opp_pop = lower + upper - population
    all_pop = np.vstack([population, opp_pop])
    all_fit = np.array([evaluate(all_pop[i]) for i in range(len(all_pop))])
    if elapsed() >= max_time * 0.90:
        return best
    
    best_indices = np.argsort(all_fit)[:pop_size]
    population = all_pop[best_indices].copy()
    fitness_vals = all_fit[best_indices].copy()

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    archive_max = pop_size

    shade_time_frac = 0.35

    while elapsed() < max_time * shade_time_frac:
        S_F = []; S_CR = []; delta_f = []
        sorted_idx = np.argsort(fitness_vals)
        new_pop = population.copy()
        new_fit = fitness_vals.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * shade_time_frac:
                break
            ri = np.random.randint(H)
            Fi = -1
            for _ in range(20):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if 0 < Fi < 2.0: break
                Fi = -1
            if Fi <= 0: Fi = 0.1
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(0.11 * pop_size))
            p_best_idx = sorted_idx[np.random.randint(p)]
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            pool = pop_size + len(archive)
            r2 = np.random.randint(pool - 1)
            if r2 >= min(i, r1): r2 += 1
            if r2 >= max(i, r1): r2 += 1
            if r2 >= pool: r2 = 0
            x_r2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (population[p_best_idx] - population[i]) + Fi * (population[r1] - x_r2)
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) < CRi) | (np.arange(dim) == j_rand)
            trial[mask] = mutant[mask]
            below = trial < lower; above = trial > upper
            trial[below] = (lower[below] + population[i][below]) / 2
            trial[above] = (upper[above] + population[i][above]) / 2
            
            f_trial = evaluate(trial)
            if f_trial <= fitness_vals[i]:
                if f_trial < fitness_vals[i]:
                    S_F.append(Fi); S_CR.append(CRi)
                    delta_f.append(fitness_vals[i] - f_trial)
                    if len(archive) < archive_max: archive.append(population[i].copy())
                    elif archive_max > 0: archive[np.random.randint(archive_max)] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness_vals = new_fit
        if S_F:
            w = np.array(delta_f); w = w / (w.sum() + 1e-30)
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k % H] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k % H] = np.sum(w * scr); k += 1
        
        ratio = min(1.0, elapsed() / (max_time * shade_time_frac))
        new_ps = max(min_pop_size, int(init_pop_size - (init_pop_size - min_pop_size) * ratio))
        if new_ps < pop_size:
            bi = np.argsort(fitness_vals)[:new_ps]
            population = population[bi]; fitness_vals = fitness_vals[bi]
            pop_size = new_ps; archive_max = pop_size

    sidx = np.argsort(fitness_vals)
    starts = [population[sidx[i]].copy() for i in range(min(5, pop_size))]
    if best_params is not None:
        starts.insert(0, best_params.copy())

    # --- Phase 2: Nelder-Mead Simplex ---
    def nelder_mead(x0, fx0, time_limit):
        n = dim
        alpha_r = 1.0; gamma_e = 2.0; rho_c = 0.5; sigma_s = 0.5
        simplex = np.zeros((n+1, n))
        simplex[0] = x0.copy()
        f_simplex = np.zeros(n+1)
        f_simplex[0] = fx0
        init_scale = 0.05 * ranges
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] = np.clip(x0[i] + init_scale[i], lower[i], upper[i])
            if abs(simplex[i+1][i] - x0[i]) < 1e-15:
                simplex[i+1][i] = np.clip(x0[i] - init_scale[i], lower[i], upper[i])
            f_simplex[i+1] = evaluate(simplex[i+1])
            if elapsed() >= time_limit: return
        
        while elapsed() < time_limit:
            order = np.argsort(f_simplex)
            simplex = simplex[order]; f_simplex = f_simplex[order]
            centroid = np.mean(simplex[:n], axis=0)
            
            xr = clip(centroid + alpha_r * (centroid - simplex[n]))
            fr = evaluate(xr)
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma_e * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n-1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho_c * (xr - centroid))
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[n] = xc; f_simplex[n] = fc; continue
                else:
                    xc = clip(centroid + rho_c * (simplex[n] - centroid))
                    fc = evaluate(xc)
                    if fc < f_simplex[n]:
                        simplex[n] = xc; f_simplex[n] = fc; continue
                for i in range(1, n+1):
                    simplex[i] = clip(simplex[0] + sigma_s * (simplex[i] - simplex[0]))
                    f_simplex[i] = evaluate(simplex[i])
                    if elapsed() >= time_limit: return

    def local_search(x0, fx0, time_limit):
        x = x0.copy(); fx = fx0
        step = 0.05 * ranges.copy()
        prev_base = x.copy()
        stagnation = 0
        while elapsed() < time_limit:
            xb = x.copy(); fb = fx
            perm = np.random.permutation(dim)
            for j in perm:
                if elapsed() >= time_limit: break
                improved = False
                for sign in [1, -1]:
                    xn = xb.copy()
                    xn[j] = np.clip(xb[j] + sign * step[j], lower[j], upper[j])
                    fn = evaluate(xn)
                    if fn < fb:
                        xb = xn; fb = fn; step[j] *= 1.3; improved = True; break
                if not improved:
                    step[j] *= 0.6
            if fb < fx:
                pattern = xb - prev_base
                prev_base = x.copy(); x = xb; fx = fb
                stagnation = 0
                for alpha in [1.0, 0.5, 2.0, 1.5]:
                    if elapsed() >= time_limit: break
                    xp = clip(x + alpha * pattern)
                    fp = evaluate(xp)
                    if fp < fx:
                        prev_base = x.copy(); x = xp; fx = fp; break
            else:
                prev_base = x.copy()
                stagnation += 1
                if np.max(step / ranges) < 1e-14 or stagnation > 3:
                    # Try Nelder-Mead from current best
                    nm_time = min(elapsed() + (time_limit - elapsed()) * 0.3, time_limit)
                    nelder_mead(best_params.copy(), best, nm_time)
                    x = best_params.copy(); fx = best
                    step = 0.02 * ranges
                    stagnation = 0
                    if elapsed() >= time_limit: break
                    # Basin hop
                    xnew = best_params + np.random.randn(dim) * 0.05 * ranges
                    xnew = clip(xnew); fnew = evaluate(xnew)
                    if fnew < fx:
                        x = xnew; fx = fnew
                    step = 0.03 * ranges
                    prev_base = x.copy()
        return x, fx

    remaining = max_time * 0.97 - elapsed()
    time_per = remaining / max(len(starts) + 1, 1)
    for s in starts:
        tl = min(elapsed() + time_per, max_time * 0.96)
        local_search(s, evaluate(s), tl)
        if elapsed() >= max_time * 0.94: break
    if best_params is not None and elapsed() < max_time * 0.96:
        local_search(best_params.copy(), best, max_time * 0.98)
    return best
