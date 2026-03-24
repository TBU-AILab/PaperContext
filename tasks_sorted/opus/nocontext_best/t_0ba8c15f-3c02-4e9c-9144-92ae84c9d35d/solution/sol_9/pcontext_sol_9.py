#Looking at the code, I can identify several areas for improvement:
#
#1. **Better initial sampling** - use more structured exploration
#2. **Improved SHADE-DE** - better parameter adaptation and population management
#3. **More aggressive CMA-ES restarts** with proper time management
#4. **Add a local search phase using Powell's method pattern**
#5. **Fix time allocation** - give more time to CMA-ES which is the strongest component
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
    evals = 0

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction

    def clip(x):
        return np.clip(x, lower, upper)

    def eval_f(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(max(80 * dim, 500), 3000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]

    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.08):
            break
        init_fitness[i] = eval_f(init_pop[i])

    valid = init_fitness < float('inf')
    valid_pop = init_pop[valid]
    valid_fit = init_fitness[valid]
    sorted_idx = np.argsort(valid_fit)

    n_archive = min(50, len(sorted_idx))
    archive_x = [valid_pop[sorted_idx[i]].copy() for i in range(n_archive)]
    archive_f = [valid_fit[sorted_idx[i]] for i in range(n_archive)]

    # --- Phase 2: SHADE Differential Evolution ---
    pop_size = min(max(10 * dim, 60), 200)
    n_elite_de = min(pop_size, len(valid_pop))

    de_pop = np.zeros((pop_size, dim))
    de_fit = np.full(pop_size, float('inf'))

    top_idx = sorted_idx[:n_elite_de]
    for i in range(n_elite_de):
        de_pop[i] = valid_pop[top_idx[i]].copy()
        de_fit[i] = valid_fit[top_idx[i]]

    for i in range(n_elite_de, pop_size):
        de_pop[i] = lower + np.random.random(dim) * ranges
        if time_ok(0.12):
            de_fit[i] = eval_f(de_pop[i])

    # SHADE memory
    H = 20
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    de_archive_pop = []

    de_end = 0.40
    gen_count = 0
    while time_ok(de_end):
        S_F = []
        S_CR = []
        delta_f = []

        # Linearly decrease p_min
        progress = min(1.0, elapsed() / (max_time * de_end))
        p_ratio = max(0.05, 0.2 - 0.15 * progress)

        for i in range(pop_size):
            if not time_ok(de_end):
                break

            ri = np.random.randint(H)
            F = -1
            while F <= 0:
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
            F = min(F, 1.0)
            
            CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

            p = max(2, int(p_ratio * pop_size))
            p_best_idx = np.argsort(de_fit)[:p]
            x_pbest = de_pop[np.random.choice(p_best_idx)]

            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            idxs.remove(r1)

            if len(de_archive_pop) > 0 and np.random.random() < len(de_archive_pop) / (len(idxs) + len(de_archive_pop)):
                x_r2 = de_archive_pop[np.random.randint(len(de_archive_pop))]
            else:
                x_r2 = de_pop[np.random.choice(idxs)]

            mutant = de_pop[i] + F * (x_pbest - de_pop[i]) + F * (de_pop[r1] - x_r2)

            trial = de_pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]

            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + de_pop[i][d]) / 2.0
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + de_pop[i][d]) / 2.0
            trial = clip(trial)

            f_trial = eval_f(trial)

            if f_trial <= de_fit[i]:
                if f_trial < de_fit[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(abs(de_fit[i] - f_trial))
                    de_archive_pop.append(de_pop[i].copy())
                    if len(de_archive_pop) > pop_size:
                        de_archive_pop.pop(np.random.randint(len(de_archive_pop)))
                de_pop[i] = trial
                de_fit[i] = f_trial

        if len(S_F) > 0:
            w = np.array(delta_f)
            w = w / (np.sum(w) + 1e-30)
            S_F_arr = np.array(S_F)
            S_CR_arr = np.array(S_CR)
            M_F[k] = np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
            M_CR[k] = np.sum(w * S_CR_arr)
            k = (k + 1) % H
        
        gen_count += 1

    # Update archive from DE
    for i in range(pop_size):
        if de_fit[i] < max(archive_f):
            worst_idx = int(np.argmax(archive_f))
            archive_x[worst_idx] = de_pop[i].copy()
            archive_f[worst_idx] = de_fit[i]

    # --- Phase 3: CMA-ES with restarts (IPOP-style) ---
    def cmaes_search(x0, sigma0, time_budget, lam_mult=1):
        nonlocal best, best_params
        t_start = elapsed()
        deadline = t_start + time_budget
        n = dim
        lam = max(lam_mult * (4 + int(3 * np.log(n))), 6)
        mu_cma = lam // 2
        weights = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs

        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))

        mean = clip(x0.copy())
        sigma = sigma0
        stagnation = 0
        prev_best_gen = float('inf')
        f_history = []
        best_f_seen = float('inf')
        no_improve_count = 0

        generation = 0
        while elapsed() < deadline and time_ok(0.94):
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for kk in range(lam):
                arx[kk] = mean + sigma * (B @ (D * arz[kk]))
                arx[kk] = clip(arx[kk])

            fitnesses = np.zeros(lam)
            for kk in range(lam):
                if elapsed() >= deadline or not time_ok(0.94):
                    return best_f_seen
                fitnesses[kk] = eval_f(arx[kk])

            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]
            fitnesses_sorted = fitnesses[idx]

            best_gen = fitnesses_sorted[0]
            f_history.append(best_gen)
            
            if best_gen < best_f_seen:
                best_f_seen = best_gen
                no_improve_count = 0
            else:
                no_improve_count += 1

            if best_gen < prev_best_gen - 1e-12:
                stagnation = 0
                prev_best_gen = best_gen
            else:
                stagnation += 1

            if stagnation > 10 + 30 * n / lam:
                return best_f_seen
            if no_improve_count > 20 + 40 * n / lam:
                return best_f_seen
            if len(f_history) > 30 and max(f_history[-30:]) - min(f_history[-30:]) < 1e-13 * (abs(best_f_seen) + 1e-30):
                return best_f_seen

            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu_cma], axis=0)

            mean_diff = mean - old_mean
            zmean = np.sum(weights[:, None] * arz[:mu_cma], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / (sigma + 1e-30)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(generation+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / (sigma + 1e-30)

            artmp = (arx[:mu_cma] - old_mean) / (sigma + 1e-30)
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * (artmp.T @ np.diag(weights) @ artmp)

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 2 * np.max(ranges))

            eigeneval += lam
            if eigeneval >= lam / (c1 + cmu_v + 1e-20) / n / 10:
                eigeneval = 0
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                except:
                    return best_f_seen

            if max(D) / (min(D) + 1e-30) > 1e14:
                return best_f_seen
            generation += 1
        
        return best_f_seen

    # Run CMA-ES with IPOP restarts
    n_cma = 0
    base_sigma = 0.3 * np.mean(ranges)
    sorted_archive = sorted(zip(archive_f, archive_x))
    lam_mult = 1
    
    while time_ok(0.88):
        remaining = max_time * 0.88 - elapsed()
        if remaining < 0.3:
            break
            
        n_restarts_left = max(1, min(6, int(remaining / 1.0)))
        time_per_run = max(0.5, remaining / n_restarts_left)
        
        if n_cma < min(3, len(sorted_archive)):
            x0 = sorted_archive[n_cma][1].copy()
            sigma0 = base_sigma * 0.15
        elif n_cma < min(6, len(sorted_archive)):
            x0 = sorted_archive[n_cma][1].copy()
            sigma0 = base_sigma * 0.3
        elif best_params is not None:
            # IPOP: increase population
            perturbation = np.random.normal(0, 1, dim) * ranges * 0.2 * (1 + n_cma * 0.1)
            x0 = clip(best_params + perturbation)
            sigma0 = base_sigma * min(2.0, 0.3 + 0.15 * n_cma)
            lam_mult = min(lam_mult * 2, 8)
        else:
            x0 = lower + np.random.random(dim) * ranges
            sigma0 = base_sigma

        cmaes_search(x0, sigma0, time_per_run, lam_mult=lam_mult if n_cma >= 6 else 1)
        n_cma += 1

    # --- Phase 4: Nelder-Mead polishing ---
    if best_params is not None and time_ok(0.96):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step_size = 0.001 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step_size[i] if best_params[i] + step_size[i] <= upper[i] else -step_size[i]

        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if not time_ok(0.96):
                f_simplex[i] = float('inf')
            else:
                f_simplex[i] = eval_f(simplex[i])

        nm_iters = 0
        while time_ok(0.98) and nm_iters < 2000:
            nm_iters += 1
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            centroid = np.mean(simplex[:n], axis=0)

            # Reflection
            xr = clip(centroid + 1.0 * (centroid - simplex[n]))
            fr = eval_f(xr)

            if fr < f_simplex[0]:
                xe = clip(centroid + 2.0 * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[n], f_simplex[n] = xe, fe
                else:
                    simplex[n], f_simplex[n] = xr, fr
            elif fr < f_simplex[n - 1]:
                simplex[n], f_simplex[n] = xr, fr
            else:
                if fr < f_simplex[n]:
                    xc = clip(centroid + 0.5 * (xr - centroid))
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[n], f_simplex[n] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.98): break
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_f(simplex[i])
                else:
                    xc = clip(centroid + 0.5 * (simplex[n] - centroid))
                    fc = eval_f(xc)
                    if fc < f_simplex[n]:
                        simplex[n], f_simplex[n] = xc, fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.98): break
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_f(simplex[i])

            if np.max(np.abs(simplex[n] - simplex[0]) / (ranges + 1e-30)) < 1e-15:
                break

    # --- Phase 5: Pattern search / coordinate descent ---
    if best_params is not None and time_ok(0.99):
        x = best_params.copy()
        fx = best
        step = 0.001 * ranges
        while time_ok(0.997):
            improved = False
            for d in range(dim):
                if not time_ok(0.997): break
                for direction in [1.0, -1.0]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    ft = eval_f(trial)
                    if ft < fx:
                        x = trial
                        fx = ft
                        improved = True
                        # Accelerate in this direction
                        while time_ok(0.997):
                            trial2 = x.copy()
                            trial2[d] = np.clip(trial2[d] + direction * step[d], lower[d], upper[d])
                            ft2 = eval_f(trial2)
                            if ft2 < fx:
                                x = trial2
                                fx = ft2
                            else:
                                break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-16:
                    break

    return best
