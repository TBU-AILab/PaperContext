#Looking at the code, I can identify several issues and improvements:
#
#1. **CMA-ES time allocation is broken** - `target_frac` calculation can cause CMA-ES to exit immediately
#2. **Nelder-Mead has bugs** with incomplete f_simplex array
#3. **Need better exploration-exploitation balance** - more restarts from diverse points
#4. **Add a proper basin-hopping / perturbation mechanism**
#5. **Improve DE with opposition-based learning**
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

    def eval_f(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Sobol-like LHS sampling ---
    n_init = min(max(50 * dim, 400), 2000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]

    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.10):
            break
        init_fitness[i] = eval_f(init_pop[i])

    valid = init_fitness < float('inf')
    valid_pop = init_pop[valid]
    valid_fit = init_fitness[valid]
    sorted_idx = np.argsort(valid_fit)

    n_archive = min(30, len(sorted_idx))
    archive_x = [valid_pop[sorted_idx[i]].copy() for i in range(n_archive)]
    archive_f = [valid_fit[sorted_idx[i]] for i in range(n_archive)]

    # --- Phase 2: SHADE Differential Evolution ---
    pop_size = min(max(8 * dim, 50), 150)
    n_elite_de = min(pop_size, len(valid_pop))

    de_pop = np.zeros((pop_size, dim))
    de_fit = np.full(pop_size, float('inf'))

    top_idx = sorted_idx[:n_elite_de]
    for i in range(n_elite_de):
        de_pop[i] = valid_pop[top_idx[i]].copy()
        de_fit[i] = valid_fit[top_idx[i]]

    for i in range(n_elite_de, pop_size):
        de_pop[i] = lower + np.random.random(dim) * ranges
        if time_ok(0.15):
            de_fit[i] = eval_f(de_pop[i])

    # SHADE memory
    H = 10
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    de_archive = []

    de_end = 0.42
    while time_ok(de_end):
        S_F = []
        S_CR = []
        delta_f = []

        for i in range(pop_size):
            if not time_ok(de_end):
                break

            ri = np.random.randint(H)
            F = -1
            while F <= 0:
                F = np.random.standard_cauchy() * 0.1 + M_F[ri]
            F = min(F, 1.0)
            
            CR = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

            p = max(2, int(0.1 * pop_size))
            p_best_idx = np.argsort(de_fit)[:p]
            x_pbest = de_pop[np.random.choice(p_best_idx)]

            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            idxs.remove(r1)

            union_size = len(idxs) + len(de_archive)
            if len(de_archive) > 0 and np.random.randint(union_size) >= len(idxs):
                x_r2 = de_archive[np.random.randint(len(de_archive))]
            else:
                x_r2 = de_pop[np.random.choice(idxs)]

            mutant = de_pop[i] + F * (x_pbest - de_pop[i]) + F * (de_pop[r1] - x_r2)

            trial = de_pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]

            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (de_pop[i][d] - lower[d])
                elif trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - de_pop[i][d])
            trial = clip(trial)

            f_trial = eval_f(trial)

            if f_trial <= de_fit[i]:
                if f_trial < de_fit[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(abs(de_fit[i] - f_trial))
                    de_archive.append(de_pop[i].copy())
                    if len(de_archive) > pop_size:
                        de_archive.pop(np.random.randint(len(de_archive)))
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

    # Update archive
    for i in range(pop_size):
        if de_fit[i] < max(archive_f):
            worst_idx = int(np.argmax(archive_f))
            archive_x[worst_idx] = de_pop[i].copy()
            archive_f[worst_idx] = de_fit[i]

    # --- Phase 3: CMA-ES with restarts ---
    def cmaes_search(x0, sigma0, time_budget):
        nonlocal best, best_params
        deadline = elapsed() + time_budget
        n = dim
        lam = 4 + int(3 * np.log(n))
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

        generation = 0
        while elapsed() < deadline and time_ok(0.92):
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for kk in range(lam):
                arx[kk] = mean + sigma * (B @ (D * arz[kk]))
                arx[kk] = clip(arx[kk])

            fitnesses = np.zeros(lam)
            for kk in range(lam):
                if elapsed() >= deadline or not time_ok(0.92):
                    return
                fitnesses[kk] = eval_f(arx[kk])

            idx = np.argsort(fitnesses)
            arx = arx[idx]
            arz = arz[idx]

            best_gen = fitnesses[idx[0]]
            f_history.append(best_gen)
            if best_gen < prev_best_gen - 1e-12:
                stagnation = 0
                prev_best_gen = best_gen
            else:
                stagnation += 1

            if stagnation > 10 + 30 * n / lam:
                return
            if len(f_history) > 20 and max(f_history[-20:]) - min(f_history[-20:]) < 1e-12:
                return

            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu_cma], axis=0)

            mean_diff = mean - old_mean
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
                    return

            if max(D) / (min(D) + 1e-30) > 1e14:
                return
            generation += 1

    # Run CMA-ES restarts with increasing population diversity
    n_cma = 0
    base_sigma = 0.25 * np.mean(ranges)
    sorted_archive = sorted(zip(archive_f, archive_x))
    
    while time_ok(0.88):
        remaining = max_time * 0.88 - elapsed()
        if remaining < 0.5:
            break
            
        time_per_run = max(1.0, min(remaining * 0.4, remaining / max(1, 4 - n_cma)))
        
        if n_cma < min(5, len(sorted_archive)):
            x0 = sorted_archive[n_cma][1].copy()
            sigma0 = base_sigma * (0.15 + 0.15 * n_cma)
        elif best_params is not None:
            perturbation = np.random.normal(0, 0.15 * (1 + n_cma * 0.1), dim) * ranges
            x0 = clip(best_params + perturbation)
            sigma0 = base_sigma * min(1.5, 0.2 + 0.2 * n_cma)
        else:
            x0 = lower + np.random.random(dim) * ranges
            sigma0 = base_sigma

        cmaes_search(x0, sigma0, time_per_run)
        n_cma += 1

    # --- Phase 4: Nelder-Mead ---
    if best_params is not None and time_ok(0.96):
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step_size = 0.002 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step_size[i] if best_params[i] + step_size[i] <= upper[i] else -step_size[i]

        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if not time_ok(0.96):
                f_simplex[i] = float('inf')
            else:
                f_simplex[i] = eval_f(simplex[i])

        while time_ok(0.97):
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            centroid = np.mean(simplex[:n], axis=0)

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
                else:
                    xc = clip(centroid + 0.5 * (simplex[n] - centroid))
                fc = eval_f(xc)
                if fc < min(fr, f_simplex[n]):
                    simplex[n], f_simplex[n] = xc, fc
                else:
                    for i in range(1, n + 1):
                        if not time_ok(0.97): break
                        simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_f(simplex[i])

            if np.max(np.abs(simplex[n] - simplex[0]) / (ranges + 1e-30)) < 1e-14:
                break

    # --- Phase 5: Fine coordinate descent ---
    if best_params is not None and time_ok(0.99):
        x = best_params.copy()
        step = 0.0005 * ranges
        while time_ok(0.995):
            improved = False
            for d in range(dim):
                if not time_ok(0.995): break
                for direction in [1.0, -1.0]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = eval_f(trial)
                    if f < best:
                        x = best_params.copy()
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break

    return best
