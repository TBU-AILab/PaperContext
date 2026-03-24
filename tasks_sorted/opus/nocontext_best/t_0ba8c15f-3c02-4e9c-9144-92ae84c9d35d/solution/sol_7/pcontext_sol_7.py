#Looking at the current algorithm, I see it's already quite sophisticated with DE + CMA-ES + local search. Let me analyze potential weaknesses and improve:
#
#1. **DE phase may be slow** - too much time spent there
#2. **CMA-ES restart strategy** could be better tuned
#3. **Local search** is basic coordinate descent - can use Nelder-Mead
#4. **Missing multi-start from diverse good points**
#5. **Better time allocation** needed
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    evals = 0

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
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: LHS sampling ---
    n_init = min(max(40 * dim, 300), 1000)
    init_pop = np.zeros((n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        for i in range(n_init):
            init_pop[i, d] = lower[d] + (perm[i] + np.random.random()) / n_init * ranges[d]

    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if not time_ok(0.12):
            break
        init_fitness[i] = eval_f(init_pop[i])

    valid = init_fitness < float('inf')
    valid_pop = init_pop[valid]
    valid_fit = init_fitness[valid]
    sorted_idx = np.argsort(valid_fit)

    # Keep top solutions as elite archive
    n_archive = min(20, len(sorted_idx))
    archive_x = [valid_pop[sorted_idx[i]].copy() for i in range(n_archive)]
    archive_f = [valid_fit[sorted_idx[i]] for i in range(n_archive)]

    # --- Phase 2: JADE-style Differential Evolution ---
    pop_size = min(max(8 * dim, 40), 120)
    n_elite_de = min(pop_size // 2, len(valid_pop))

    de_pop = np.zeros((pop_size, dim))
    de_fit = np.full(pop_size, float('inf'))

    top_idx = sorted_idx[:n_elite_de]
    for i in range(n_elite_de):
        de_pop[i] = valid_pop[top_idx[i]].copy()
        de_fit[i] = valid_fit[top_idx[i]]

    for i in range(n_elite_de, pop_size):
        de_pop[i] = lower + np.random.random(dim) * ranges
        if time_ok(0.18):
            de_fit[i] = eval_f(de_pop[i])

    mu_F = 0.5
    mu_CR = 0.5
    de_archive = []

    de_end = 0.45
    gen = 0
    while time_ok(de_end):
        S_F = []
        S_CR = []
        delta_f = []

        for i in range(pop_size):
            if not time_ok(de_end):
                break

            F = np.clip(np.random.standard_cauchy() * 0.1 + mu_F, 0.05, 1.0)
            CR = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)

            p = max(2, int(0.05 * pop_size + 1))
            p_best_idx = np.argsort(de_fit)[:p]
            x_pbest = de_pop[np.random.choice(p_best_idx)]

            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            idxs.remove(r1)

            if len(de_archive) > 0 and np.random.random() < 0.5:
                combined = list(range(len(idxs)))
                r2_from_archive = np.random.random() < len(de_archive) / (len(idxs) + len(de_archive))
                if r2_from_archive:
                    x_r2 = de_archive[np.random.randint(len(de_archive))]
                else:
                    r2 = np.random.choice(idxs)
                    x_r2 = de_pop[r2]
            else:
                r2 = np.random.choice(idxs)
                x_r2 = de_pop[r2]

            mutant = de_pop[i] + F * (x_pbest - de_pop[i]) + F * (de_pop[r1] - x_r2)

            trial = de_pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) < CR
            mask[j_rand] = True
            trial[mask] = mutant[mask]

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
                    delta_f.append(de_fit[i] - f_trial)
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
            mu_F = 0.9 * mu_F + 0.1 * np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
            mu_CR = 0.9 * mu_CR + 0.1 * np.sum(w * S_CR_arr)

        gen += 1

    # Update archive with DE results
    for i in range(pop_size):
        if de_fit[i] < max(archive_f):
            worst_idx = np.argmax(archive_f)
            archive_x[worst_idx] = de_pop[i].copy()
            archive_f[worst_idx] = de_fit[i]

    # --- Phase 3: CMA-ES with IPOP restarts ---
    def cmaes_search(x0, sigma0, end_fraction):
        nonlocal best, best_params
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

        mean = x0.copy()
        sigma = sigma0
        stagnation = 0
        prev_best_gen = float('inf')
        f_history = []

        generation = 0
        while time_ok(end_fraction):
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = mean + sigma * (B @ (D * arz[k]))
                for dd in range(n):
                    while arx[k][dd] < lower[dd] or arx[k][dd] > upper[dd]:
                        if arx[k][dd] < lower[dd]:
                            arx[k][dd] = 2 * lower[dd] - arx[k][dd]
                        if arx[k][dd] > upper[dd]:
                            arx[k][dd] = 2 * upper[dd] - arx[k][dd]
                arx[k] = clip(arx[k])

            fitnesses = np.zeros(lam)
            for k in range(lam):
                if not time_ok(end_fraction):
                    return
                fitnesses[k] = eval_f(arx[k])

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

            # Check flat fitness
            if len(f_history) > 20:
                recent = f_history[-20:]
                if max(recent) - min(recent) < 1e-12:
                    return

            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[:mu_cma], axis=0)

            mean_diff = mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ mean_diff) / sigma
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(generation+1))) / chiN < 1.4 + 2/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * mean_diff / sigma

            artmp = (arx[:mu_cma] - old_mean) / sigma
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

    n_cma = 0
    base_sigma = 0.3 * np.mean(ranges)

    while time_ok(0.88):
        if n_cma < len(archive_x):
            x0 = archive_x[n_cma].copy() + np.random.normal(0, 0.01, dim) * ranges
            x0 = clip(x0)
            sigma0 = base_sigma * 0.3
        elif best_params is not None:
            x0 = best_params + np.random.normal(0, 0.2, dim) * ranges
            x0 = clip(x0)
            sigma0 = base_sigma * min(1.0, 0.3 * (1.3 ** (n_cma - len(archive_x))))
        else:
            x0 = lower + np.random.random(dim) * ranges
            sigma0 = base_sigma

        target_frac = min(elapsed() / max_time + max(0.06, 0.3 * (max_time - elapsed()) / max_time), 0.88)
        cmaes_search(x0, sigma0, target_frac)
        n_cma += 1

    # --- Phase 4: Nelder-Mead local search ---
    if best_params is not None and time_ok(0.97):
        n = dim
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5

        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        step_size = 0.005 * ranges
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step_size[i] if best_params[i] + step_size[i] <= upper[i] else -step_size[i]

        f_simplex = np.array([eval_f(simplex[i]) for i in range(n + 1) if time_ok(0.96)])
        if len(f_simplex) < n + 1:
            for i in range(len(f_simplex), n + 1):
                f_simplex = np.append(f_simplex, float('inf'))

        while time_ok(0.97):
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:n], axis=0)

            # Reflect
            xr = clip(centroid + alpha_nm * (centroid - simplex[n]))
            fr = eval_f(xr)

            if fr < f_simplex[0]:
                xe = clip(centroid + gamma_nm * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho_nm * (xr - centroid))
                else:
                    xc = clip(centroid + rho_nm * (simplex[n] - centroid))
                fc = eval_f(xc)
                if fc < min(fr, f_simplex[n]):
                    simplex[n] = xc; f_simplex[n] = fc
                else:
                    for i in range(1, n + 1):
                        if not time_ok(0.97):
                            break
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_f(simplex[i])

            diam = np.max(np.abs(simplex[n] - simplex[0]) / (ranges + 1e-30))
            if diam < 1e-13:
                break

    # --- Phase 5: Final coordinate descent ---
    if best_params is not None and time_ok(0.99):
        x = best_params.copy()
        step = 0.001 * ranges
        while time_ok(0.995):
            improved = False
            for d in range(dim):
                if not time_ok(0.995):
                    break
                cur_f = best
                for direction in [1.0, -1.0]:
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + direction * step[d], lower[d], upper[d])
                    f = eval_f(trial)
                    if f < cur_f:
                        x = best_params.copy()
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-14:
                    break

    return best
#
#Key improvements over the previous version:
#1. **Weighted Lehmer mean for F adaptation** using improvement deltas (proper JADE)
#2. **Diverse CMA-ES restarts** cycling through archive elite points, not just best
#3. **Nelder-Mead** as a proper simplex-based local search before coordinate descent
#4. **Flat fitness detection** in CMA-ES to avoid wasting time
#5. **Better time budgeting** with more time for CMA-ES exploration
#6. **Finer initial step sizes** in local search phases
#7. **Larger initial LHS** for better global coverage
