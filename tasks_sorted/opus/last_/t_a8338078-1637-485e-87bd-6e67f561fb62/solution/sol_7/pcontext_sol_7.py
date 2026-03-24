#Looking at the previous algorithm, I can see several areas for improvement:
#
#1. **Better time allocation** - too much time on initial phases, not enough on CMA-ES restarts
#2. **Improved CMA-ES with weighted recombination and mirrored sampling**
#3. **Better restart strategy** - more aggressive local restarts around best
#4. **Smarter boundary handling**
#5. **Add a pattern search that's more efficient than coordinate descent**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    mid = (lower + upper) / 2.0

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def time_left():
        return max_time - elapsed()

    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction

    evals = 0
    def eval_f(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Latin Hypercube Sampling
    def lhs_sample(n_samples):
        result = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = np.random.permutation(n_samples)
            for i in range(n_samples):
                result[i, d] = lower[d] + ((perm[i] + np.random.random()) / n_samples) * ranges[d]
        return result

    # Sobol-like scrambled sampling for better coverage
    def quasi_random_sample(n_samples):
        # Use LHS as base
        return lhs_sample(n_samples)

    # Phase 1: Initial sampling - keep it efficient
    n_init = min(max(15 * dim, 80), 600)
    if dim > 60:
        n_init = min(n_init, 250)

    population = []
    samples = lhs_sample(n_init)
    for i in range(n_init):
        if not time_ok(0.06):
            break
        f = eval_f(samples[i])
        population.append((f, samples[i].copy()))

    # Center point
    if time_ok(0.07):
        eval_f(mid.copy())

    # Opposition-based learning on top candidates
    population.sort(key=lambda t: t[0])
    n_opp = min(10, len(population))
    for i in range(n_opp):
        if not time_ok(0.08):
            break
        opp = lower + upper - population[i][1]
        f = eval_f(opp)
        population.append((f, opp.copy()))

    population.sort(key=lambda t: t[0])

    # Quick DE phase to refine initial population
    def de_phase(pop, time_frac_end, max_gens=25):
        nonlocal best, best_x
        pop = [p for p in pop]
        np_size = len(pop)

        for gen in range(max_gens):
            if not time_ok(time_frac_end):
                break
            new_pop = []
            F = 0.5 + 0.3 * np.random.random()
            CR = 0.85 + 0.1 * np.random.random()

            for i in range(np_size):
                if not time_ok(time_frac_end):
                    break
                idxs = list(range(np_size))
                idxs.remove(i)
                chosen = np.random.choice(idxs, 3, replace=False)
                a, b, c = pop[chosen[0]], pop[chosen[1]], pop[chosen[2]]

                if best_x is not None:
                    mutant = pop[i][1] + F * (best_x - pop[i][1]) + F * (b[1] - c[1])
                else:
                    mutant = a[1] + F * (b[1] - c[1])

                trial = pop[i][1].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = np.clip(trial, lower, upper)

                ft = eval_f(trial)
                if ft <= pop[i][0]:
                    new_pop.append((ft, trial))
                else:
                    new_pop.append(pop[i])

            if len(new_pop) == np_size:
                pop = new_pop
        return pop

    de_pop_size = min(max(25, 5 * dim), 100, len(population))
    de_pop = [(p[0], p[1].copy()) for p in population[:de_pop_size]]
    de_pop = de_phase(de_pop, 0.18)

    de_pop.sort(key=lambda t: t[0])
    elite_pool = []
    for i in range(min(15, len(de_pop))):
        elite_pool.append((de_pop[i][0], de_pop[i][1].copy()))

    # CMA-ES implementation
    def cmaes_run(x0, sigma0, lam_override=None, max_time_abs=None):
        nonlocal best, best_x

        if max_time_abs is None:
            max_time_abs = elapsed() + 0.15 * max_time

        n = dim
        lam = lam_override if lam_override else max(6, 4 + int(3 * np.log(n)))
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        xmean = np.clip(x0.copy(), lower, upper)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)

        use_sep = (n > 40)

        if use_sep:
            diagC = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0

        gen = 0
        best_local = float('inf')
        best_local_x = xmean.copy()
        stag_count = 0
        f_history = []

        while elapsed() < max_time_abs and time_ok(0.95):
            if not use_sep and eigen_countdown <= 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, B = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    eigen_countdown = max(1, int(lam / (10 * n)))
                except:
                    return best_local_x
                if np.max(D) / (np.min(D) + 1e-30) > 1e14:
                    return best_local_x
            if not use_sep:
                eigen_countdown -= 1

            arxs = []
            arfitness = []

            for k in range(lam):
                if not time_ok(0.95) or elapsed() >= max_time_abs:
                    return best_local_x

                z = np.random.randn(n)
                if use_sep:
                    y = np.sqrt(diagC) * z
                else:
                    y = B @ (D * z)

                x = xmean + sigma * y
                # Bounce boundary handling
                for d_i in range(n):
                    lo, hi = lower[d_i], upper[d_i]
                    cnt = 0
                    while (x[d_i] < lo or x[d_i] > hi) and cnt < 10:
                        if x[d_i] < lo:
                            x[d_i] = 2 * lo - x[d_i]
                        if x[d_i] > hi:
                            x[d_i] = 2 * hi - x[d_i]
                        cnt += 1
                x = np.clip(x, lower, upper)

                f = eval_f(x)
                arxs.append(x)
                arfitness.append(f)

            idx = np.argsort(arfitness)

            if arfitness[idx[0]] < best_local:
                best_local = arfitness[idx[0]]
                best_local_x = arxs[idx[0]].copy()
                stag_count = 0
            else:
                stag_count += 1

            f_history.append(arfitness[idx[0]])

            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arxs[idx[i]]

            diff = xmean - xold

            if use_sep:
                inv_sqrt_diag = 1.0 / (np.sqrt(diagC) + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_diag * diff) / max(sigma, 1e-30)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / max(sigma, 1e-30)

            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / max(sigma, 1e-30)

            if use_sep:
                artmp_sq = np.zeros(n)
                for i in range(mu):
                    d_vec = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                    artmp_sq += weights[i] * d_vec**2
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * artmp_sq
                diagC = np.maximum(diagC, 1e-20)
            else:
                artmp_pos = np.zeros((mu, n))
                for i in range(mu):
                    artmp_pos[i] = (arxs[idx[i]] - xold) / max(sigma, 1e-30)

                rank_mu_pos = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_pos += weights[i] * np.outer(artmp_pos[i], artmp_pos[i])

                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * rank_mu_pos

            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5 * np.max(ranges))

            gen += 1

            if sigma < 1e-16 * np.mean(ranges):
                return best_local_x
            if stag_count > 15 + 10 * n / lam:
                return best_local_x
            if gen > 150 + 100 * n / lam:
                return best_local_x

            if len(f_history) > 25:
                recent = f_history[-25:]
                if abs(max(recent) - min(recent)) < 1e-14 * (abs(recent[0]) + 1e-30):
                    return best_local_x

        return best_local_x

    default_lam = max(6, 4 + int(3 * np.log(dim)))

    # CMA-ES from top elite points
    n_elite_starts = min(3, len(elite_pool))
    for i in range(n_elite_starts):
        if not time_ok(0.35):
            break
        rem = time_left()
        x0 = elite_pool[i][1].copy()
        if i == 0 and best_x is not None:
            x0 = best_x.copy()
        sig0 = 0.2 * np.mean(ranges) * (0.3 + 0.7 * np.random.random())
        t_budget = elapsed() + min(rem * 0.25, rem / max(n_elite_starts - i, 1))
        cmaes_run(x0, sig0, max_time_abs=t_budget)

    # BIPOP restarts - main optimization loop
    large_budget = 0
    small_budget = 0
    restart = 0

    while time_ok(0.85):
        rem = time_left()
        if rem < 0.2:
            break

        use_large = (large_budget <= small_budget) or np.random.random() < 0.25

        if use_large:
            power = min(restart + 1, 7)
            large_lam = int(default_lam * (2 ** np.random.uniform(0, power)))
            large_lam = min(large_lam, max(512, 8 * dim))
            large_lam = max(large_lam, default_lam)

            if np.random.random() < 0.3 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * 0.4 * ranges
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sig0 = (0.2 + 0.3 * np.random.random()) * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.3, rem)
            cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
            large_budget += elapsed() - t_start
        else:
            small_lam = max(6, int(default_lam * (0.5 + 0.5 * np.random.random())))

            if best_x is not None and np.random.random() < 0.8:
                spread = 0.003 + 0.1 * np.random.random()
                x0 = best_x + np.random.randn(dim) * spread * ranges
                x0 = np.clip(x0, lower, upper)
                sig0 = (0.005 + 0.06 * np.random.random()) * np.mean(ranges)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.15 * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.12, rem)
            cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
            small_budget += elapsed() - t_start

        restart += 1

    # Phase 3: Nelder-Mead polishing
    if best_x is not None and time_ok(0.90):
        n = dim
        nm_scale = 0.005
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = best

        for i in range(n):
            if not time_ok(0.93):
                break
            simplex[i + 1] = best_x.copy()
            step = nm_scale * ranges[i]
            if step < 1e-15:
                step = 1e-6
            simplex[i + 1][i] += step * (1 if np.random.random() < 0.5 else -1)
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            f_simplex[i + 1] = eval_f(simplex[i + 1])

        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5

        for iteration in range(300 * n):
            if not time_ok(0.97):
                break

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
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), lower, upper)
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.97):
                                break
                            simplex[i] = np.clip(simplex[0] + shrink * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])
                else:
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), lower, upper)
                    fc = eval_f(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.97):
                                break
                            simplex[i] = np.clip(simplex[0] + shrink * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])

            f_range = abs(f_simplex[-1] - f_simplex[0])
            if f_range < 1e-15 * (abs(f_simplex[0]) + 1e-30):
                break

    # Phase 4: Fine-grained coordinate descent
    if best_x is not None and time_ok(0.97):
        x_curr = best_x.copy()
        f_curr = best

        for scale_exp in range(-2, -10, -1):
            if not time_ok(0.995):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved = True
            passes = 0
            while improved and time_ok(0.995) and passes < 3:
                improved = False
                passes += 1
                for i in np.random.permutation(dim):
                    if not time_ok(0.995):
                        break
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[i] = np.clip(x_curr[i] + direction * step_size, lower[i], upper[i])
                        ft = eval_f(x_trial)
                        if ft < f_curr:
                            f_curr = ft
                            x_curr = x_trial
                            improved = True
                            break

    return best
