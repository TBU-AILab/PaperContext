#Looking at the previous algorithm, I can identify several improvement opportunities:
#
#1. **Better CMA-ES implementation** - improve boundary handling, eigendecomposition efficiency
#2. **Smarter restart strategy** - more aggressive early restarts from best points
#3. **Better local search** - Nelder-Mead simplex + pattern search
#4. **Time management** - more efficient allocation across phases
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction

    def eval_f(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Sobol-like + LHS sampling with opposition
    n_init = min(max(10 * dim, 50), 400)
    if dim > 80:
        n_init = min(n_init, 100)

    # LHS
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i] + lower[i]

    population = []
    for i in range(n_init):
        if not time_ok(0.07):
            break
        f = eval_f(init_points[i])
        population.append((f, init_points[i].copy()))
        # Opposition
        if time_ok(0.07):
            opp = lower + upper - init_points[i]
            fo = eval_f(opp)
            population.append((fo, opp.copy()))

    # Center point
    if time_ok(0.08):
        eval_f(center.copy())

    population.sort(key=lambda t: t[0])

    # Keep track of top solutions for restarts
    elite_pool = []
    for i in range(min(10, len(population))):
        elite_pool.append(population[i][1].copy())

    # Phase 2: CMA-ES with BIPOP restarts
    def cmaes_run(x0, sigma0, lam_override=None, max_time_abs=None):
        nonlocal best, best_x

        if max_time_abs is None:
            max_time_abs = elapsed() + 0.2 * max_time

        n = dim
        lam = lam_override if lam_override else max(6, 4 + int(3 * np.log(n)))
        if lam % 2 == 1:
            lam += 1
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
        stag_count = 0
        flat_count = 0
        f_history = []
        best_local_x = xmean.copy()

        while time_ok(0.92) and elapsed() < max_time_abs:
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
                # Condition check
                if np.max(D) / (np.min(D) + 1e-30) > 1e14:
                    return best_local_x
            if not use_sep:
                eigen_countdown -= 1

            arxs = []
            arfitness = []

            for k in range(lam):
                if not time_ok(0.92) or elapsed() >= max_time_abs:
                    return best_local_x

                z = np.random.randn(n)
                if use_sep:
                    y = np.sqrt(diagC) * z
                else:
                    y = B @ (D * z)

                x = xmean + sigma * y
                # Mirror boundary handling
                for d_i in range(n):
                    lo, hi = lower[d_i], upper[d_i]
                    if hi == lo:
                        x[d_i] = lo
                        continue
                    while x[d_i] < lo or x[d_i] > hi:
                        if x[d_i] < lo:
                            x[d_i] = 2 * lo - x[d_i]
                        if x[d_i] > hi:
                            x[d_i] = 2 * hi - x[d_i]
                x = np.clip(x, lower, upper)

                f = eval_f(x)
                arxs.append(x)
                arfitness.append(f)

            idx = np.argsort(arfitness)
            local_best_f = arfitness[idx[0]]

            if local_best_f < best_local:
                best_local = local_best_f
                best_local_x = arxs[idx[0]].copy()
                stag_count = 0
                flat_count = 0
            else:
                stag_count += 1

            f_history.append(local_best_f)

            frange = abs(arfitness[idx[0]] - arfitness[idx[-1]])
            if frange < 1e-15 * (abs(arfitness[idx[0]]) + 1e-30):
                flat_count += 1
                if flat_count > 8:
                    return best_local_x

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
                # Limit condition
                max_diag = np.max(diagC)
                diagC = np.maximum(diagC, max_diag * 1e-14)
            else:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                rank_mu_update = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_update += weights[i] * np.outer(artmp[i], artmp[i])
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * rank_mu_update

            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5 * np.max(ranges))

            gen += 1

            if sigma < 1e-16 * np.mean(ranges):
                return best_local_x
            if stag_count > 30 + 15 * n / lam:
                return best_local_x
            if gen > 500 + 200 * n / lam:
                return best_local_x

            if len(f_history) > 30:
                recent = f_history[-30:]
                if abs(recent[-1] - recent[0]) < 1e-15 * (abs(recent[0]) + 1e-30):
                    return best_local_x

        return best_local_x

    # Run CMA-ES restarts with BIPOP
    default_lam = max(6, 4 + int(3 * np.log(dim)))
    large_lam = default_lam
    small_budget_used = 0
    large_budget_used = 0
    restart = 0
    num_elite_starts = min(5, len(elite_pool))

    while time_ok(0.88):
        rem_time = max_time * 0.92 - elapsed()
        if rem_time < 0.3:
            break

        if restart < num_elite_starts:
            x0 = elite_pool[restart].copy()
            sig0 = 0.2 * np.mean(ranges) * (0.5 + 0.5 * np.random.random())
            if restart == 0 and best_x is not None:
                x0 = best_x.copy()
                sig0 = 0.15 * np.mean(ranges)
            t_budget = elapsed() + min(rem_time * 0.25, rem_time / max(num_elite_starts - restart, 1))
            result_x = cmaes_run(x0, sig0, max_time_abs=t_budget)
            if result_x is not None and restart < len(elite_pool):
                # Try to add good result to pool
                pass
        else:
            use_large = (large_budget_used <= small_budget_used) or np.random.random() < 0.3

            if use_large:
                power = min(restart - num_elite_starts + 1, 6)
                large_lam = int(default_lam * (2 ** power))
                large_lam = min(large_lam, max(512, 10 * dim))

                if np.random.random() < 0.4 and best_x is not None:
                    x0 = best_x + np.random.randn(dim) * 0.3 * ranges
                    x0 = np.clip(x0, lower, upper)
                else:
                    x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = (0.2 + 0.2 * np.random.random()) * np.mean(ranges)

                t_start = elapsed()
                t_budget = elapsed() + min(rem_time * 0.4, rem_time)
                cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
                large_budget_used += elapsed() - t_start
            else:
                small_lam = max(6, int(default_lam * (0.5 + np.random.random())))

                if best_x is not None and np.random.random() < 0.6:
                    spread = 0.01 + 0.2 * np.random.random()
                    x0 = best_x + np.random.randn(dim) * spread * ranges
                    x0 = np.clip(x0, lower, upper)
                    sig0 = (0.02 + 0.15 * np.random.random()) * np.mean(ranges)
                else:
                    x0 = lower + np.random.uniform(0, 1, dim) * ranges
                    sig0 = 0.2 * np.mean(ranges)

                t_start = elapsed()
                t_budget = elapsed() + min(rem_time * 0.2, rem_time)
                cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
                small_budget_used += elapsed() - t_start

        restart += 1

    # Phase 3: Pattern search local refinement
    if best_x is not None and time_ok(0.94):
        x_curr = best_x.copy()
        f_curr = best

        for scale_exp in range(-2, -7, -1):
            if not time_ok(0.97):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved_any = True
            while improved_any and time_ok(0.97):
                improved_any = False
                # Random permutation of dimensions
                perm = np.random.permutation(dim)
                for i in perm:
                    if not time_ok(0.97):
                        break
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[i] = np.clip(x_curr[i] + direction * step_size, lower[i], upper[i])
                        ft = eval_f(x_trial)
                        if ft < f_curr:
                            f_curr = ft
                            x_curr = x_trial
                            improved_any = True
                            break

    # Phase 4: Final gradient-based refinement
    if best_x is not None and time_ok(0.975):
        x_curr = best_x.copy()
        f_curr = best
        h = 1e-7 * np.mean(ranges)

        for _ in range(5):
            if not time_ok(0.99):
                break
            grad = np.zeros(dim)
            for i in range(dim):
                if not time_ok(0.99):
                    break
                xp = x_curr.copy()
                xp[i] = min(x_curr[i] + h, upper[i])
                fp = eval_f(xp)
                xm = x_curr.copy()
                xm[i] = max(x_curr[i] - h, lower[i])
                fm = eval_f(xm)
                grad[i] = (fp - fm) / (xp[i] - xm[i] + 1e-30)
                if min(fp, fm) < f_curr:
                    f_curr = min(fp, fm)
                    x_curr = xp if fp < fm else xm

            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-30 and time_ok(0.99):
                d = -grad / gnorm
                for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
                    if not time_ok(0.995):
                        break
                    step = alpha * np.mean(ranges)
                    xt = np.clip(x_curr + step * d, lower, upper)
                    ft = eval_f(xt)
                    if ft < f_curr:
                        f_curr = ft
                        x_curr = xt

    return best
