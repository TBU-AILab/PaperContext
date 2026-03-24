#Looking at the previous algorithm, I can identify several improvement opportunities:
#
#1. **More efficient time allocation** - spend more time on CMA-ES which is the core optimizer
#2. **Better CMA-ES implementation** - fix bugs and improve convergence
#3. **Adaptive population sizing** - better BIPOP strategy
#4. **Improved local search** - more aggressive refinement
#5. **Better initial sampling** - Latin Hypercube instead of Halton for better coverage
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

    # Phase 1: Initial sampling
    n_init = min(max(20 * dim, 100), 800)
    if dim > 60:
        n_init = min(n_init, 300)

    population = []
    samples = lhs_sample(n_init)
    for i in range(n_init):
        if not time_ok(0.08):
            break
        f = eval_f(samples[i])
        population.append((f, samples[i].copy()))

    # Opposition-based learning
    population.sort(key=lambda t: t[0])
    n_opp = min(15, len(population))
    for i in range(n_opp):
        if not time_ok(0.10):
            break
        opp = lower + upper - population[i][1]
        f = eval_f(opp)
        population.append((f, opp.copy()))

    # Evaluate center
    if time_ok(0.10):
        eval_f(mid.copy())

    population.sort(key=lambda t: t[0])

    elite_pool = []
    for i in range(min(20, len(population))):
        elite_pool.append((population[i][0], population[i][1].copy()))

    # Quick DE phase
    def de_phase(pop, time_frac_end):
        nonlocal best, best_x
        pop = [p for p in pop]
        np_size = len(pop)
        
        gen = 0
        while time_ok(time_frac_end) and gen < 30:
            new_pop = []
            F = 0.5 + 0.3 * np.random.random()
            CR = 0.8 + 0.15 * np.random.random()
            
            for i in range(np_size):
                if not time_ok(time_frac_end):
                    break
                idxs = list(range(np_size))
                idxs.remove(i)
                a, b, c = [pop[j] for j in np.random.choice(idxs, 3, replace=False)]

                # DE/current-to-best/1
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
            gen += 1
        return pop

    de_pop_size = min(max(30, 6 * dim), 120, len(population))
    de_pop = [(p[0], p[1].copy()) for p in population[:de_pop_size]]
    de_pop = de_phase(de_pop, 0.20)

    de_pop.sort(key=lambda t: t[0])
    elite_pool = []
    for i in range(min(20, len(de_pop))):
        elite_pool.append((de_pop[i][0], de_pop[i][1].copy()))
    elite_pool.sort(key=lambda t: t[0])

    # Phase 2: CMA-ES
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
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
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

        while elapsed() < max_time_abs and time_ok(0.94):
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
                if not time_ok(0.94) or elapsed() >= max_time_abs:
                    return best_local_x

                z = np.random.randn(n)
                if use_sep:
                    y = np.sqrt(diagC) * z
                else:
                    y = B @ (D * z)

                x = xmean + sigma * y
                # Bounce boundary
                for d_i in range(n):
                    lo, hi = lower[d_i], upper[d_i]
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

            if arfitness[idx[0]] < best_local:
                best_local = arfitness[idx[0]]
                best_local_x = arxs[idx[0]].copy()
                stag_count = 0
            else:
                stag_count += 1

            f_history.append(np.median(arfitness))

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
                diagC = (1 - c1 - cmu) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu * artmp_sq
                diagC = np.maximum(diagC, 1e-20)
            else:
                artmp_pos = np.zeros((mu, n))
                for i in range(mu):
                    artmp_pos[i] = (arxs[idx[i]] - xold) / max(sigma, 1e-30)

                rank_mu_pos = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_pos += weights[i] * np.outer(artmp_pos[i], artmp_pos[i])

                C = (1 - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu * rank_mu_pos

            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5 * np.max(ranges))

            gen += 1

            if sigma < 1e-16 * np.mean(ranges):
                return best_local_x
            if stag_count > 20 + 10 * n / lam:
                return best_local_x
            if gen > 200 + 150 * n / lam:
                return best_local_x

            if len(f_history) > 30:
                recent = f_history[-30:]
                if abs(recent[-1] - recent[0]) < 1e-14 * (abs(recent[0]) + 1e-30):
                    return best_local_x

        return best_local_x

    # CMA-ES from elite points first
    default_lam = max(6, 4 + int(3 * np.log(dim)))
    n_elite_starts = min(4, len(elite_pool))
    
    for i in range(n_elite_starts):
        if not time_ok(0.40):
            break
        rem = time_left()
        x0 = elite_pool[i][1].copy()
        sig0 = 0.2 * np.mean(ranges) * (0.5 + np.random.random())
        if i == 0 and best_x is not None:
            x0 = best_x.copy()
        t_budget = elapsed() + min(rem * 0.3, rem / max(n_elite_starts - i, 1))
        cmaes_run(x0, sig0, max_time_abs=t_budget)

    # BIPOP restarts
    large_budget = 0
    small_budget = 0
    restart = 0

    while time_ok(0.82):
        rem = time_left()
        if rem < 0.3:
            break

        use_large = (large_budget <= small_budget) or np.random.random() < 0.3

        if use_large:
            power = min(restart + 1, 8)
            large_lam = int(default_lam * (2 ** np.random.uniform(0, power)))
            large_lam = min(large_lam, max(512, 10 * dim))
            large_lam = max(large_lam, default_lam)

            if np.random.random() < 0.35 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * 0.5 * ranges
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sig0 = (0.2 + 0.3 * np.random.random()) * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.35, rem)
            cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
            large_budget += elapsed() - t_start
        else:
            small_lam = max(6, int(default_lam * (0.5 + 0.5 * np.random.random())))

            if best_x is not None and np.random.random() < 0.75:
                spread = 0.005 + 0.12 * np.random.random()
                x0 = best_x + np.random.randn(dim) * spread * ranges
                x0 = np.clip(x0, lower, upper)
                sig0 = (0.01 + 0.08 * np.random.random()) * np.mean(ranges)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.2 * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.15, rem)
            cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
            small_budget += elapsed() - t_start

        restart += 1

    # Phase 3: Nelder-Mead from best
    if best_x is not None and time_ok(0.88):
        n = dim
        nm_scale = 0.01
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = eval_f(simplex[0])

        for i in range(n):
            if not time_ok(0.92):
                break
            simplex[i + 1] = best_x.copy()
            step = nm_scale * ranges[i]
            if step < 1e-15:
                step = 1e-6
            simplex[i + 1][i] += step
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
            f_simplex[i + 1] = eval_f(simplex[i + 1])

        for iteration in range(500 * n):
            if not time_ok(0.96):
                break

            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:-1], axis=0)

            xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lower, upper)
            fr = eval_f(xr)

            if fr < f_simplex[0]:
                xe = np.clip(centroid + 2.0 * (xr - centroid), lower, upper)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = np.clip(centroid + 0.5 * (xr - centroid), lower, upper)
                    fc = eval_f(xc)
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.96):
                                break
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])
                else:
                    xc = np.clip(centroid - 0.5 * (centroid - simplex[-1]), lower, upper)
                    fc = eval_f(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(0.96):
                                break
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), lower, upper)
                            f_simplex[i] = eval_f(simplex[i])

            f_range = abs(f_simplex[-1] - f_simplex[0])
            if f_range < 1e-15 * (abs(f_simplex[0]) + 1e-30):
                break

    # Phase 4: Coordinate descent refinement
    if best_x is not None and time_ok(0.96):
        x_curr = best_x.copy()
        f_curr = best

        for scale_exp in range(-2, -9, -1):
            if not time_ok(0.99):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved = True
            while improved and time_ok(0.99):
                improved = False
                for i in np.random.permutation(dim):
                    if not time_ok(0.99):
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
