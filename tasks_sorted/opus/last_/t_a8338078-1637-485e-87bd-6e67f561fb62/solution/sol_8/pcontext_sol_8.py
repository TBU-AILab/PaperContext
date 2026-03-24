#Looking at the previous algorithm, I see several opportunities for improvement:
#
#1. **Better time allocation** - spend more time on CMA-ES optimization
#2. **Adaptive population sizing** - scale better with dimension
#3. **Improved restart strategy** - use IPOP/BIPOP more effectively
#4. **Better local search** - add golden section and Powell-like methods
#5. **Reduce overhead** - streamline boundary handling
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

    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction

    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # LHS sampling
    def lhs_sample(n_samples):
        result = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = np.random.permutation(n_samples)
            for i in range(n_samples):
                result[i, d] = lower[d] + ((perm[i] + np.random.random()) / n_samples) * ranges[d]
        return result

    # Phase 1: Initial sampling
    n_init = min(max(20 * dim, 100), 500)
    if dim > 50:
        n_init = min(n_init, 200)

    population = []
    samples = lhs_sample(n_init)
    for i in range(n_init):
        if not time_ok(0.05):
            break
        f = eval_f(samples[i])
        population.append((f, samples[i].copy()))

    eval_f(mid.copy())

    # Opposition-based
    population.sort(key=lambda t: t[0])
    n_opp = min(8, len(population))
    for i in range(n_opp):
        if not time_ok(0.06):
            break
        opp = lower + upper - population[i][1]
        f = eval_f(opp)
        population.append((f, opp.copy()))

    population.sort(key=lambda t: t[0])

    # Quick DE phase
    def de_phase(pop, time_frac_end, max_gens=30):
        np_size = len(pop)
        pop = list(pop)
        for gen in range(max_gens):
            if not time_ok(time_frac_end):
                break
            F = 0.5 + 0.3 * np.random.random()
            CR = 0.8 + 0.15 * np.random.random()
            new_pop = []
            for i in range(np_size):
                if not time_ok(time_frac_end):
                    break
                idxs = list(range(np_size))
                idxs.remove(i)
                chosen = np.random.choice(idxs, 3, replace=False)
                a, b, c = pop[chosen[0]], pop[chosen[1]], pop[chosen[2]]
                
                # current-to-best/1
                if best_x is not None and np.random.random() < 0.7:
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

    de_pop_size = min(max(20, 4 * dim), 80, len(population))
    de_pop = [(p[0], p[1].copy()) for p in population[:de_pop_size]]
    de_pop = de_phase(de_pop, 0.15)
    de_pop.sort(key=lambda t: t[0])

    elite_pool = []
    for i in range(min(10, len(de_pop))):
        elite_pool.append((de_pop[i][0], de_pop[i][1].copy()))

    # CMA-ES
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

        while elapsed() < max_time_abs and time_ok(0.96):
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
                if not time_ok(0.96) or elapsed() >= max_time_abs:
                    return best_local_x
                z = np.random.randn(n)
                if use_sep:
                    y = np.sqrt(diagC) * z
                else:
                    y = B @ (D * z)
                x = xmean + sigma * y
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
            if gen > 200 + 150 * n / lam:
                return best_local_x

            if len(f_history) > 30:
                recent = f_history[-30:]
                if abs(max(recent) - min(recent)) < 1e-14 * (abs(recent[0]) + 1e-30):
                    return best_local_x

        return best_local_x

    default_lam = max(6, 4 + int(3 * np.log(dim)))

    # CMA-ES from top elite points
    n_elite_starts = min(3, len(elite_pool))
    for i in range(n_elite_starts):
        if not time_ok(0.30):
            break
        rem = time_left() if time_ok(0.30) else 0
        x0 = elite_pool[i][1].copy()
        if i == 0 and best_x is not None:
            x0 = best_x.copy()
        sig0 = 0.15 * np.mean(ranges) * (0.3 + 0.7 * np.random.random())
        t_budget = elapsed() + min(rem * 0.2, rem / max(n_elite_starts - i, 1))
        cmaes_run(x0, sig0, max_time_abs=t_budget)

    # BIPOP restarts
    large_budget = 0
    small_budget = 0
    restart = 0

    while time_ok(0.82):
        rem = max_time - elapsed()
        if rem < 0.3:
            break

        use_large = (large_budget <= small_budget) or np.random.random() < 0.3

        if use_large:
            power = min(restart + 1, 8)
            large_lam = int(default_lam * (2 ** np.random.uniform(0, power)))
            large_lam = min(large_lam, max(512, 8 * dim))
            large_lam = max(large_lam, default_lam)

            if np.random.random() < 0.35 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * 0.3 * ranges
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sig0 = (0.2 + 0.3 * np.random.random()) * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.28, rem)
            cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
            large_budget += elapsed() - t_start
        else:
            small_lam = max(6, int(default_lam * (0.5 + 0.5 * np.random.random())))

            if best_x is not None and np.random.random() < 0.85:
                spread = 0.002 + 0.08 * np.random.random()
                x0 = best_x + np.random.randn(dim) * spread * ranges
                x0 = np.clip(x0, lower, upper)
                sig0 = (0.003 + 0.05 * np.random.random()) * np.mean(ranges)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.15 * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.10, rem)
            cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
            small_budget += elapsed() - t_start

        restart += 1

    # Powell-like coordinate descent with golden section
    if best_x is not None and time_ok(0.88):
        x_curr = best_x.copy()
        f_curr = best

        def line_min(x, direction, max_step):
            gr = (np.sqrt(5) - 1) / 2
            a_lo, a_hi = -max_step, max_step
            c = a_hi - gr * (a_hi - a_lo)
            d = a_lo + gr * (a_hi - a_lo)
            fc = eval_f(np.clip(x + c * direction, lower, upper))
            fd = eval_f(np.clip(x + d * direction, lower, upper))
            for _ in range(12):
                if not time_ok(0.96):
                    break
                if fc < fd:
                    a_hi = d
                    d = c
                    fd = fc
                    c = a_hi - gr * (a_hi - a_lo)
                    fc = eval_f(np.clip(x + c * direction, lower, upper))
                else:
                    a_lo = c
                    c = d
                    fc = fd
                    d = a_lo + gr * (a_hi - a_lo)
                    fd = eval_f(np.clip(x + d * direction, lower, upper))
            alpha_best = (a_lo + a_hi) / 2
            x_new = np.clip(x + alpha_best * direction, lower, upper)
            f_new = eval_f(x_new)
            return x_new, f_new

        for cycle in range(5):
            if not time_ok(0.94):
                break
            for i in np.random.permutation(dim):
                if not time_ok(0.95):
                    break
                direction = np.zeros(dim)
                direction[i] = 1.0
                step = 0.01 * ranges[i]
                x_new, f_new = line_min(x_curr, direction, step)
                if f_new < f_curr:
                    x_curr = x_new
                    f_curr = f_new

    # Fine coordinate descent
    if best_x is not None and time_ok(0.96):
        x_curr = best_x.copy()
        f_curr = best

        for scale_exp in range(-3, -11, -1):
            if not time_ok(0.995):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved = True
            passes = 0
            while improved and time_ok(0.995) and passes < 3:
                improved = False
                passes += 1
                for i in np.random.permutation(dim):
                    if not time_ok(0.998):
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
