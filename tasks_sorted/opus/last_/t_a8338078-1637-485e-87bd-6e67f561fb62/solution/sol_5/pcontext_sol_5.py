#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better initialization** - use quasi-random sequences (Halton) for better coverage
#2. **Improved CMA-ES** - better step-size adaptation, weighted recombination, active CMA
#3. **Smarter restart strategy** - adaptive restart with better time budgeting
#4. **More effective local search** - Nelder-Mead simplex method
#5. **Better boundary handling** - use penalty + resampling hybrid
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

    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Halton sequence generator
    def halton_seq(index, base):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result

    def primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]

    primes = primes_up_to(max(dim * 5, 100))

    def halton_point(index):
        x = np.zeros(dim)
        for d in range(dim):
            x[d] = halton_seq(index, primes[d % len(primes)])
        return lower + x * ranges

    # Phase 1: Quasi-random initialization with Halton + opposition
    n_init = min(max(15 * dim, 80), 500)
    if dim > 60:
        n_init = min(n_init, 200)

    population = []

    # Halton sampling
    for i in range(1, n_init + 1):
        if not time_ok(0.08):
            break
        x = halton_point(i + np.random.randint(0, 100))
        # Add small perturbation for diversity
        x += np.random.randn(dim) * 0.01 * ranges
        x = np.clip(x, lower, upper)
        f = eval_f(x)
        population.append((f, x.copy()))

    # Opposition-based points for top candidates
    population.sort(key=lambda t: t[0])
    n_opp = min(20, len(population))
    for i in range(n_opp):
        if not time_ok(0.10):
            break
        opp = lower + upper - population[i][1]
        f = eval_f(opp)
        population.append((f, opp.copy()))

    # Center and random diagonal points
    if time_ok(0.10):
        eval_f(mid.copy())

    population.sort(key=lambda t: t[0])

    elite_pool = []
    for i in range(min(15, len(population))):
        elite_pool.append((population[i][0], population[i][1].copy()))

    # Differential Evolution phase (quick exploitation of population)
    def de_step(pop, F=0.8, CR=0.9):
        nonlocal best, best_x
        new_pop = []
        np_size = len(pop)
        for i in range(np_size):
            if not time_ok(0.18):
                break
            idxs = list(range(np_size))
            idxs.remove(i)
            a, b, c = [pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            # DE/best/1
            if best_x is not None and np.random.random() < 0.5:
                base_vec = best_x
            else:
                base_vec = a[1]
            mutant = base_vec + F * (b[1] - c[1])
            # Crossover
            trial = pop[i][1].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lower, upper)
            ft = eval_f(trial)
            if ft <= pop[i][0]:
                new_pop.append((ft, trial))
            else:
                new_pop.append(pop[i])
        return new_pop

    # Quick DE generations
    de_pop_size = min(max(30, 5 * dim), 100, len(population))
    de_pop = [(p[0], p[1].copy()) for p in population[:de_pop_size]]

    n_de_gens = max(3, min(15, int((max_time * 0.10) / max(0.01, elapsed() / max(len(population), 1) * de_pop_size))))
    for g in range(n_de_gens):
        if not time_ok(0.18):
            break
        F = 0.5 + 0.3 * np.random.random()
        CR = 0.8 + 0.2 * np.random.random()
        de_pop = de_step(de_pop, F=F, CR=CR)

    de_pop.sort(key=lambda t: t[0])
    for i in range(min(5, len(de_pop))):
        found = False
        for ef, ex in elite_pool:
            if np.allclose(ex, de_pop[i][1], atol=1e-10):
                found = True
                break
        if not found:
            elite_pool.append((de_pop[i][0], de_pop[i][1].copy()))
    elite_pool.sort(key=lambda t: t[0])
    elite_pool = elite_pool[:20]

    # Phase 2: CMA-ES with restarts
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

        # Active CMA weights
        weights_neg = np.log(lam + 0.5) - np.log(np.arange(mu + 1, lam + 1))
        weights_neg = weights_neg / np.sum(weights_neg)
        alpha_mu_neg = 1 + c1_val(n, mueff) / max(cmu_val_f(n, mueff), 1e-30)
        alpha_mueff_neg = 1 + 2 * mueff / (mueff + 2)

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

        use_sep = (n > 50)

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

        while elapsed() < max_time_abs and time_ok(0.92):
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
            arz = []

            for k in range(lam):
                if not time_ok(0.92) or elapsed() >= max_time_abs:
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
                    if hi <= lo:
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
                arz.append(z)

            idx = np.argsort(arfitness)
            local_best_f = arfitness[idx[0]]

            if local_best_f < best_local:
                best_local = local_best_f
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
                max_diag = np.max(diagC)
                diagC = np.maximum(diagC, max_diag * 1e-14)
            else:
                # Active CMA update
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
            if stag_count > 25 + 15 * n / lam:
                return best_local_x
            if gen > 300 + 200 * n / lam:
                return best_local_x

            if len(f_history) > 40:
                recent = f_history[-40:]
                if abs(recent[-1] - recent[0]) < 1e-14 * (abs(recent[0]) + 1e-30):
                    return best_local_x

        return best_local_x

    def c1_val(n, mueff):
        return 2 / ((n + 1.3)**2 + mueff)

    def cmu_val_f(n, mueff):
        return min(1 - c1_val(n, mueff), 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))

    # Run CMA-ES with BIPOP restarts
    default_lam = max(6, 4 + int(3 * np.log(dim)))
    large_budget_used = 0
    small_budget_used = 0
    restart = 0

    # First: run CMA from best known points
    n_elite_starts = min(5, len(elite_pool))
    for i in range(n_elite_starts):
        if not time_ok(0.50):
            break
        rem = time_left()
        x0 = elite_pool[i][1].copy()
        sig0 = 0.15 * np.mean(ranges) * (0.5 + np.random.random())
        if i == 0 and best_x is not None:
            x0 = best_x.copy()
            sig0 = 0.2 * np.mean(ranges)
        t_budget = elapsed() + min(rem * 0.25, rem / max(n_elite_starts - i, 1))
        cmaes_run(x0, sig0, max_time_abs=t_budget)
        restart += 1

    # BIPOP restarts
    while time_ok(0.85):
        rem = time_left()
        if rem < 0.5:
            break

        use_large = (large_budget_used <= small_budget_used) or np.random.random() < 0.3

        if use_large:
            power = min(restart - n_elite_starts + 1, 7)
            large_lam = int(default_lam * (2 ** (np.random.uniform(0, power))))
            large_lam = min(large_lam, max(512, 10 * dim))
            large_lam = max(large_lam, default_lam)

            if np.random.random() < 0.3 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * 0.4 * ranges
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sig0 = (0.2 + 0.3 * np.random.random()) * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.35, rem)
            cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
            large_budget_used += elapsed() - t_start
        else:
            small_lam = max(6, int(default_lam * (0.5 + 0.5 * np.random.random())))

            if best_x is not None and np.random.random() < 0.7:
                spread = 0.005 + 0.15 * np.random.random()
                x0 = best_x + np.random.randn(dim) * spread * ranges
                x0 = np.clip(x0, lower, upper)
                sig0 = (0.01 + 0.1 * np.random.random()) * np.mean(ranges)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.2 * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.15, rem)
            cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
            small_budget_used += elapsed() - t_start

        restart += 1

    # Phase 3: Nelder-Mead simplex from best point
    if best_x is not None and time_ok(0.90):
        def nelder_mead(x_start, max_t_abs, initial_scale=0.05):
            nonlocal best, best_x
            n = dim
            # Build simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start.copy()
            f_simplex = np.zeros(n + 1)
            f_simplex[0] = eval_f(simplex[0])

            for i in range(n):
                if not time_ok(0.95) or elapsed() >= max_t_abs:
                    return
                simplex[i + 1] = x_start.copy()
                step = initial_scale * ranges[i]
                if step < 1e-15:
                    step = 1e-6
                simplex[i + 1][i] += step
                simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
                f_simplex[i + 1] = eval_f(simplex[i + 1])

            alpha_r, gamma_e, rho_c, sigma_s = 1.0, 2.0, 0.5, 0.5

            for iteration in range(500 * n):
                if not time_ok(0.97) or elapsed() >= max_t_abs:
                    break

                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]

                # Centroid excluding worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha_r * (centroid - simplex[-1]), lower, upper)
                fr = eval_f(xr)

                if fr < f_simplex[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma_e * (xr - centroid), lower, upper)
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    if fr < f_simplex[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + rho_c * (xr - centroid), lower, upper)
                        fc = eval_f(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            f_simplex[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                if not time_ok(0.97) or elapsed() >= max_t_abs:
                                    return
                                simplex[i] = np.clip(simplex[0] + sigma_s * (simplex[i] - simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - rho_c * (centroid - simplex[-1]), lower, upper)
                        fc = eval_f(xc)
                        if fc < f_simplex[-1]:
                            simplex[-1] = xc
                            f_simplex[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                if not time_ok(0.97) or elapsed() >= max_t_abs:
                                    return
                                simplex[i] = np.clip(simplex[0] + sigma_s * (simplex[i] - simplex[0]), lower, upper)
                                f_simplex[i] = eval_f(simplex[i])

                # Convergence check
                f_range = abs(f_simplex[-1] - f_simplex[0])
                if f_range < 1e-15 * (abs(f_simplex[0]) + 1e-30):
                    break
                diam = np.max(np.abs(simplex[-1] - simplex[0]))
                if diam < 1e-15:
                    break

        rem = time_left()
        t_nm = elapsed() + min(rem * 0.5, rem)
        nelder_mead(best_x.copy(), t_nm, initial_scale=0.02)

    # Phase 4: Pattern search refinement
    if best_x is not None and time_ok(0.96):
        x_curr = best_x.copy()
        f_curr = best

        for scale_exp in range(-2, -8, -1):
            if not time_ok(0.98):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved_any = True
            while improved_any and time_ok(0.98):
                improved_any = False
                perm = np.random.permutation(dim)
                for i in perm:
                    if not time_ok(0.98):
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

    # Phase 5: Final gradient descent
    if best_x is not None and time_ok(0.985):
        x_curr = best_x.copy()
        f_curr = best
        h = 1e-8 * np.mean(ranges)

        for _ in range(3):
            if not time_ok(0.995):
                break
            grad = np.zeros(dim)
            for i in range(dim):
                if not time_ok(0.995):
                    break
                xp = x_curr.copy()
                xp[i] = min(x_curr[i] + h, upper[i])
                fp = eval_f(xp)
                xm = x_curr.copy()
                xm[i] = max(x_curr[i] - h, lower[i])
                fm = eval_f(xm)
                actual_h = xp[i] - xm[i]
                if abs(actual_h) > 1e-30:
                    grad[i] = (fp - fm) / actual_h

            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-30:
                d = -grad / gnorm
                # Line search with golden section
                for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]:
                    if not time_ok(0.998):
                        break
                    step = alpha * np.mean(ranges)
                    xt = np.clip(x_curr + step * d, lower, upper)
                    ft = eval_f(xt)
                    if ft < f_curr:
                        f_curr = ft
                        x_curr = xt

    return best
