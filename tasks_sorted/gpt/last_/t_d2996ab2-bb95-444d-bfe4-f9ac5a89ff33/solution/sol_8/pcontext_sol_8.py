import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libraries).

    Key upgrades vs previous version:
      - L-SHADE-like Differential Evolution with:
          * current-to-pbest/1 mutation
          * external archive (for diversity)
          * historical memories for F and CR (adaptation from successful trials)
          * linear population size reduction (more exploration early, faster later)
      - Stronger initialization: stratified + opposite points + elite jitter
      - Safer bound handling via reflection
      - Lightweight local refinement (coordinate pattern search) on the incumbent
      - Quantized cache to avoid repeat evaluations

    Returns:
      best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))
    eps = 1e-12

    # ----------------- trivial cases -----------------
    if dim <= 0:
        try:
            y = float(func([]))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [max(0.0, hi[i] - lo[i]) for i in range(dim)]
    max_span = max(span) if span else 0.0

    if max_span <= eps:
        x0 = lo[:]
        try:
            y = float(func(x0))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    def time_left():
        return time.time() < deadline

    def clip(v, a, b):
        return a if v < a else (b if v > b else v)

    def reflect_into(v, a, b):
        if a == b:
            return a
        w = b - a
        x = float(v)
        if x < a or x > b:
            x = a + (x - a) % (2.0 * w)
            if x > b:
                x = b - (x - b)
        return clip(x, a, b)

    def project_reflect(x):
        return [reflect_into(x[i], lo[i], hi[i]) for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) if span[i] > eps else lo[i] for i in range(dim)]

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            y[i] = lo[i] if span[i] <= eps else (lo[i] + hi[i] - x[i])
        return project_reflect(y)

    # ----------------- quantized cache -----------------
    # Slightly coarser than before to improve hit rate without harming optimization too much.
    qstep = []
    for i in range(dim):
        s = span[i]
        qstep.append(0.0 if s <= eps else max(1e-12, 8e-7 * s))
    cache = {}
    cache_keys = []
    CACHE_MAX = 80000

    def key_of(x):
        k = []
        for i in range(dim):
            if qstep[i] == 0.0:
                k.append(0)
            else:
                k.append(int(round((x[i] - lo[i]) / qstep[i])))
        return tuple(k)

    def cache_put(k, y):
        if k in cache:
            cache[k] = y
            return
        if len(cache) >= CACHE_MAX:
            # evict ~2%
            ev = max(80, CACHE_MAX // 50)
            for _ in range(ev):
                if not cache_keys:
                    break
                idx = random.randrange(len(cache_keys))
                kk = cache_keys[idx]
                cache_keys[idx] = cache_keys[-1]
                cache_keys.pop()
                cache.pop(kk, None)
                if len(cache) < CACHE_MAX:
                    break
        cache[k] = y
        cache_keys.append(k)

    def safe_eval(x):
        x = project_reflect(x)
        k = key_of(x)
        y = cache.get(k)
        if y is not None:
            return y, x
        try:
            y = func(x)
            y = float("inf") if y is None else float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")
        cache_put(k, y)
        return y, x

    # ----------------- light local search -----------------
    def pattern_search(x0, f0, step0, max_iter):
        x = x0[:]
        fx = f0
        step = step0
        coords = list(range(dim))
        it = 0
        while it < max_iter and time_left():
            it += 1
            improved = False
            random.shuffle(coords)
            for j in coords:
                if span[j] <= eps:
                    continue
                cur = x[j]
                # try both directions
                for sgn in (1.0, -1.0):
                    xt = x[:]
                    xt[j] = reflect_into(cur + sgn * step, lo[j], hi[j])
                    ft, xt = safe_eval(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                        break
                if not time_left():
                    break
            # expand if improving, shrink otherwise
            step *= (1.35 if improved else 0.6)
            if step <= 1e-14 * max_span:
                break
        return fx, x

    # ----------------- initialization -----------------
    best = float("inf")
    x_best = rand_uniform()
    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    # Create a decent initial pool; use stratification + opposition.
    init_n = max(50, 14 * dim)
    bins = max(5, int(math.sqrt(init_n)) + 1)

    pool = []
    for k in range(init_n):
        if not time_left():
            return best
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
            else:
                b = (random.randrange(bins) + k) % bins
                a = lo[i] + (b / bins) * span[i]
                c = lo[i] + ((b + 1) / bins) * span[i]
                x[i] = random.uniform(a, c)
        fx, x = safe_eval(x)
        pool.append((fx, x))
        if fx < best:
            best, x_best = fx, x[:]
        if (k & 1) == 0 and time_left():
            xo = opposition(x)
            fo, xo = safe_eval(xo)
            pool.append((fo, xo))
            if fo < best:
                best, x_best = fo, xo[:]

    pool.sort(key=lambda t: t[0])

    # ----------------- L-SHADE-like DE -----------------
    # Population sizing
    NP_max = max(24, min(120, 12 * dim))
    NP_min = max(10, min(28, 4 * dim))

    # If max_time is tiny, keep smaller population to increase generations
    if max_time <= 1.0:
        NP_max = max(18, min(NP_max, 40))
        NP_min = max(10, min(NP_min, 20))

    # Initial population from best of pool + jitter + random top-ups
    pop = []
    fit = []
    take = min(len(pool), NP_max)
    for i in range(take):
        x = pool[i][1][:]
        # jitter to avoid duplicates (stronger early)
        for j in range(dim):
            if span[j] > eps and random.random() < 0.75:
                x[j] = reflect_into(x[j] + (random.random() * 2 - 1) * 0.03 * span[j], lo[j], hi[j])
        fx, x = safe_eval(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, x_best = fx, x[:]

    while len(pop) < NP_max and time_left():
        x = rand_uniform()
        fx, x = safe_eval(x)
        pop.append(x)
        fit.append(fx)
        if fx < best:
            best, x_best = fx, x[:]

    if not pop:
        return best

    # Archive for diversity (A)
    archive = []
    archive_fit = []
    A_MAX = NP_max

    def archive_add(x, fx):
        # store unique-ish by cache key (cheap)
        if len(archive) >= A_MAX:
            idx = random.randrange(len(archive))
            archive[idx] = x[:]
            archive_fit[idx] = fx
        else:
            archive.append(x[:])
            archive_fit.append(fx)

    # Memories for F and CR (historical successful values)
    H = 6  # small memory works fine under tight budgets
    M_F = [0.5] * H
    M_CR = [0.9] * H
    mem_idx = 0

    def cauchy(mu, gamma):
        # sample from Cauchy(mu, gamma) using inverse CDF
        return mu + gamma * math.tan(math.pi * (random.random() - 0.5))

    def normal(mu, sigma):
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def pick_distinct_indices(n, exclude, k):
        # returns k distinct indices in [0,n) excluding 'exclude'
        res = []
        while len(res) < k:
            r = random.randrange(n - 1)
            r = r + 1 + exclude if r >= exclude else r
            ok = True
            for v in res:
                if v == r:
                    ok = False
                    break
            if ok:
                res.append(r)
        return res

    def pbest_index(sorted_idx, p):
        # choose random index among top p fraction in sorted_idx list
        top = max(2, int(math.ceil(p * len(sorted_idx))))
        top = min(top, len(sorted_idx))
        return sorted_idx[random.randrange(top)]

    t_start = time.time()
    last_polish = t_start
    no_improve = 0
    gen = 0

    while time_left():
        gen += 1
        n = len(pop)
        if n < 6:
            # rebuild minimal viable pop
            while len(pop) < 6 and time_left():
                x = rand_uniform()
                fx, x = safe_eval(x)
                pop.append(x)
                fit.append(fx)
                if fx < best:
                    best, x_best = fx, x[:]
            n = len(pop)

        # Sort indices by fitness (ascending)
        sorted_idx = sorted(range(n), key=lambda i: fit[i])

        # p decreases slightly with dimension; keeps more exploitation in low-d
        p = min(0.3, max(0.08, 2.0 / max(2.0, math.sqrt(dim))))
        # Successful parameter storage
        S_F = []
        S_CR = []
        dF = []  # fitness improvements for weighted update

        prev_best = best

        # Determine target population size by linear reduction over time
        # Use elapsed fraction as proxy for progress.
        frac = (time.time() - t0) / max(1e-9, (deadline - t0))
        frac = clip(frac, 0.0, 1.0)
        target_NP = int(round(NP_max - frac * (NP_max - NP_min)))
        target_NP = max(NP_min, min(NP_max, target_NP))

        # One generation
        for ii in range(n):
            if not time_left():
                break

            i = ii  # keep original order for speed
            xi = pop[i]
            fi = fit[i]

            # choose memory slot
            r = random.randrange(H)
            mu_F = M_F[r]
            mu_CR = M_CR[r]

            # sample CR ~ N(mu, 0.1), clipped
            CRi = normal(mu_CR, 0.1)
            CRi = clip(CRi, 0.0, 1.0)

            # sample F ~ Cauchy(mu, 0.1) until in (0,1]
            Fi = cauchy(mu_F, 0.1)
            tries = 0
            while (Fi <= 0.0 or Fi > 1.0) and tries < 6:
                Fi = cauchy(mu_F, 0.1)
                tries += 1
            if Fi <= 0.0:
                Fi = 0.5
            if Fi > 1.0:
                Fi = 1.0

            # Mutation: current-to-pbest/1 with archive
            ipbest = pbest_index(sorted_idx, p)
            xpbest = pop[ipbest]

            r1, r2 = pick_distinct_indices(n, i, 2)
            xr1 = pop[r1]

            # r2 can come from pop or archive
            use_archive = (archive and random.random() < 0.5)
            if use_archive:
                xr2 = archive[random.randrange(len(archive))]
            else:
                xr2 = pop[r2]

            v = [0.0] * dim
            for j in range(dim):
                v[j] = xi[j] + Fi * (xpbest[j] - xi[j]) + Fi * (xr1[j] - xr2[j])

            # Binomial crossover
            jrand = random.randrange(dim)
            u = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CRi:
                    u[j] = v[j]
            u = project_reflect(u)

            fu, u = safe_eval(u)
            if fu < best:
                best, x_best = fu, u[:]

            # Selection + archive update
            if fu <= fi:
                # parent goes to archive (diversity)
                archive_add(xi, fi)

                pop[i] = u
                fit[i] = fu

                # store successful params if strict improvement (or tiny tie improvement)
                if fu < fi:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    dF.append(fi - fu)

        # Update memories using weighted Lehmer mean for F and weighted mean for CR
        if S_F:
            # weights proportional to fitness improvements
            wsum = sum(dF) + 1e-300
            # Lehmer mean for F: sum(w*F^2)/sum(w*F)
            numF = 0.0
            denF = 0.0
            numCR = 0.0
            for k in range(len(S_F)):
                w = dF[k] / wsum
                f = S_F[k]
                cr = S_CR[k]
                numF += w * f * f
                denF += w * f
                numCR += w * cr
            if denF <= 1e-300:
                new_MF = 0.5
            else:
                new_MF = numF / denF
            new_MCR = numCR

            M_F[mem_idx] = clip(new_MF, 0.05, 1.0)
            M_CR[mem_idx] = clip(new_MCR, 0.0, 1.0)
            mem_idx = (mem_idx + 1) % H

        # Reduce population size by removing worst individuals
        if len(pop) > target_NP:
            # remove worst to reach target
            # get worst indices sorted descending by fitness
            worst = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
            remove = len(pop) - target_NP
            to_remove = set(worst[:remove])
            pop = [pop[i] for i in range(len(pop)) if i not in to_remove]
            fit = [fit[i] for i in range(len(fit)) if i not in to_remove]
            # shrink archive cap as well
            A_MAX = max(target_NP, NP_min)
            while len(archive) > A_MAX:
                idx = random.randrange(len(archive))
                archive[idx] = archive[-1]
                archive.pop()
                archive_fit[idx] = archive_fit[-1]
                archive_fit.pop()

        # Stagnation + polishing schedule
        if best < prev_best:
            no_improve = 0
        else:
            no_improve += 1

        # occasional local polish (more often when stagnating)
        now = time.time()
        polish_period = 0.15 if dim <= 10 else 0.22
        if time_left() and (now - last_polish) > polish_period and (no_improve >= 2 or gen % 6 == 0):
            last_polish = now
            f0, x0 = safe_eval(x_best[:])
            step0 = max(1e-10 * max_span, 0.02 * max_span / (1.0 + 0.03 * gen))
            fl, xl = pattern_search(x0, f0, step0, max_iter=max(8, 2 * dim))
            if fl < best:
                best, x_best = fl, xl[:]
            # inject into worst
            iw = max(range(len(pop)), key=lambda i: fit[i])
            if fl < fit[iw]:
                pop[iw] = xl[:]
                fit[iw] = fl

        # Stronger restart injection if stagnation persists
        if no_improve >= 5 and time_left():
            no_improve = 0
            n = len(pop)
            # replace ~20% worst
            repl = max(2, n // 5)
            worst_idx = sorted(range(n), key=lambda i: fit[i], reverse=True)[:repl]
            # top elites for sampling
            sorted_idx = sorted(range(n), key=lambda i: fit[i])
            topk = max(2, min(n, max(6, n // 4)))
            for wi in worst_idx:
                if not time_left():
                    break
                # sample around a good point with heavy tails + occasional opposition
                base = pop[sorted_idx[int((random.random() ** 1.8) * topk)]]
                x = base[:]
                for j in range(dim):
                    if span[j] <= eps:
                        x[j] = lo[j]
                    else:
                        u = random.random()
                        if u < 0.7:
                            step = (random.random() * 2 - 1) * 0.10 * span[j]
                        else:
                            step = 0.05 * span[j] * math.tan(math.pi * (random.random() - 0.5))
                        x[j] = reflect_into(x[j] + step, lo[j], hi[j])
                if random.random() < 0.25:
                    x = opposition(x)
                fx, x = safe_eval(x)
                pop[wi] = x
                fit[wi] = fx
                if fx < best:
                    best, x_best = fx, x[:]

    return best
