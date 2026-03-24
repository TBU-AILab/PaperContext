import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Upgrade vs previous:
      - Adds *CMA-ES-lite* (diagonal covariance, rank-1 update + evolution path).
        This is often much stronger than DE/NM on smooth-ish continuous problems.
      - Keeps cheap global exploration (Sobol-ish LHS + opposition + heavy-tail).
      - Keeps a bounded coordinate pattern search as a reliable last-mile improver.
      - Uses strict time checks and a quantized cache to avoid repeats.

    Returns:
      best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))
    eps = 1e-12

    # --- trivial cases ---
    if dim <= 0:
        try:
            y = float(func([]))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [max(hi[i] - lo[i], 0.0) for i in range(dim)]
    max_span = max(span) if span else 0.0

    if max_span <= eps:
        x0 = [lo[i] for i in range(dim)]
        try:
            y = float(func(x0))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    def time_left():
        return time.time() < deadline

    def clip(v, a, b):
        return a if v < a else (b if v > b else v)

    def project(x):
        return [clip(float(x[i]), lo[i], hi[i]) for i in range(dim)]

    def center():
        return [(lo[i] + hi[i]) * 0.5 for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) if span[i] > eps else lo[i] for i in range(dim)]

    # --- cache (quantized) ---
    qstep = []
    for i in range(dim):
        s = span[i]
        if s <= eps:
            qstep.append(0.0)
        else:
            qstep.append(max(1e-12, 2e-7 * s))

    cache = {}
    cache_keys = []
    CACHE_MAX = 40000

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
            # random evict ~2%
            for _ in range(max(20, CACHE_MAX // 50)):
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
        x = project(x)
        k = key_of(x)
        y = cache.get(k)
        if y is not None:
            return y, x
        try:
            y = func(x)
            if y is None:
                y = float("inf")
            y = float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")
        cache_put(k, y)
        return y, x

    # --- elites ---
    best = float("inf")
    x_best = center()
    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    ELITE_MAX = max(18, 6 * dim)
    elites = [(best, x_best[:])]

    def push_elite(fx, x):
        nonlocal best, x_best, elites
        if fx < best:
            best = fx
            x_best = x[:]
        elites.append((fx, x[:]))
        if len(elites) > 5 * ELITE_MAX:
            elites.sort(key=lambda t: t[0])
            elites[:] = elites[:ELITE_MAX]

    # --- sampling helpers ---
    def stratified(m, seed_shift=0):
        # simple LHS-like stratified per-dimension (independent bins)
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
            else:
                b = (random.randrange(m) + seed_shift) % m
                a = lo[i] + (b / m) * span[i]
                c = lo[i] + ((b + 1) / m) * span[i]
                x[i] = random.uniform(a, c)
        return x

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                y[i] = lo[i]
            else:
                o = lo[i] + hi[i] - x[i]
                mid = 0.5 * (lo[i] + hi[i])
                y[i] = clip(mid + random.random() * (o - mid), lo[i], hi[i])
        return y

    # --- local search: coordinate pattern search ---
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
                # try +/- step
                for sgn in (1.0, -1.0):
                    xj = clip(cur + sgn * step, lo[j], hi[j])
                    if abs(xj - cur) <= 1e-18:
                        continue
                    xt = x[:]
                    xt[j] = xj
                    ft, xt = safe_eval(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                        break
                if improved and not time_left():
                    break
            step = step * (1.25 if improved else 0.5)
            if step <= 1e-11 * (max_span if max_span > 0 else 1.0):
                break
        return fx, x

    # --- CMA-ES-lite (diagonal) ---
    # We maintain:
    #   m: mean
    #   sigma: global step
    #   D: per-coordinate scaling (sqrt of diag covariance)
    #   p: evolution path
    def cma_diag_optimize(m0, sigma0, budget_evals, lam=None):
        nonlocal best, x_best
        if budget_evals <= 0 or not time_left():
            return

        n = dim
        # population size
        if lam is None:
            lam = max(8, 4 + int(3 * math.log(n + 1.0)))
        mu = max(2, lam // 2)

        # recombination weights (log)
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(w)
        w = [wi / wsum for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # strategy parameters (diagonal variant)
        # conservative for robustness
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)  # small
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

        # expected length of N(0,I)
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        m = m0[:]
        sigma = max(1e-15, float(sigma0))
        # init diagonal scaling
        D = [1.0] * n
        p = [0.0] * n

        evals = 0
        gen = 0

        # helper: sample gaussian using Box-Muller
        spare = [None]
        def gauss01():
            if spare[0] is not None:
                z = spare[0]
                spare[0] = None
                return z
            u1 = max(1e-300, random.random())
            u2 = random.random()
            r = math.sqrt(-2.0 * math.log(u1))
            z0 = r * math.cos(2.0 * math.pi * u2)
            z1 = r * math.sin(2.0 * math.pi * u2)
            spare[0] = z1
            return z0

        while evals < budget_evals and time_left():
            gen += 1
            # sample population
            pop = []
            for _ in range(lam):
                if evals >= budget_evals or not time_left():
                    break
                z = [gauss01() for _ in range(n)]
                x = [0.0] * n
                for j in range(n):
                    if span[j] <= eps:
                        x[j] = lo[j]
                    else:
                        x[j] = m[j] + sigma * D[j] * z[j]
                fx, x = safe_eval(x)
                evals += 1
                pop.append((fx, x, z))
                push_elite(fx, x)

            if len(pop) < 2:
                break

            pop.sort(key=lambda t: t[0])
            # new mean (recombination)
            m_old = m[:]
            m = [0.0] * n
            zmean = [0.0] * n
            for i in range(mu):
                wi = w[i]
                xi = pop[i][1]
                zi = pop[i][2]
                for j in range(n):
                    m[j] += wi * xi[j]
                    zmean[j] += wi * zi[j]

            # evolution path update (in z-space because diagonal)
            ccs = math.sqrt(cs * (2.0 - cs) * mueff)
            for j in range(n):
                p[j] = (1.0 - cs) * p[j] + ccs * zmean[j]

            # step-size control
            pnorm = math.sqrt(sum(pj * pj for pj in p))
            sigma *= math.exp((cs / damps) * (pnorm / chi_n - 1.0))

            # covariance (diagonal) rank-1 update using p
            # D^2 <- (1-c1)*D^2 + c1*(p^2)
            one_c1 = 1.0 - c1
            for j in range(n):
                dj2 = D[j] * D[j]
                dj2 = one_c1 * dj2 + c1 * (p[j] * p[j] + 1e-30)
                D[j] = math.sqrt(max(1e-30, dj2))

            # keep mean in-bounds (important with clipping)
            m = project(m)

            # mild restart/diversification if sigma collapses or stagnation
            if sigma <= 1e-14 * (max_span if max_span > 0 else 1.0):
                break

            # occasional "inject best" to stabilize
            if gen % 7 == 0 and time_left():
                m = x_best[:]

        # end CMA loop

    # --- initial exploration ---
    init_n = max(80, 25 * dim)
    m_bins = max(6, int(math.sqrt(init_n)) + 2)

    for k in range(init_n):
        if not time_left():
            return best
        x = stratified(m_bins, seed_shift=k) if (k % 2 == 0) else rand_uniform()
        fx, x = safe_eval(x)
        push_elite(fx, x)
        if (k % 3 == 0) and time_left():
            xo = opposition(x)
            fo, xo = safe_eval(xo)
            push_elite(fo, xo)

    # --- main loop: CMA bursts + local polish + diversification ---
    round_id = 0
    no_improve = 0

    while time_left():
        round_id += 1
        elites.sort(key=lambda t: t[0])
        elites = elites[:min(len(elites), ELITE_MAX)]
        prev_best = best

        # CMA-ES-lite burst around best (and sometimes around another elite)
        # set sigma from box size
        base_sigma = 0.18 * max_span / math.sqrt(dim)
        base_sigma = max(base_sigma, 1e-9 * max_span)

        # budget small but frequent
        burst_evals = max(40, 10 * dim)
        cma_diag_optimize(x_best[:], base_sigma / (1.0 + 0.08 * round_id), burst_evals)

        if time_left() and len(elites) > 3 and (round_id % 3 == 0):
            # second burst around a different elite
            idx = min(len(elites) - 1, 1 + int((random.random() ** 1.8) * (len(elites) - 1)))
            m0 = elites[idx][1]
            cma_diag_optimize(m0[:], 0.7 * base_sigma, max(25, 6 * dim))

        # local polish on best
        if time_left():
            step0 = (0.06 / (1.0 + 0.10 * round_id)) * max_span
            step0 = max(step0, 1e-7 * max_span)
            f0, x0 = safe_eval(x_best[:])
            fl, xl = pattern_search(x0, f0, step0, max_iter=max(10, 2 * dim))
            push_elite(fl, xl)

        # diversification if stagnating
        if best < prev_best:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 2:
            burst = max(40, 12 * dim)
            for _ in range(burst):
                if not time_left():
                    return best
                r = random.random()
                if r < 0.45:
                    x = rand_uniform()
                    if random.random() < 0.5:
                        x = opposition(x)
                else:
                    # heavy-tail around current best
                    x = x_best[:]
                    for j in range(dim):
                        if span[j] <= eps:
                            x[j] = lo[j]
                            continue
                        u = random.random()
                        step = (0.10 * span[j]) * math.tan(math.pi * (u - 0.5))  # Cauchy-like
                        x[j] = clip(x[j] + step, lo[j], hi[j])
                fx, x = safe_eval(x)
                push_elite(fx, x)
            no_improve = 0

    return best
