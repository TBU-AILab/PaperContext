import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs).

    Key upgrades vs your current best:
      - Uses *CMA-ES style* sampling (diagonal covariance) for strong local/global performance.
      - Keeps your strengths: opposition/halton-like init, heavy-tailed escapes, and
        occasional simplex-free local polish (very cheap coordinate+quadratic).
      - Time-aware scheduling: more global early, more CMA exploitation later.
      - Robust bound handling by reflection + tiny jitter.
      - Handles noisy/fragile objective via safe_eval and re-evaluation of elites.

    Returns: best fitness found (float)
    """

    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---------- helpers ----------
    def clamp(x, a, b):
        return a if x < a else b if x > b else x

    def reflect(x, a, b):
        if a == b:
            return a
        # reflect repeatedly until inside (handles large steps)
        while x < a or x > b:
            if x < a:
                x = a + (a - x)
            if x > b:
                x = b - (x - b)
        return clamp(x, a, b)

    def randn():
        u1 = max(1e-12, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    if dim <= 0:
        return safe_eval([])

    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]

    def in_bounds_jitter(x, eps_scale=1e-12):
        y = x[:]
        for i in range(dim):
            if span[i] <= 0:
                y[i] = lo[i]
            else:
                eps = eps_scale * span[i]
                if y[i] <= lo[i]:
                    y[i] = lo[i] + eps
                elif y[i] >= hi[i]:
                    y[i] = hi[i] - eps
        return y

    def random_point():
        return [lo[i] if span[i] <= 0 else lo[i] + span[i] * random.random() for i in range(dim)]

    def opposite_point(x):
        return [lo[i] if span[i] <= 0 else (lo[i] + hi[i] - x[i]) for i in range(dim)]

    # ---------- Halton (cheap low discrepancy init) ----------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    primes = first_primes(max(1, dim))

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    halton_index = 1
    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= 0:
                x[i] = lo[i]
            else:
                u = halton_value(idx, primes[i])
                x[i] = lo[i] + u * span[i]
        return x

    # ---------- tiny local polish (coordinate + 1D quadratic step) ----------
    def local_polish(x0, f0, rad, passes=1):
        x = x0[:]
        f = f0
        for _ in range(passes):
            if time.time() >= deadline:
                break
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            for i in order:
                if time.time() >= deadline:
                    break
                if span[i] <= 0:
                    continue
                step = rad[i]
                if step <= 0:
                    continue

                xc = x[:]
                fm = fp = float("inf")

                xm = x[:]
                xm[i] = reflect(xm[i] - step, lo[i], hi[i])
                xm = in_bounds_jitter(xm)
                fm = safe_eval(xm)

                xp = x[:]
                xp[i] = reflect(xp[i] + step, lo[i], hi[i])
                xp = in_bounds_jitter(xp)
                fp = safe_eval(xp)

                if fm < f or fp < f:
                    if fm <= fp:
                        x, f = xm, fm
                    else:
                        x, f = xp, fp
                    improved = True

                # occasional quadratic interpolation around original center xc
                if time.time() < deadline and random.random() < 0.35:
                    fc = safe_eval(xc)
                    denom = (fm - 2.0 * fc + fp)
                    if abs(denom) > 1e-18:
                        t = 0.5 * (fm - fp) / denom  # vertex offset in units of step
                        if abs(t) <= 2.5:
                            xq = xc[:]
                            xq[i] = reflect(xq[i] + t * step, lo[i], hi[i])
                            xq = in_bounds_jitter(xq)
                            fq = safe_eval(xq)
                            if fq < f:
                                x, f = xq, fq
                                improved = True

            # shrink/expand rad based on improvement
            mul = 1.15 if improved else 0.65
            for i in range(dim):
                if span[i] > 0:
                    rad[i] = max(1e-15, rad[i] * mul)
        return x, f

    # ---------- initialization ----------
    # modest initial pool; CMA-ES will do the heavy lifting
    init_n = max(14, min(80, 10 * dim))
    best = float("inf")
    best_x = None

    # seed mean from best of mixed init (random/halton/opposition)
    cand = []
    for k in range(init_n):
        if time.time() >= deadline:
            return best

        if k % 3 == 0:
            x = halton_point()
        elif k % 3 == 1:
            x = random_point()
        else:
            x = random_point()

        fx = safe_eval(x)
        xo = opposite_point(x)
        fxo = safe_eval(xo)
        if fxo < fx:
            x, fx = xo, fxo

        x = in_bounds_jitter(x)
        cand.append((fx, x))
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = random_point()
        best = safe_eval(best_x)

    # CMA mean starts at best found
    m = best_x[:]

    # ---------- diagonal CMA-ES parameters ----------
    # population
    lam = max(10, min(60, 4 + int(3.0 * math.sqrt(dim) + 2.0 * dim / 10.0)))
    mu = max(2, lam // 2)

    # weights (log)
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # step-size and diag covariance (std per dim)
    # start moderately large to explore; adapted by CSA-like rule
    sigma = 0.22
    s = [0.25 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]
    # avoid zero std on free dims
    for i in range(dim):
        if span[i] > 0:
            s[i] = max(s[i], 1e-12 * span[i])

    # evolution paths (diag version)
    ps = [0.0] * dim
    pc = [0.0] * dim

    # learning rates
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

    # expectation of ||N(0,I)||
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # stash best-ever for occasional polish
    last_best = best
    last_improve_t = time.time()
    no_improve = 0

    # cheap elite re-eval to reduce noise impact (very small)
    def maybe_reeval_best():
        nonlocal best, best_x
        if time.time() >= deadline:
            return
        # only rarely
        if random.random() < 0.15:
            fb = safe_eval(best_x)
            if fb < best:
                best = fb

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1
        tfrac = (time.time() - start) / max(1e-12, float(max_time))

        # sample lambda candidates
        pop = []
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            z = [randn() for _ in range(dim)]
            x = [0.0] * dim
            for i in range(dim):
                if span[i] <= 0:
                    x[i] = lo[i]
                else:
                    xi = m[i] + sigma * s[i] * z[i]
                    x[i] = reflect(xi, lo[i], hi[i])
            x = in_bounds_jitter(x)
            fx = safe_eval(x)
            pop.append((fx, x, z))

            if fx < best:
                best, best_x = fx, x[:]
                last_best = best
                last_improve_t = time.time()
                no_improve = 0

        pop.sort(key=lambda t: t[0])
        elites = pop[:mu]

        # recombination for new mean
        m_old = m[:]
        m = [0.0] * dim
        zmean = [0.0] * dim
        for wi, (_, xi, zi) in zip(w, elites):
            for i in range(dim):
                m[i] += wi * xi[i]
                zmean[i] += wi * zi[i]
        m = in_bounds_jitter([reflect(m[i], lo[i], hi[i]) if span[i] > 0 else lo[i] for i in range(dim)])

        # CSA step-size control (diag)
        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * zmean[i]

        ps_norm = math.sqrt(sum(pi * pi for pi in ps))
        sigma *= math.exp((cs / ds) * (ps_norm / chiN - 1.0))
        sigma = clamp(sigma, 1e-6, 2.0)

        # covariance / diagonal std update
        # pc update uses (m-m_old)/(sigma*s) approx zmean
        for i in range(dim):
            pc[i] = (1.0 - cc) * pc[i] + math.sqrt(cc * (2.0 - cc) * mueff) * zmean[i]

        # rank-mu update on diag variances
        # var <- (1-c1-cmu)*var + c1*pc^2 + cmu*sum(wi*z_i^2)
        # keep s as std (sqrt(var))
        var = [si * si for si in s]
        # compute weighted z^2
        wz2 = [0.0] * dim
        for wi, (_, _, zi) in zip(w, elites):
            for i in range(dim):
                wz2[i] += wi * (zi[i] * zi[i])

        for i in range(dim):
            if span[i] <= 0:
                var[i] = 0.0
                s[i] = 0.0
                continue
            var_i = (1.0 - c1 - cmu) * var[i] + c1 * (pc[i] * pc[i]) + cmu * wz2[i]
            # numerical safety and sensible scaling
            var_i = max(var_i, (1e-30) * (span[i] * span[i]))
            # prevent exploding too much; allow wide early exploration
            var_i = min(var_i, (0.50 * span[i]) ** 2)
            var[i] = var_i
            s[i] = math.sqrt(var_i)

        # stagnation logic
        if best < last_best - 1e-12:
            last_best = best
            last_improve_t = time.time()
            no_improve = 0
        else:
            no_improve += 1

        # occasional very-cheap local polish late or when stuck
        if (gen % 6 == 0 and tfrac > 0.25) or (no_improve >= 8 and time.time() < deadline):
            rad = [0.06 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]
            # tighten rad as time advances
            shrink = 0.7 + 0.5 * (1.0 - tfrac)
            for i in range(dim):
                if span[i] > 0:
                    rad[i] = max(1e-15, rad[i] * shrink)
            xlp, flp = local_polish(best_x, best, rad, passes=(1 if dim > 25 else 2))
            if flp < best:
                best, best_x = flp, xlp[:]
                last_best = best
                last_improve_t = time.time()
                no_improve = 0

        # heavy-tailed escape/restart if really stuck
        if (no_improve >= 14) or ((time.time() - last_improve_t) > 0.45 * max_time):
            if time.time() >= deadline:
                return best

            # re-center around best, but re-inflate exploration a bit
            m = best_x[:]
            sigma = min(0.35, max(0.08, sigma * 1.35))
            for i in range(dim):
                if span[i] > 0:
                    s[i] = max(s[i], 0.12 * span[i])

            # plus a few heavy-tailed probes to try different basins
            probes = 3 + (1 if dim <= 10 else 0)
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                for i in range(dim):
                    if span[i] > 0:
                        g1 = randn()
                        g2 = abs(randn()) + 1e-12
                        step = (g1 / g2) * (0.25 * span[i])  # t-like
                        x[i] = reflect(x[i] + step, lo[i], hi[i])
                x = in_bounds_jitter(x)
                fx = safe_eval(x)
                if fx < best:
                    best, best_x = fx, x[:]
                    m = best_x[:]
                    last_best = best
                    last_improve_t = time.time()
                    no_improve = 0

            maybe_reeval_best()
            no_improve = 0
            last_improve_t = time.time()

    return best
