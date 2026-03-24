import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs prior version:
      - Better bound handling for adaptation: internal search is done in an *unbounded*
        transformed space (logit), so CMA-style updates are not distorted by reflection/clamping.
      - Uses a stronger hybrid: (1) diagonal CMA-ES-like search in transformed space
        (2) adaptive local coordinate search around the incumbent in *original* space
        (3) occasional heavy-tailed/global steps + restarts on stagnation
      - Quasi-random (Halton) seeding + opposition + mid/jitter seeding.
      - Time-aware: estimates evaluation speed and sizes populations / local budgets accordingly.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    var_idx = [i for i in range(dim) if spans[i] > 0.0]

    if not var_idx:
        x0 = [lows[i] for i in range(dim)]
        return float(func(x0))

    # ----------------- helpers -----------------
    def now():
        return time.time()

    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    # avoid infinities in transform
    EPS_B = 1e-12

    def x_to_y_scalar(x, lo, hi):
        """Map x in [lo,hi] to y in R via stabilized logit."""
        if hi == lo:
            return 0.0
        w = hi - lo
        u = (x - lo) / w
        u = clamp(u, EPS_B, 1.0 - EPS_B)
        return math.log(u / (1.0 - u))

    def y_to_x_scalar(y, lo, hi):
        """Map y in R to x in [lo,hi] via sigmoid."""
        if hi == lo:
            return lo
        # stable sigmoid
        if y >= 0:
            e = math.exp(-y)
            s = 1.0 / (1.0 + e)
        else:
            e = math.exp(y)
            s = e / (1.0 + e)
        return lo + (hi - lo) * s

    def x_to_y(x):
        y = [0.0] * n
        for k, j in enumerate(var_idx):
            y[k] = x_to_y_scalar(x[j], lows[j], highs[j])
        return y

    def y_to_x(y, x_template=None):
        x = (x_template[:] if x_template is not None else [lows[i] for i in range(dim)])
        for i in range(dim):
            if spans[i] <= 0.0:
                x[i] = lows[i]
        for k, j in enumerate(var_idx):
            x[j] = y_to_x_scalar(y[k], lows[j], highs[j])
        return x

    # evaluation accounting (for time awareness)
    eval_count = 0
    eval_time = 0.0

    def eval_f(x):
        nonlocal eval_count, eval_time
        t = time.time()
        fx = float(func(x))
        eval_time += (time.time() - t)
        eval_count += 1
        return fx

    def evals_per_sec():
        spent = max(1e-9, eval_time)
        return eval_count / spent

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if spans[i] > 0.0:
                x[i] = lows[i] + random.random() * spans[i]
            else:
                x[i] = lows[i]
        return x

    def opposite_point(x):
        return [lows[i] + highs[i] - x[i] if spans[i] > 0.0 else lows[i] for i in range(dim)]

    # Halton init (fast simple)
    def first_primes(npr):
        primes = []
        p = 2
        while len(primes) < npr:
            ok = True
            r = int(math.isqrt(p))
            for q in primes:
                if q > r:
                    break
                if p % q == 0:
                    ok = False
                    break
            if ok:
                primes.append(p)
            p += 1
        return primes

    def van_der_corput(idx, base):
        v = 0.0
        denom = 1.0
        while idx:
            idx, rem = divmod(idx, base)
            denom *= base
            v += rem / denom
        return v

    primes = first_primes(max(1, dim))
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = (van_der_corput(idx, primes[i]) + halton_shift[i]) % 1.0
            x[i] = lows[i] + u * spans[i] if spans[i] > 0.0 else lows[i]
        return x

    # heavy-tailed step
    def cauchy():
        # tan(pi*(u-0.5))
        u = random.random()
        u = clamp(u, 1e-12, 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # ----------------- local coordinate search (in x-space) -----------------
    def local_search(x_best, f_best, budget, init_frac):
        if budget <= 0 or now() >= deadline:
            return f_best, x_best

        x = x_best[:]
        fx = f_best
        used = 0

        steps = [0.0] * dim
        for j in var_idx:
            steps[j] = max(1e-14, init_frac * spans[j])

        dims = var_idx[:]
        dims.sort(key=lambda j: spans[j], reverse=True)

        noimp = 0
        while used < budget and now() < deadline:
            best_cand = None
            best_fc = fx

            # evaluate coordinate +/- steps and also a small quadratic probe occasionally
            for j in dims:
                if used >= budget or now() >= deadline:
                    break
                s = steps[j]
                if s <= 0.0:
                    continue

                xj = x[j]
                # try +/- s
                for sgn in (1.0, -1.0):
                    y = x[:]
                    y[j] = clamp(xj + sgn * s, lows[j], highs[j])
                    fy = eval_f(y)
                    used += 1
                    if fy < best_fc:
                        best_fc = fy
                        best_cand = y
                    if used >= budget or now() >= deadline:
                        break

            if best_cand is not None and best_fc < fx:
                x, fx = best_cand, best_fc
                for j in var_idx:
                    steps[j] *= 1.25
                noimp = 0
            else:
                for j in var_idx:
                    steps[j] *= 0.5
                noimp += 1
                if noimp >= 2:
                    break

            # stop if all steps tiny
            tiny = True
            for j in var_idx:
                if steps[j] > 1e-12 * spans[j]:
                    tiny = False
                    break
            if tiny:
                break

        return fx, x

    # ----------------- initialization -----------------
    # probe once to estimate eval speed
    x_probe = rand_point()
    f_probe = eval_f(x_probe)
    x_best = x_probe[:]
    f_best = f_probe

    n = len(var_idx)

    eps = evals_per_sec()
    init_n = 10 + 4 * n
    if max_time <= 1.0:
        init_n = min(init_n, 24)
    if eps < 150:
        init_n = max(10, init_n // 2)

    mid = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    for i in range(init_n):
        if now() >= deadline:
            return f_best
        r = i % 6
        if r == 0:
            x = halton_point()
        elif r == 1:
            x = rand_point()
        elif r == 2:
            x = mid[:]
            for j in var_idx:
                x[j] = clamp(x[j] + random.gauss(0.0, 0.30 * spans[j]), lows[j], highs[j])
        elif r == 3:
            x = halton_point()
            for j in var_idx:
                x[j] = clamp(x[j] + random.gauss(0.0, 0.15 * spans[j]), lows[j], highs[j])
        elif r == 4:
            x = rand_point()
            xo = opposite_point(x)
            fx = eval_f(x)
            if fx < f_best:
                f_best, x_best = fx, x[:]
            if now() >= deadline:
                return f_best
            fo = eval_f(xo)
            if fo < f_best:
                f_best, x_best = fo, xo[:]
            continue
        else:
            # small random walk from current best
            x = x_best[:]
            for j in var_idx:
                x[j] = clamp(x[j] + random.gauss(0.0, 0.10 * spans[j]), lows[j], highs[j])

        fx = eval_f(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]

    # quick local tighten once after init
    if now() < deadline:
        fb2, xb2 = local_search(x_best, f_best, budget=4 * n + 8, init_frac=0.10)
        if fb2 < f_best:
            f_best, x_best = fb2, xb2[:]

    # ----------------- Diagonal CMA-ES-like in transformed y-space -----------------
    # state in y
    m = x_to_y(x_best)

    # per-dim std in y-space; start moderate, with floor
    sigma = 0.8
    sigma_min = 1e-12
    sigma_max = 3.0
    D = [1.0] * n  # diag scales in y; starts isotropic

    # population size
    eps = evals_per_sec()
    lam = 4 + int(3 * math.log(n + 1.0))
    if eps > 1200:
        lam = min(64, lam + 12 + 2 * n)
    elif eps > 500:
        lam = min(56, lam + 8 + n)
    if eps < 150:
        lam = max(6, min(lam, 10 + n))
    lam = int(clamp(lam, 6, 80))
    mu = lam // 2

    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    c_sigma = (mueff + 2.0) / (n + mueff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))

    p_sigma = [0.0] * n
    p_c = [0.0] * n

    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)) if n > 0 else 1.0

    gen = 0
    last_improve_gen = 0
    best_seen = f_best

    # template for fixed dims when mapping from y
    x_template = x_best[:]

    while now() < deadline:
        gen += 1

        # time left fraction
        tleft = max(0.0, deadline - now())
        frac_left = tleft / max(1e-9, float(max_time))

        # occasional local search
        if (gen % 8 == 0) or (gen - last_improve_gen >= 10) or (frac_left < 0.25 and gen % 3 == 0):
            eps = evals_per_sec()
            base = 6 * n + 10
            if eps < 150:
                base = 3 * n + 8
            if eps < 70:
                base = 2 * n + 6
            init_frac = 0.08 if frac_left > 0.4 else 0.05
            fb2, xb2 = local_search(x_best, f_best, budget=base, init_frac=init_frac)
            if fb2 < f_best:
                f_best, x_best = fb2, xb2[:]
                x_template = x_best[:]
                # pull mean toward the new best (in y-space)
                yb = x_to_y(x_best)
                for k in range(n):
                    m[k] = 0.6 * m[k] + 0.4 * yb[k]
                last_improve_gen = gen

        # sample & evaluate population
        pop = []
        for _ in range(lam):
            if now() >= deadline:
                return f_best

            # mixture: mostly Gaussian, sometimes heavy-tailed jump in y
            if random.random() < 0.15:
                z = [cauchy() for _ in range(n)]
                scale = 0.35  # keep cauchy from being too wild
            else:
                z = [random.gauss(0.0, 1.0) for _ in range(n)]
                scale = 1.0

            y = [m[k] + (sigma * D[k]) * (scale * z[k]) for k in range(n)]
            x = y_to_x(y, x_template)
            fx = eval_f(x)

            pop.append((fx, z, y, x))
            if fx < f_best:
                f_best = fx
                x_best = x[:]
                x_template = x_best[:]
                last_improve_gen = gen

        pop.sort(key=lambda t: t[0])

        # recombination in y-space
        old_m = m[:]
        m = [0.0] * n
        for i in range(mu):
            w = weights[i]
            y_i = pop[i][2]
            for k in range(n):
                m[k] += w * y_i[k]

        # y-step in normalized coordinates (approx): (m-old_m)/(sigma*D)
        ystep = [0.0] * n
        for k in range(n):
            denom = max(1e-30, sigma * D[k])
            ystep[k] = (m[k] - old_m[k]) / denom

        # p_sigma
        for k in range(n):
            p_sigma[k] = (1.0 - c_sigma) * p_sigma[k] + math.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * ystep[k]

        # step-size
        norm_ps = math.sqrt(sum(v * v for v in p_sigma))
        sigma *= math.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1.0))
        sigma = clamp(sigma, sigma_min, sigma_max)

        # p_c
        hsig = 1.0 if norm_ps / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * gen)) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
        for k in range(n):
            p_c[k] = (1.0 - c_c) * p_c[k] + hsig * math.sqrt(c_c * (2.0 - c_c) * mueff) * ystep[k]

        # diagonal covariance update in y-space: Ck = Dk^2
        C = [D[k] * D[k] for k in range(n)]

        rank_mu = [0.0] * n
        for i in range(mu):
            w = weights[i]
            z_i = pop[i][1]
            for k in range(n):
                rank_mu[k] += w * (z_i[k] * z_i[k])

        for k in range(n):
            Ck = C[k]
            Ck = (1.0 - c1 - c_mu) * Ck + c1 * (p_c[k] * p_c[k]) + c_mu * Ck * rank_mu[k]
            if Ck < 1e-30:
                Ck = 1e-30
            if Ck > 1e30:
                Ck = 1e30
            C[k] = Ck
            D[k] = math.sqrt(Ck)

        # stagnation handling / restart-like diversification
        if f_best < best_seen - 1e-15:
            best_seen = f_best
        else:
            # if no improvement for a while, increase exploration
            if gen - last_improve_gen >= 18:
                last_improve_gen = gen
                sigma = min(sigma_max, sigma * 1.8)
                # jitter mean around best in y-space
                yb = x_to_y(x_best)
                for k in range(n):
                    m[k] = 0.5 * m[k] + 0.5 * (yb[k] + random.gauss(0.0, 0.6))
                # widen some dimensions
                for k in range(n):
                    D[k] = max(D[k], 0.7)

        # safety: if mean drifts too far, pull back toward best (helps with transforms)
        if gen % 12 == 0:
            yb = x_to_y(x_best)
            for k in range(n):
                m[k] = 0.85 * m[k] + 0.15 * yb[k]

    return f_best
