import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-budgeted box-constrained minimizer.

    Key improvements vs prior submissions:
      - Uses a *diagonal* CMA-ES (very fast, robust) as the main engine:
          * O(dim) sampling/update per candidate (no O(dim^2) covariance).
          * Strong step-size control (CSA) + per-dimension scale adaptation.
      - Adds a *separable coordinate pattern search* refinement around the best,
        triggered opportunistically near the end and after improvements.
      - Uses bound handling via reflection (better than clamp for stochastic steps).
      - Multi-start restarts with increasing population + stagnation-triggered
        "reset sigma" and random injections.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]

    # ---------- helpers ----------
    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    def reflect_bounds_inplace(x):
        # Reflect into [lo, hi] with period 2w; handles steps outside the box smoothly.
        for i in range(dim):
            a, b = lo[i], hi[i]
            if a == b:
                x[i] = a
                continue
            xi = x[i]
            if xi < a or xi > b:
                w = b - a
                y = (xi - a) % (2.0 * w)
                x[i] = (a + y) if (y <= w) else (b - (y - w))
        return x

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    # Box-Muller normal generator
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(random.random(), 1e-300)
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare = z1
        _has_spare = True
        return z0

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm2(a):
        return math.sqrt(dot(a, a))

    # ---------- initial seeding ----------
    best = float("inf")
    best_x = None

    def try_update(x):
        nonlocal best, best_x
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    if now() >= deadline:
        return best

    # Center
    x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    try_update(x0)

    # Limited corners (cheap diversity)
    corner_bits = min(dim, 5)  # up to 32 corners
    max_corners = min(10, 1 << corner_bits)
    for mask in range(max_corners):
        if now() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            if i < corner_bits:
                x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
            else:
                x[i] = 0.5 * (lo[i] + hi[i])
        try_update(x)

    # Random seeds
    seed_count = 10 + 3 * dim
    for _ in range(seed_count):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------- coordinate pattern search refinement ----------
    def coord_refine(x_start, f_start, time_limit):
        x = x_start[:]
        fx = f_start

        # initial step per dimension
        step = [0.15 * span_safe[i] for i in range(dim)]
        min_step = [1e-12 * span_safe[i] for i in range(dim)]

        # a small number of rounds; keep it cheap
        rounds = 2
        for _ in range(rounds):
            improved_any = False
            # random coordinate order helps on some functions
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if now() >= time_limit:
                    return x, fx
                sj = step[j]
                if sj <= min_step[j]:
                    continue

                # try +/- moves
                base = x[j]
                # +sj
                y = x[:]
                y[j] = base + sj
                reflect_bounds_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved_any = True
                    continue
                # -sj
                y = x[:]
                y[j] = base - sj
                reflect_bounds_inplace(y)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    improved_any = True
                    continue

            if not improved_any:
                for j in range(dim):
                    step[j] *= 0.5
            else:
                for j in range(dim):
                    step[j] *= 0.85

        return x, fx

    # ---------- Diagonal CMA-ES with restarts ----------
    avg_span = sum(span_safe) / float(dim)
    sigma0 = 0.25 * avg_span
    sigma_min = 1e-14 * avg_span
    sigma_max = 0.9 * avg_span

    base_lambda = max(8, 4 + int(3 * math.log(dim + 1.0)))
    restart = 0

    # end-game refine scheduling
    last_refine = now()
    refine_period = max(0.15, min(1.0, 0.12 * max_time))

    while now() < deadline:
        # IPOP-like restarts (moderate growth)
        lam = base_lambda * (2 ** (restart // 2))
        lam = min(lam, 80 + 4 * dim)  # keep bounded
        mu = lam // 2

        # log weights
        weights = [0.0] * mu
        for i in range(mu):
            weights[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        wsum = sum(weights)
        for i in range(mu):
            weights[i] /= wsum
        mueff = 1.0 / sum(w * w for w in weights)

        # Diagonal CMA parameters
        # step-size control
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        # diagonal covariance learning rate (aggressive but stable)
        cdiag = min(0.6, (2.0 / (dim + 2.0)) * (0.2 + 0.8 * (mueff / (mueff + 2.0))))

        # E||N(0,I)||
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # initialize mean
        if best_x is not None and random.random() < 0.8:
            m = best_x[:]
            # slight random offset
            for i in range(dim):
                m[i] += (2.0 * random.random() - 1.0) * 0.01 * span_safe[i]
            reflect_bounds_inplace(m)
        else:
            m = rand_point()

        # diagonal "std" per dimension (sqrt of diag(C))
        # start with relative spans to normalize dimensions
        std = [max(1e-12, 0.35 * span_safe[i]) for i in range(dim)]
        # global sigma
        sigma = sigma0 * (0.5 + 1.5 * random.random())
        sigma = max(sigma_min, min(sigma, sigma_max))

        # evolution path for sigma (CSA) in z-space
        ps = [0.0] * dim

        gen = 0
        no_improve = 0
        best_at_start = best

        while now() < deadline:
            gen += 1

            # sample + evaluate population
            pop = []  # (f, idx)
            arz = []  # z vectors
            arx = []  # x vectors

            for k in range(lam):
                if now() >= deadline:
                    return best
                z = [randn() for _ in range(dim)]
                x = [m[i] + sigma * std[i] * z[i] for i in range(dim)]
                reflect_bounds_inplace(x)
                fx = evaluate(x)

                arz.append(z)
                arx.append(x)
                pop.append((fx, k))

                if fx < best:
                    best = fx
                    best_x = x[:]
                    no_improve = 0

            pop.sort(key=lambda t: t[0])

            # recombination for mean in x-space and zmean in z-space
            m_old = m[:]
            m = [0.0] * dim
            zmean = [0.0] * dim
            for i in range(mu):
                _, idx = pop[i]
                wi = weights[i]
                xi = arx[idx]
                zi = arz[idx]
                for d in range(dim):
                    m[d] += wi * xi[d]
                    zmean[d] += wi * zi[d]
            reflect_bounds_inplace(m)

            # update ps (CSA) in z-space
            c = math.sqrt(cs * (2.0 - cs) * mueff)
            for d in range(dim):
                ps[d] = (1.0 - cs) * ps[d] + c * zmean[d]
            ps_norm = norm2(ps)

            # update sigma
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            # update diagonal std via weighted variance of selected steps (in z-space)
            # target: std^2 <- (1-c)*std^2 + c * sum(wi * (std*zi)^2) / ??? (diag only)
            # Use zi^2 as proxy to adapt axis scaling; keep strictly positive.
            for d in range(dim):
                s2 = 0.0
                for i in range(mu):
                    _, idx = pop[i]
                    wi = weights[i]
                    zd = arz[idx][d]
                    s2 += wi * (zd * zd)
                # s2 ~ 1 when distribution matches; adapt multiplicatively around 1
                # keep adaptation gentle to avoid premature shrink/expand
                factor = math.exp(0.5 * cdiag * (s2 - 1.0))
                std[d] = max(1e-15 * span_safe[d], min(std[d] * factor, 2.0 * span_safe[d]))

            # stagnation bookkeeping
            if best >= best_at_start - 1e-15 * (1.0 + abs(best_at_start)):
                no_improve += 1
            else:
                best_at_start = best
                no_improve = 0

            # opportunistic refinement
            if best_x is not None and (now() - last_refine) >= refine_period:
                last_refine = now()
                remain = deadline - now()
                tl = now() + min(0.06 * max_time, 0.25 * remain, 0.5)
                xr, fr = coord_refine(best_x, best, tl)
                if fr < best:
                    best, best_x = fr, xr[:]
                    # pull mean toward refined solution
                    m = xr[:]

            # restart triggers
            if sigma <= sigma_min * 1.05:
                break
            if no_improve > (12 + 2 * dim):
                # partial reset: keep mean but re-expand sigma a bit + jitter
                if best_x is not None and random.random() < 0.7:
                    m = best_x[:]
                for d in range(dim):
                    m[d] += 0.02 * span_safe[d] * randn()
                reflect_bounds_inplace(m)
                sigma = min(sigma_max, max(sigma, 0.15 * sigma0))
                no_improve = 0
                # if still stuck for long, restart fully
                if gen > (25 + dim):
                    break

        restart += 1

    return best
