import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-budgeted box-constrained minimizer (self-contained, no numpy).

    Main changes vs your best (CMA-ES full covariance):
      1) Switch to a *fast diagonal* CMA-ES core (O(dim) per sample) so it can
         evaluate many more candidates in the same time.
      2) Add an explicit *trust-region local search* around the best solution:
         - coordinate steps + short random-subspace line searches
         - step-size shrinks on failure, expands on success
      3) Better use of evaluations via a tiny "elite pool" and
         periodic intensification from elites.
      4) Robust bound handling via reflection.

    Returns:
        best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0.0 else 1.0 for s in span]
    avg_span = sum(span_safe) / float(dim) if dim > 0 else 1.0

    # ---------------- helpers ----------------
    def now():
        return time.time()

    def reflect_inplace(x):
        # reflect each coordinate into [lo,hi] with period 2w
        for i in range(dim):
            a = lo[i]
            b = hi[i]
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

    def evaluate(x):
        return float(func(x))

    # Box-Muller normal
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

    def norm2(v):
        s = 0.0
        for i in range(dim):
            s += v[i] * v[i]
        return math.sqrt(s)

    # ---------------- best bookkeeping ----------------
    best = float("inf")
    best_x = None

    # small elite pool (kept sorted)
    elite_k = 6
    elite = []  # list of (f, x)

    def elite_add(f, x):
        nonlocal elite
        elite.append((f, x[:]))
        elite.sort(key=lambda t: t[0])
        if len(elite) > elite_k:
            elite = elite[:elite_k]

    def try_update(x):
        nonlocal best, best_x
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x[:]
        elite_add(f, x)
        return f

    if now() >= deadline:
        return best

    # ---------------- seeding ----------------
    # center
    x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
    reflect_inplace(x0)
    try_update(x0)

    # a few corners (limited)
    corner_bits = min(dim, 5)  # up to 32 corners
    max_corners = min(12, 1 << corner_bits)
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

    # random seeds
    seed_count = 12 + 4 * dim
    for _ in range(seed_count):
        if now() >= deadline:
            return best
        try_update(rand_point())

    # ---------------- local trust-region search ----------------
    def local_trust_region(x_start, f_start, time_limit):
        """
        Cheap but strong local improvement:
          - coordinate moves with adaptive step
          - occasional random-subspace directional line search (few probes)
        """
        x = x_start[:]
        fx = f_start

        # initial step sizes
        step = [0.10 * span_safe[i] for i in range(dim)]
        min_step = [1e-14 * span_safe[i] for i in range(dim)]
        max_step = [0.50 * span_safe[i] for i in range(dim)]

        # number of "sweeps" kept small
        sweeps = 2
        for _ in range(sweeps):
            if now() >= time_limit:
                break

            improved = False

            # coordinate exploration in random order
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if now() >= time_limit:
                    return x, fx
                sj = step[j]
                if sj <= min_step[j]:
                    continue

                base = x[j]

                # try +/- sj
                for direction in (1.0, -1.0):
                    y = x[:]
                    y[j] = base + direction * sj
                    reflect_inplace(y)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break

            # random-subspace direction line search (helps on rotated valleys)
            if now() < time_limit:
                # Build a sparse random direction affecting ~sqrt(dim) coords
                k = max(1, int(math.sqrt(dim)))
                idxs = random.sample(range(dim), k) if dim >= k else list(range(dim))
                d = [0.0] * dim
                for j in idxs:
                    d[j] = randn()
                dn = norm2(d)
                if dn > 0:
                    inv = 1.0 / dn
                    for j in idxs:
                        d[j] *= inv

                    # probe a few alphas based on average step
                    base_step = 0.0
                    for j in idxs:
                        base_step += step[j]
                    base_step = base_step / float(len(idxs)) if idxs else 0.0
                    base_step = max(base_step, 1e-16 * avg_span)

                    # 3-point "line": -a, +a, +2a
                    for alpha in (-base_step, base_step, 2.0 * base_step):
                        if now() >= time_limit:
                            break
                        y = x[:]
                        for j in idxs:
                            y[j] += alpha * d[j]
                        reflect_inplace(y)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved = True

            # adapt step sizes
            if improved:
                for j in range(dim):
                    step[j] = min(max_step[j], step[j] * 1.15)
            else:
                for j in range(dim):
                    step[j] *= 0.5

        return x, fx

    # ---------------- Diagonal CMA-ES core with restarts ----------------
    # base parameters
    base_lambda = max(10, 4 + int(3 * math.log(dim + 1.0)))
    sigma0 = 0.30 * avg_span
    sigma_min = 1e-16 * avg_span
    sigma_max = 0.90 * avg_span

    restart = 0
    last_intensify = now()
    intensify_period = max(0.15, min(1.0, 0.10 * float(max_time)))

    while now() < deadline:
        # IPOP-ish restart schedule (but bounded)
        lam = base_lambda * (2 ** (restart // 2))
        lam = min(lam, 120)  # cap for time-limited setting
        mu = max(2, lam // 2)

        # log weights
        weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        mueff = 1.0 / sum(w * w for w in weights)

        # CSA parameters
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        # diag adaptation rate (mild, stable)
        cdiag = min(0.5, 2.0 / (dim + 2.0))

        # initialize mean from elite/best
        if best_x is not None and random.random() < 0.85:
            m = best_x[:]
        elif elite and random.random() < 0.85:
            m = elite[random.randrange(len(elite))][1][:]
        else:
            m = rand_point()

        # add small jitter to mean
        for i in range(dim):
            m[i] += (2.0 * random.random() - 1.0) * 0.01 * span_safe[i]
        reflect_inplace(m)

        # per-dimension std (sqrt(diag(C))) initialized to box scale
        std = [max(1e-15 * span_safe[i], 0.35 * span_safe[i]) for i in range(dim)]

        sigma = sigma0 * (0.5 + 1.5 * random.random())
        sigma = max(sigma_min, min(sigma, sigma_max))

        ps = [0.0] * dim

        no_improve = 0
        best_anchor = best

        while now() < deadline:
            # population sampling
            arz = []
            arx = []
            pop = []

            for k in range(lam):
                if now() >= deadline:
                    return best
                z = [randn() for _ in range(dim)]
                x = [m[i] + sigma * std[i] * z[i] for i in range(dim)]
                reflect_inplace(x)
                fx = evaluate(x)

                arz.append(z)
                arx.append(x)
                pop.append((fx, k))

                if fx < best:
                    best = fx
                    best_x = x[:]
                    elite_add(fx, x)
                    no_improve = 0

            pop.sort(key=lambda t: t[0])

            # recombination
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
            reflect_inplace(m)

            # CSA path
            c = math.sqrt(cs * (2.0 - cs) * mueff)
            for d in range(dim):
                ps[d] = (1.0 - cs) * ps[d] + c * zmean[d]
            psn = norm2(ps)

            sigma *= math.exp((cs / damps) * (psn / chiN - 1.0))
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            # diagonal std update from selected individuals (use z^2 statistics)
            for d in range(dim):
                s2 = 0.0
                for i in range(mu):
                    _, idx = pop[i]
                    s2 += weights[i] * (arz[idx][d] * arz[idx][d])
                # move std so that E[z^2]=1; multiplicative gentle update
                std[d] *= math.exp(0.5 * cdiag * (s2 - 1.0))
                # keep in sane range wrt box
                if std[d] < 1e-15 * span_safe[d]:
                    std[d] = 1e-15 * span_safe[d]
                elif std[d] > 2.0 * span_safe[d]:
                    std[d] = 2.0 * span_safe[d]

            # stagnation tracking
            if best >= best_anchor - 1e-15 * (1.0 + abs(best_anchor)):
                no_improve += 1
            else:
                best_anchor = best
                no_improve = 0

            # periodic intensification: local trust-region on current best
            if best_x is not None and (now() - last_intensify) >= intensify_period:
                last_intensify = now()
                remain = deadline - now()
                tl = now() + min(0.08 * float(max_time), 0.25 * remain, 0.6)
                xr, fr = local_trust_region(best_x, best, tl)
                if fr < best:
                    best, best_x = fr, xr[:]
                    elite_add(fr, xr)
                    # pull mean to refined solution (exploitation)
                    m = xr[:]

            # restart criteria
            if sigma <= sigma_min * 1.2:
                break
            if no_improve > (8 + 2 * dim):
                # partial reset around an elite/best point with expanded sigma
                if elite:
                    m = elite[random.randrange(len(elite))][1][:]
                elif best_x is not None:
                    m = best_x[:]
                for d in range(dim):
                    m[d] += 0.03 * span_safe[d] * randn()
                reflect_inplace(m)
                sigma = min(sigma_max, max(sigma, 0.20 * sigma0))
                no_improve = 0
                # if still stuck, end this restart
                if random.random() < 0.35:
                    break

        restart += 1

    return best
