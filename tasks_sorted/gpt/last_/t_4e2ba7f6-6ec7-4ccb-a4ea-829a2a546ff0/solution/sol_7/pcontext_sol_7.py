import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Main improvements vs provided code:
      - Uses CMA-ES style adaptation (diagonal covariance) with mirrored sampling:
        very strong on continuous bounded problems under tight time.
      - Interleaves a light local search (pattern/coordinate) around the incumbent.
      - Robust bound handling via reflection; optional occasional uniform injections.
      - Evaluation cache to avoid repeats; strict time checks.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    if any(s <= 0.0 for s in spans):
        x = [lows[i] for i in range(dim)]
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # ---------- helpers ----------
    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        # reflect until in range (handles big jumps)
        while v < lo or v > hi:
            if v < lo:
                v = lo + (lo - v)
            else:
                v = hi - (v - hi)
        return clamp(v, lo, hi)

    def ensure_bounds(x):
        return [clamp(x[i], lows[i], highs[i]) for i in range(dim)]

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- evaluation + cache ----------
    best = float("inf")
    best_x = None

    cache = {}
    cache_decimals = 12

    def key_of(x):
        return tuple(round(v, cache_decimals) for v in x)

    def eval_f(x):
        nonlocal best, best_x
        k = key_of(x)
        if k in cache:
            fx = cache[k]
        else:
            fx = float(func(x))
            cache[k] = fx
        if fx < best:
            best = fx
            best_x = x[:]
        return fx

    # ---------- quick local search (time-safe) ----------
    def local_pattern_search(x0, f0, max_sweeps=2, step0_frac=0.08):
        if x0 is None:
            return x0, f0
        x = x0[:]
        fx = f0

        steps = [max(1e-15, step0_frac * spans[i]) for i in range(dim)]
        min_steps = [1e-12 * spans[i] + 1e-15 for i in range(dim)]

        for _ in range(max_sweeps):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)

            for j in order:
                if time.time() >= deadline:
                    break
                sj = steps[j]
                if sj <= min_steps[j]:
                    continue

                xj = x[j]

                xp = x[:]
                xp[j] = reflect(xj + sj, lows[j], highs[j])
                fp = eval_f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    steps[j] = min(steps[j] * 1.3, spans[j])
                    continue

                xm = x[:]
                xm[j] = reflect(xj - sj, lows[j], highs[j])
                fm = eval_f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True
                    steps[j] = min(steps[j] * 1.3, spans[j])
                    continue

                steps[j] = sj * 0.6

            if not improved:
                small = True
                for j in range(dim):
                    steps[j] *= 0.7
                    if steps[j] > min_steps[j]:
                        small = False
                if small:
                    break
        return x, fx

    # ---------- initialization ----------
    # start from several random points; take best
    init_tries = max(6, min(30, 2 * dim + 6))
    for _ in range(init_tries):
        if time.time() >= deadline:
            return best
        x = rand_uniform_vec()
        eval_f(x)

    # if we have time, do one tiny local improvement from the best
    if best_x is not None and time.time() < deadline:
        bx, bf = local_pattern_search(best_x, best, max_sweeps=1, step0_frac=0.12)
        best_x, best = bx, bf

    # ---------- Diagonal CMA-ES core ----------
    # population sizes (standard-ish)
    lam = int(4 + 3 * math.log(dim + 1.0))
    lam = max(10, min(60, lam))
    mu = lam // 2

    # recombination weights
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # strategy parameters (diag version)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    # init mean at best found
    m = best_x[:] if best_x is not None else rand_uniform_vec()

    # initial step-size relative to span
    sigma = 0.25
    # diagonal "covariance" factors
    D = [1.0] * dim

    # evolution paths
    pc = [0.0] * dim
    ps = [0.0] * dim

    # expected norm of N(0, I)
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # stagnation tracking
    last_best = best
    stagn = 0

    # utilities
    def randn():
        # Box-Muller (uses only random.random)
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-16)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        return r * math.cos(theta)

    # time-safe loop
    gen = 0
    while time.time() < deadline:
        gen += 1

        # sample population with mirrored sampling to reduce noise
        zs = []
        xs = []
        fs = []

        half = lam // 2
        for k in range(half):
            if time.time() >= deadline:
                return best

            z = [randn() for _ in range(dim)]
            # candidate
            x = [0.0] * dim
            for i in range(dim):
                x[i] = reflect(m[i] + (sigma * D[i]) * z[i], lows[i], highs[i])
            fx = eval_f(x)

            # mirrored candidate
            z2 = [-zi for zi in z]
            x2 = [0.0] * dim
            for i in range(dim):
                x2[i] = reflect(m[i] + (sigma * D[i]) * z2[i], lows[i], highs[i])
            fx2 = eval_f(x2) if time.time() < deadline else float("inf")

            zs.append(z);   xs.append(x);   fs.append(fx)
            zs.append(z2);  xs.append(x2);  fs.append(fx2)

        # if lam odd, add one more
        if len(fs) < lam and time.time() < deadline:
            z = [randn() for _ in range(dim)]
            x = [reflect(m[i] + (sigma * D[i]) * z[i], lows[i], highs[i]) for i in range(dim)]
            fx = eval_f(x)
            zs.append(z); xs.append(x); fs.append(fx)

        # occasional diversity injection (helps under tricky bounds)
        if random.random() < 0.08 and time.time() < deadline:
            x = rand_uniform_vec()
            eval_f(x)

        # sort by fitness
        idx = list(range(len(fs)))
        idx.sort(key=lambda i: fs[i])

        # recombination: new mean
        old_m = m[:]
        m = [0.0] * dim
        zmean = [0.0] * dim
        for r in range(mu):
            i = idx[r]
            wi = w[r]
            xi = xs[i]
            zi = zs[i]
            for j in range(dim):
                m[j] += wi * xi[j]
                zmean[j] += wi * zi[j]

        # update evolution path ps (diag invsqrt(C) == 1/D)
        # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * ( (m-old_m) / (sigma*D) )
        coeff_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        invsig = 1.0 / max(1e-15, sigma)
        for j in range(dim):
            y = (m[j] - old_m[j]) * invsig / max(1e-15, D[j])
            ps[j] = (1.0 - cs) * ps[j] + coeff_ps * y

        # step-size control
        ps_norm = math.sqrt(sum(v * v for v in ps))
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        # keep sigma sane relative to bounds
        sigma = max(1e-12, min(1.0, sigma))

        # hsig for pc update
        hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen))) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0

        # update evolution path pc
        coeff_pc = math.sqrt(cc * (2.0 - cc) * mueff) * hsig
        for j in range(dim):
            y = (m[j] - old_m[j]) * invsig
            pc[j] = (1.0 - cc) * pc[j] + coeff_pc * y

        # update diagonal covariance factors D (via diag C)
        # Cdiag <- (1-c1-cmu)*Cdiag + c1*pc^2 + cmu*sum(wi*(yi^2))
        # where yi = (xi-old_m)/sigma  (using selected mu)
        Cdiag = [D[j] * D[j] for j in range(dim)]
        # rank-mu component
        y2 = [0.0] * dim
        for r in range(mu):
            i = idx[r]
            wi = w[r]
            xi = xs[i]
            for j in range(dim):
                y = (xi[j] - old_m[j]) * invsig
                y2[j] += wi * (y * y)

        for j in range(dim):
            Cdiag[j] = (1.0 - c1 - cmu) * Cdiag[j] + c1 * (pc[j] * pc[j]) + cmu * y2[j]
            if Cdiag[j] < 1e-30:
                Cdiag[j] = 1e-30
            D[j] = math.sqrt(Cdiag[j])

        # opportunistic local search every few generations or when stagnating
        if best < last_best - 1e-12 * (abs(last_best) + 1.0):
            last_best = best
            stagn = 0
        else:
            stagn += 1

        if (gen % 6 == 0 or stagn >= 10) and best_x is not None and time.time() < deadline:
            stagn = 0
            step0 = 0.10 if sigma > 0.08 else 0.05
            bx, bf = local_pattern_search(best_x, best, max_sweeps=2, step0_frac=step0)
            best_x, best = bx, bf
            # pull mean toward improved best (keeps CMA focused)
            for j in range(dim):
                m[j] = 0.7 * m[j] + 0.3 * best_x[j]

        # soft restart if sigma collapses too much
        if sigma < 1e-6 and time.time() < deadline:
            sigma = 0.20
            # re-center near best, add slight randomization
            if best_x is not None:
                m = [reflect(best_x[j] + 0.02 * spans[j] * randn(), lows[j], highs[j]) for j in range(dim)]
            else:
                m = rand_uniform_vec()
            pc = [0.0] * dim
            ps = [0.0] * dim
            D = [1.0] * dim

    return best
