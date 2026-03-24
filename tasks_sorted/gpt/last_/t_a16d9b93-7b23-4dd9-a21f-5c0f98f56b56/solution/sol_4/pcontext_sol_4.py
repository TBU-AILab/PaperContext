import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over previous version:
      - Switch to a compact CMA-ES–style optimizer (diagonal covariance) with:
          * cumulative step-size adaptation (CSA)
          * rank-1 + rank-mu covariance update (diagonal only)
      - Mirrored sampling (antithetic pairs) to cut noise and improve use of evaluations
      - Forced global injections + adaptive restarts on stagnation
      - Lightweight local coordinate search around incumbent late in the run
      - Evaluation cache with quantization to reduce redundant calls

    Returns:
      best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def clamp(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if v < lo: v = lo
            if v > hi: v = hi
            y[i] = v
        return y

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Box-Muller gaussian
    _has_spare = False
    _spare = 0.0
    def gauss():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare = z1
        _has_spare = True
        return z0

    def spans():
        s = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            s[i] = (hi - lo) if hi > lo else 1.0
        return s

    span = spans()

    # quantized cache
    cache = {}
    def key_of(x):
        k = []
        for i in range(dim):
            lo, hi = bounds[i]
            sp = hi - lo
            if sp <= 0:
                k.append(0)
            else:
                # 2e4 buckets across range (reasonably fine, helps duplicates)
                q = int((x[i] - lo) / sp * 20000.0)
                k.append(q)
        return tuple(k)

    def evaluate(x):
        x = clamp(x)
        k = key_of(x)
        if k in cache:
            return cache[k], x
        try:
            v = float(func(x))
            if not is_finite(v):
                v = float("inf")
        except Exception:
            v = float("inf")
        cache[k] = v
        return v, x

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    # ---------- initialization ----------
    # Population sizes (CMA-ES defaults)
    lam = int(4 + math.floor(3 * math.log(max(2, dim))))
    lam = max(10, min(64, lam))
    mu = lam // 2

    # log weights
    weights = [0.0] * mu
    for i in range(mu):
        weights[i] = math.log(mu + 0.5) - math.log(i + 1.0)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # Strategy parameters (diagonal CMA-ish)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    # Expected norm of N(0,I)
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # Start from best of a small random batch
    x_best = None
    f_best = float("inf")
    warm = max(lam, 2 * mu)
    mean = rand_vec()
    for _ in range(warm):
        if time.time() >= deadline:
            return f_best
        x = rand_vec()
        fx, x = evaluate(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]
            mean = x[:]

    # global step-size and diagonal covariance (D are stddev multipliers)
    # Start moderate (1/3 of span on average)
    avg_span = sum(span) / max(1, dim)
    sigma = 0.25 * (avg_span if avg_span > 0 else 1.0)
    D = [1.0] * dim

    # evolution paths
    pc = [0.0] * dim
    ps = [0.0] * dim

    last_improve = time.time()
    best_hist = [f_best]

    # ---------- local search ----------
    def local_coordinate_search(x0, f0, steps=2):
        x = x0[:]
        fx = f0
        # step size relative to current sigma and spans
        for _ in range(steps):
            improved = False
            for d in range(dim):
                if time.time() >= deadline:
                    return fx, x
                lo, hi = bounds[d]
                sp = span[d]
                if sp <= 0:
                    continue
                h = max(1e-12 * sp, min(0.15 * sp, 0.75 * sigma * D[d]))
                if h <= 0:
                    continue

                # try +/- h
                xm = x[:]; xm[d] -= h
                xp = x[:]; xp[d] += h
                fm, xm = evaluate(xm)
                fp, xp = evaluate(xp)

                if fm < fx and fm <= fp:
                    x, fx = xm, fm
                    improved = True
                elif fp < fx:
                    x, fx = xp, fp
                    improved = True

                # small shrink if no improvement in this coord
            if not improved:
                break
        return fx, x

    # ---------- main loop ----------
    gen = 0
    while time.time() < deadline:
        gen += 1

        # Adaptive restart on long stagnation
        if (time.time() - last_improve) > 0.45 * max_time:
            # restart around best (or global if none), inflate sigma, reset paths
            mean = x_best[:] if x_best is not None and random.random() < 0.85 else rand_vec()
            sigma = max(sigma, 0.35 * avg_span)
            D = [1.0] * dim
            pc = [0.0] * dim
            ps = [0.0] * dim
            last_improve = time.time()

        # create offspring (use mirrored pairs)
        arz = []   # z ~ N(0, I)
        arx = []   # x = mean + sigma * D * z
        fitness = []

        # small chance of global injection to avoid premature convergence
        inject = 1 if random.random() < 0.15 else 0

        k = 0
        while k < lam and time.time() < deadline:
            if inject and k == 0:
                x = rand_vec()
                fx, x = evaluate(x)
                z = [0.0] * dim
                arz.append(z); arx.append(x); fitness.append(fx)
                k += 1
                continue

            # mirrored sampling
            z = [gauss() for _ in range(dim)]
            x = [mean[i] + sigma * D[i] * z[i] for i in range(dim)]
            fx, x = evaluate(x)
            arz.append(z); arx.append(x); fitness.append(fx)
            k += 1
            if k >= lam or time.time() >= deadline:
                break

            z2 = [-zi for zi in z]
            x2 = [mean[i] + sigma * D[i] * z2[i] for i in range(dim)]
            fx2, x2 = evaluate(x2)
            arz.append(z2); arx.append(x2); fitness.append(fx2)
            k += 1

        if not fitness:
            break

        # sort by fitness
        idx = list(range(len(fitness)))
        idx.sort(key=lambda i: fitness[i])

        # update incumbent best
        if fitness[idx[0]] < f_best:
            f_best = fitness[idx[0]]
            x_best = arx[idx[0]][:]
            last_improve = time.time()

        # selection
        sel = idx[:mu]

        old_mean = mean[:]
        # new mean (in x-space)
        mean = [0.0] * dim
        for j, ii in enumerate(sel):
            w = weights[j]
            xi = arx[ii]
            for d in range(dim):
                mean[d] += w * xi[d]
        mean = clamp(mean)

        # compute y = (mean - old_mean) / (sigma * D)
        y = [0.0] * dim
        for d in range(dim):
            denom = sigma * D[d]
            y[d] = (mean[d] - old_mean[d]) / denom if denom > 0 else 0.0

        # update ps (CSA) using diagonal approximation => use y directly
        for d in range(dim):
            ps[d] = (1.0 - cs) * ps[d] + math.sqrt(cs * (2.0 - cs) * mueff) * y[d]

        # compute hsig
        ps_norm = norm(ps)
        # prevents premature hsig false
        thresh = (1.4 + 2.0 / (dim + 1.0)) * chiN
        hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen))) < thresh else 0.0

        # update pc
        for d in range(dim):
            pc[d] = (1.0 - cc) * pc[d] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (mean[d] - old_mean[d]) / (sigma if sigma > 0 else 1.0)

        # update diagonal covariance D^2 (store in D as sqrt(diag(C)))
        # rank-one + rank-mu updates on diagonal only
        # accumulate weighted z^2 of selected individuals
        z2w = [0.0] * dim
        for j, ii in enumerate(sel):
            w = weights[j]
            zi = arz[ii]
            for d in range(dim):
                z2w[d] += w * (zi[d] * zi[d])

        for d in range(dim):
            # current variance component
            cvar = D[d] * D[d]
            # rank-one uses pc in x-space: convert to normalized by sigma and D? diagonal approx:
            # use (pc/(sigma))^2 as contribution; scale by 1/D^2 to keep consistent -> approximate with (pc/(sigma))^2
            pc_term = (pc[d] / (sigma if sigma > 0 else 1.0))
            pc2 = pc_term * pc_term

            cvar = (1.0 - c1 - cmu) * cvar + c1 * (pc2 + (1.0 - hsig) * cc * (2.0 - cc) * cvar) + cmu * z2w[d] * cvar
            # numeric safety + convert back to D
            if not is_finite(cvar) or cvar <= 1e-300:
                cvar = 1e-300
            D[d] = math.sqrt(cvar)

        # update sigma
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))

        # keep sigma within sensible bounds wrt search box
        sigma = max(1e-18 * avg_span, min(sigma, 2.0 * avg_span + 1e-12))

        # occasional local improvement near end / when converged
        elapsed = (time.time() - t0) / max(1e-12, max_time)
        if x_best is not None and (gen % 9 == 0) and elapsed > 0.55 and (time.time() + 0.005 < deadline):
            f_loc, x_loc = local_coordinate_search(x_best, f_best, steps=2)
            if f_loc < f_best:
                f_best, x_best = f_loc, x_loc[:]
                mean = x_best[:]
                last_improve = time.time()

        # track stagnation trend (tiny, cheap)
        best_hist.append(f_best)
        if len(best_hist) > 25:
            best_hist.pop(0)
            # if absolutely flat, nudge sigma upward a bit
            if abs(best_hist[0] - best_hist[-1]) <= 1e-14 * (1.0 + abs(best_hist[-1])):
                sigma *= 1.15

    return f_best
