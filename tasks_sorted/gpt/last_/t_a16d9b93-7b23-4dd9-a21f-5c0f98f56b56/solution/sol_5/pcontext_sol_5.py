import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Key improvements vs previous code:
      1) Proper *diagonal CMA-ES* update in normalized coordinates (z-space):
         - covariance update uses selected z-vectors (rank-mu) + evolution path (rank-1)
         - avoids the inconsistent "multiply by current variance" issue that can freeze/warp D
      2) Robust box handling via *reflection* (better than clamp for search dynamics)
      3) Two-phase schedule:
         - early: larger sigma + more global injections
         - late: reduced injections + stronger local search (pattern search)
      4) Restarts on stagnation with increasing exploration
      5) Light evaluation cache (quantized) kept, but reflection reduces duplicates anyway

    Returns:
      best fitness found (float)
    """

    t0 = time.time()
    deadline = t0 + max_time

    # ---------------- helpers ----------------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    span = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    # reflect a point into bounds (keeps continuity better than clamp)
    def reflect_into_box(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if hi <= lo:
                y[i] = lo
                continue
            w = hi - lo
            # map to [0, 2w) then reflect to [0, w]
            t = (v - lo) % (2.0 * w)
            if t < 0.0:
                t += 2.0 * w
            if t > w:
                t = 2.0 * w - t
            y[i] = lo + t
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

    def dot(a, b):
        s = 0.0
        for i in range(dim):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

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
                # buckets across range
                q = int((x[i] - lo) / sp * 30000.0)
                k.append(q)
        return tuple(k)

    def evaluate(x):
        x = reflect_into_box(x)
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

    # ---------------- initialization ----------------
    # CMA-ES population defaults (bounded for time)
    lam = int(4 + math.floor(3 * math.log(max(2, dim))))
    lam = max(12, min(64, lam))
    mu = lam // 2

    # recombination weights
    weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)

    # diagonal CMA parameters
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    avg_span = sum(span) / max(1, dim)

    # warm start: sample a batch, pick best as initial mean
    x_best = None
    f_best = float("inf")
    mean = rand_vec()
    warm = max(lam * 2, 24)
    for _ in range(warm):
        if time.time() >= deadline:
            return f_best
        x = rand_vec()
        fx, x = evaluate(x)
        if fx < f_best:
            f_best, x_best = fx, x[:]
            mean = x[:]

    # global step size
    sigma = 0.35 * (avg_span if avg_span > 0 else 1.0)

    # diagonal stds in normalized sampling space
    # (D multiplies z; covariance diag is D^2)
    D = [1.0] * dim

    # evolution paths
    pc = [0.0] * dim
    ps = [0.0] * dim

    # stagnation / restarts
    last_improve_t = time.time()
    restart_count = 0

    # ---------------- local search (pattern search) ----------------
    def pattern_search(x0, f0, budget_evals=30):
        x = x0[:]
        fx = f0
        # initial step per-dimension relative to span and sigma
        step = [min(0.2 * span[i], max(1e-12 * span[i], 0.6 * sigma * D[i])) for i in range(dim)]
        evals = 0

        while evals < budget_evals and time.time() < deadline:
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline or evals >= budget_evals:
                    break
                if step[i] <= 1e-15 * span[i]:
                    continue

                xp = x[:]; xp[i] += step[i]
                xm = x[:]; xm[i] -= step[i]

                fp, xp = evaluate(xp); evals += 1
                fm, xm = evaluate(xm); evals += 1

                if fp < fx or fm < fx:
                    if fp <= fm:
                        x, fx = xp, fp
                    else:
                        x, fx = xm, fm
                    improved = True

            if not improved:
                # shrink steps
                for i in range(dim):
                    step[i] *= 0.5
                # stop if all tiny
                if max(step) <= 1e-14 * (avg_span + 1.0):
                    break
        return fx, x

    # ---------------- main loop ----------------
    gen = 0
    while time.time() < deadline:
        gen += 1
        elapsed = (time.time() - t0) / max(1e-12, max_time)

        # restart on stagnation
        # (earlier restarts if no improvement; more exploration with each restart)
        stagn_lim = (0.18 + 0.06 * restart_count) * max_time
        if (time.time() - last_improve_t) > stagn_lim:
            restart_count += 1
            # with some probability restart around best; otherwise global
            if x_best is not None and random.random() < 0.75:
                mean = x_best[:]
            else:
                mean = rand_vec()
            # inflate sigma and reset paths; keep D mild
            sigma = max(sigma, (0.45 + 0.10 * min(3, restart_count)) * avg_span)
            D = [1.0] * dim
            pc = [0.0] * dim
            ps = [0.0] * dim
            last_improve_t = time.time()

        # phase-dependent injection probability
        # more global early; less later
        p_inject = 0.22 if elapsed < 0.35 else (0.10 if elapsed < 0.75 else 0.05)
        # slightly increase injections on restarts
        p_inject = min(0.35, p_inject + 0.04 * min(3, restart_count))

        # sample offspring (mirrored)
        arz = []      # z vectors
        arx = []      # candidate x
        fit = []

        k = 0
        while k < lam and time.time() < deadline:
            if random.random() < p_inject:
                x = rand_vec()
                fx, x = evaluate(x)
                arz.append([0.0] * dim)
                arx.append(x)
                fit.append(fx)
                k += 1
                continue

            z = [gauss() for _ in range(dim)]
            x = [mean[i] + sigma * D[i] * z[i] for i in range(dim)]
            fx, x = evaluate(x)
            arz.append(z); arx.append(x); fit.append(fx)
            k += 1
            if k >= lam or time.time() >= deadline:
                break

            z2 = [-v for v in z]
            x2 = [mean[i] + sigma * D[i] * z2[i] for i in range(dim)]
            fx2, x2 = evaluate(x2)
            arz.append(z2); arx.append(x2); fit.append(fx2)
            k += 1

        if not fit:
            break

        # sort
        idx = list(range(len(fit)))
        idx.sort(key=lambda i: fit[i])

        # update best
        if fit[idx[0]] < f_best:
            f_best = fit[idx[0]]
            x_best = arx[idx[0]][:]
            last_improve_t = time.time()

        # recombination: compute weighted mean in x-space
        old_mean = mean[:]
        mean = [0.0] * dim
        for j in range(mu):
            ii = idx[j]
            w = weights[j]
            xi = arx[ii]
            for d in range(dim):
                mean[d] += w * xi[d]
        mean = reflect_into_box(mean)

        # compute y = (mean - old_mean) / sigma  (x-space normalized by sigma)
        inv_sigma = 1.0 / sigma if sigma > 0 else 0.0
        y = [(mean[d] - old_mean[d]) * inv_sigma for d in range(dim)]

        # update ps in z-space: use y / D  (because y = D * z_mean approximately)
        for d in range(dim):
            yd = y[d] / (D[d] if D[d] > 0 else 1.0)
            ps[d] = (1.0 - cs) * ps[d] + math.sqrt(cs * (2.0 - cs) * mueff) * yd

        ps_norm = norm(ps)
        thresh = (1.4 + 2.0 / (dim + 1.0)) * chiN
        denom = math.sqrt(max(1e-300, 1.0 - (1.0 - cs) ** (2.0 * gen)))
        hsig = 1.0 if (ps_norm / denom) < thresh else 0.0

        # update pc in x-space
        for d in range(dim):
            pc[d] = (1.0 - cc) * pc[d] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y[d]

        # ---- diagonal covariance update (properly in z-space) ----
        # rank-mu term: E[z_i^2] over selected
        z2w = [0.0] * dim
        for j in range(mu):
            ii = idx[j]
            w = weights[j]
            zi = arz[ii]
            for d in range(dim):
                z2w[d] += w * (zi[d] * zi[d])

        # rank-1 term in z-space: (pc/(sigma*D))^2
        for d in range(dim):
            if D[d] <= 0.0:
                D[d] = 1.0
            pcz = (pc[d] * inv_sigma) / D[d]  # convert pc to z-coordinates
            pcz2 = pcz * pcz

            # update variance in z-space: D^2
            D2 = D[d] * D[d]
            # (1-hsig) correction approximated for diagonal case
            D2 = (1.0 - c1 - cmu) * D2 + c1 * (pcz2 + (1.0 - hsig) * cc * (2.0 - cc) * D2) + cmu * z2w[d]
            if not is_finite(D2) or D2 < 1e-300:
                D2 = 1e-300
            D[d] = math.sqrt(D2)

        # sigma update (CSA)
        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))

        # keep sigma reasonable
        sigma = max(1e-18 * (avg_span + 1.0), min(sigma, 3.0 * (avg_span + 1e-12)))

        # late-stage local refinement around best
        if x_best is not None and elapsed > 0.60 and (gen % 6 == 0) and (time.time() + 0.01 < deadline):
            # budget grows late and with restarts
            bud = 18 if elapsed < 0.85 else 36
            bud += 6 * min(3, restart_count)
            f_loc, x_loc = pattern_search(x_best, f_best, budget_evals=bud)
            if f_loc < f_best:
                f_best, x_best = f_loc, x_loc[:]
                mean = x_best[:]
                last_improve_t = time.time()

    return f_best
