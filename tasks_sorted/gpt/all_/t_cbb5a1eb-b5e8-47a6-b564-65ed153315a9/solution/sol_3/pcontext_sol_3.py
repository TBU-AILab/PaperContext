import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded minimizer (no external libs).

    Key improvements vs your last set:
      - Uses a *small CMA-ES-like* search distribution: mean + diagonal covariance (per-dimension stds)
      - Uses *weighted recombination* from multiple top samples (not only best)
      - Uses *evolution path* (momentum) to accelerate progress in valleys
      - Uses *mirrored bounds* (better than clip), plus *boundary-aware sampling*
      - Uses *multi-start schedule* with Halton exploration + periodic "CMA restart" on stagnation
      - Uses *cheap coordinate polish* near the end (few evals, high ROI)

    Returns: best objective value found within max_time.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    # ---------- bounds ----------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")
    spans_nz = [s if s > 0.0 else 1.0 for s in spans]

    def now():
        return time.time()

    def evaluate(x):
        return float(func(x))

    # ---------- RNG helpers ----------
    _has_spare = False
    _spare = 0.0

    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _spare = z1
        _has_spare = True
        return z0

    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        y = (v - lo) % (2.0 * w)
        return (lo + y) if (y <= w) else (hi - (y - w))

    def mirror_point(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    # ---------- Halton (for exploration / restarts) ----------
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

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------- small elite archive ----------
    elite_size = max(6, min(24, 6 + int(2.5 * math.sqrt(dim))))
    elites = []  # list of (f, x)

    def push_elite(fx, x):
        nonlocal elites
        item = (fx, x[:])
        if not elites:
            elites = [item]
            return
        if len(elites) >= elite_size and fx >= elites[-1][0]:
            return
        # insert sorted
        lo, hi = 0, len(elites)
        while lo < hi:
            mid = (lo + hi) // 2
            if fx < elites[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        elites.insert(lo, item)
        if len(elites) > elite_size:
            elites.pop()

    def get_best():
        if not elites:
            return float("inf"), None
        return elites[0][0], elites[0][1][:]

    # ---------- initialization: Halton + opposition + random ----------
    best = float("inf")
    best_x = None

    init_n = max(24, min(220, 30 + 14 * int(math.sqrt(dim))))
    for _ in range(init_n):
        if now() >= deadline:
            return best

        if random.random() < 0.80:
            x = halton_point(hal_k)
            hal_k += 1
        else:
            x = rand_uniform_point()

        fx = evaluate(x)
        push_elite(fx, x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition point (cheap extra probe)
        if now() >= deadline:
            return best
        xo = [lows[i] + highs[i] - x[i] for i in range(dim)]
        mirror_point(xo)
        fo = evaluate(xo)
        push_elite(fo, xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]
        push_elite(best, best_x)

    # ---------- Diagonal-CMA-like state ----------
    # mean
    m = best_x[:]
    fm = best

    # diagonal stds (in absolute units)
    # start moderately wide; tie to spans
    sigma0 = 0.25
    d = [max(1e-12, sigma0 * spans_nz[i]) for i in range(dim)]

    # evolution path (momentum)
    p = [0.0] * dim

    # population sizes (small, time-friendly)
    lam = max(10, min(44, 12 + 4 * int(math.sqrt(dim))))
    mu = max(4, min(lam // 2, 6 + int(1.5 * math.sqrt(dim))))
    if mu >= lam:
        mu = max(2, lam // 2)

    # recombination weights (log)
    weights = []
    for i in range(mu):
        weights.append(math.log(mu + 0.5) - math.log(i + 1.0))
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # learning rates (gentle; diagonal only)
    c_m = 1.0  # mean learning rate
    c_p = 0.25  # path learning rate
    c_d = 0.20  # diag std learning rate

    # stagnation/restart controls
    no_best_gens = 0
    gen = 0

    # coordinate polish controls
    polish_period = max(7, 3 + int(math.sqrt(dim)))
    polish_coords = max(1, min(dim, 10))

    while now() < deadline:
        gen += 1

        # parent mean for sampling: use archive best sometimes to reduce drift
        if elites and random.random() < 0.20:
            m = elites[0][1][:]

        # sample offspring
        pop = []  # (f, x, z)
        # boundary-aware: with small probability, sample near bounds to handle optima on edges
        near_bounds = (random.random() < 0.08)

        for _ in range(lam):
            if now() >= deadline:
                break
            z = [randn() for _ in range(dim)]
            x = [0.0] * dim
            for i in range(dim):
                if spans[i] == 0.0:
                    x[i] = lows[i]
                else:
                    step = d[i] * z[i] + 0.20 * p[i]  # path injection
                    x[i] = m[i] + step

                    if near_bounds and random.random() < 0.25:
                        # snap some coordinates toward a random bound (exploration of edges)
                        x[i] = lows[i] if (random.random() < 0.5) else highs[i]

            mirror_point(x)
            fx = evaluate(x)
            pop.append((fx, x, z))

        if not pop:
            break

        pop.sort(key=lambda t: t[0])

        # update elites / best
        for j in range(min(len(pop), max(2, lam // 3))):
            push_elite(pop[j][0], pop[j][1])

        best_new, best_x_new = get_best()
        if best_new < best:
            best, best_x = best_new, best_x_new
            no_best_gens = 0
        else:
            no_best_gens += 1

        # recombine mean using top mu
        old_m = m[:]
        m = [0.0] * dim
        for i in range(mu):
            wi = weights[i]
            xi = pop[i][1]
            for k2 in range(dim):
                m[k2] += wi * xi[k2]

        # update evolution path based on mean shift
        # normalize by d to get approx z-space step
        for i in range(dim):
            if spans[i] == 0.0:
                p[i] = 0.0
                continue
            step_norm = (m[i] - old_m[i]) / max(1e-12, d[i])
            p[i] = (1.0 - c_p) * p[i] + c_p * step_norm * d[i]

        # update diagonal stds using weighted z^2 from selected individuals
        # (keeps d positive, adapts anisotropy)
        for i in range(dim):
            if spans[i] == 0.0:
                d[i] = 1e-12
                continue
            ez2 = 0.0
            for j in range(mu):
                wi = weights[j]
                zj = pop[j][2][i]
                ez2 += wi * (zj * zj)
            # target ez2 ~ 1.0; if <1 shrink, if >1 grow
            factor = math.exp(0.5 * c_d * (ez2 - 1.0))
            di = d[i] * factor
            # keep within reasonable range tied to span
            lo = 1e-12
            hi = 0.9 * spans_nz[i]
            if di < lo:
                di = lo
            elif di > hi:
                di = hi
            d[i] = di

        # accept mean's fitness sometimes (not necessary but stabilizes bookkeeping)
        # (evaluate mean occasionally to keep a "current" anchor)
        if gen % 3 == 0 and now() < deadline:
            m_eval = m[:]
            mirror_point(m_eval)
            fm_eval = evaluate(m_eval)
            push_elite(fm_eval, m_eval)
            if fm_eval < best:
                best, best_x = fm_eval, m_eval[:]

        # restart / diversification on stagnation
        if no_best_gens >= (10 + int(2.2 * math.sqrt(dim))):
            no_best_gens = 0
            # restart mean from a fresh Halton point or from a random elite
            if random.random() < 0.65:
                m = halton_point(hal_k)
                hal_k += 1
            else:
                if elites:
                    m = elites[min(len(elites) - 1, int((random.random() ** 2) * len(elites)))][1][:]
                else:
                    m = rand_uniform_point()

            # reset path, re-widen d
            p = [0.0] * dim
            widen = 2.0 ** random.uniform(-0.25, 0.75)  # usually widen a bit
            for i in range(dim):
                d[i] = max(1e-12, min(0.9 * spans_nz[i], d[i] * widen))

        # endgame coordinate polish (limited cost)
        time_left = deadline - now()
        if time_left <= 0:
            break
        endgame = (time_left / float(max_time)) < 0.22 if max_time > 0 else True
        if endgame or (gen % polish_period == 0):
            # pick most "uncertain" coords (largest d)
            idxs = list(range(dim))
            idxs.sort(key=lambda i: d[i], reverse=True)
            idxs = idxs[:polish_coords]

            x0 = best_x[:]
            f0 = best

            for i in idxs:
                if now() >= deadline or spans[i] == 0.0:
                    continue
                delta = max(1e-12, (0.6 if endgame else 1.0) * d[i])

                xp = x0[:]
                xp[i] += delta
                mirror_point(xp)
                fp = evaluate(xp)

                xm = x0[:]
                xm[i] -= delta
                mirror_point(xm)
                fm2 = evaluate(xm)

                if fp < f0 or fm2 < f0:
                    if fp <= fm2:
                        x0, f0 = xp, fp
                    else:
                        x0, f0 = xm, fm2

            if f0 < best:
                best, best_x = f0, x0[:]
                push_elite(f0, x0)
                # tighten around improved point
                for i in range(dim):
                    d[i] = max(1e-12, d[i] * 0.85)

    return best
