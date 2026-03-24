import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libraries).

    Key improvements vs your listed algorithms:
      - Stronger exploitation near the best: adaptive coordinate/pattern search
        with per-dimension step sizes + occasional quadratic (1D) refinement.
      - Better exploration without heavy overhead: low-discrepancy seeding (Halton)
        + LHS mix + opposition, then a compact (1+λ)-ES with diagonal sigmas.
      - Smarter restart logic: detects stagnation; restarts with radius tied to
        remaining time and progress.
      - Cache is safer: quantization depends on time AND current step scale, so
        it won't destroy late-stage precision when sigmas become tiny.
      - Robust eval wrapper (NaN/inf/exceptions).

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0:
        return float("inf")
    max_time = float(max_time)
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if not (spans[i] > 0.0):
            spans[i] = 1.0

    # ---------------- helpers ----------------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # cheap approx N(0,1) via CLT
    def randn():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0) * 0.7071067811865475

    def cauchy():
        u = random.random()
        if u <= 1e-12:
            u = 1e-12
        elif u >= 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------------- cache (time + scale adaptive quantization) ----------------
    cache = {}

    def quant_key(x, q):
        k = []
        for i in range(dim):
            u = (x[i] - lows[i]) / spans[i]
            if u < 0.0: u = 0.0
            if u > 1.0: u = 1.0
            k.append(int(u * q + 0.5))
        return tuple(k)

    # scale-aware: if typical sigma is tiny, increase q to avoid merging distinct points
    def eval_cached(x, sigma_scale_hint):
        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # base q increases with time
        base_pow = 15 + int(12 * frac)  # 2^15 .. 2^27
        # extra resolution when the search radius is small
        # sigma_scale_hint ~ typical sigma/span in [0,1]
        # when sigma_scale_hint -> 0, add bits
        if sigma_scale_hint <= 0.0:
            extra = 8
        else:
            extra = int(max(0.0, min(8.0, -math.log(max(1e-18, sigma_scale_hint), 2.0) - 2.0)))
        q = 1 << min(30, base_pow + extra)

        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    # ---------------- low-discrepancy seeding ----------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
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

    def van_der_corput(n, base, perm):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, d = divmod(n, base)
            d = perm[d]
            denom *= base
            v += d / denom
        return v

    primes = first_primes(dim)
    perms = []
    for b in primes:
        p = list(range(b))
        random.shuffle(p)
        perms.append(p)

    def halton_point(idx):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(idx, primes[i], perms[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # LHS perms for mix
    def lhs_points(n):
        strata = n
        perms_lhs = []
        for i in range(dim):
            p = list(range(strata))
            random.shuffle(p)
            perms_lhs.append(p)
        for k in range(strata):
            x = [0.0] * dim
            for i in range(dim):
                a = perms_lhs[i][k] / float(strata)
                b = (perms_lhs[i][k] + 1) / float(strata)
                u = a + (b - a) * random.random()
                x[i] = lows[i] + u * spans[i]
            yield x

    # ---------------- init ----------------
    # initial sigma hint for caching during seeding
    seed_sigma_hint = 0.25

    best_x = rand_vec()
    best = eval_cached(best_x, seed_sigma_hint)

    seed_n = max(24, 10 * dim)
    seed_n = min(seed_n, 700)
    if max_time < 0.05:
        seed_n = max(3, min(seed_n, 12))

    # seed: mix Halton / LHS / random + opposition
    # (opposition often helps on bounded problems)
    lhs_iter = lhs_points(max(8, seed_n // 3))
    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best

        r = random.random()
        if r < 0.50:
            x = halton_point(k)
        elif r < 0.85:
            try:
                x = next(lhs_iter)
            except StopIteration:
                x = rand_vec()
        else:
            x = rand_vec()

        fx = eval_cached(x, seed_sigma_hint)
        if fx < best:
            best, best_x = fx, x[:]

        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - x[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp, seed_sigma_hint)
        if fo < best:
            best, best_x = fo, opp[:]

    # ---------------- main optimizer: (1+λ)-ES + pattern search polishing ----------------
    x = best_x[:]
    fx = best

    # diagonal sigmas
    sigma = [0.18 * spans[i] for i in range(dim)]
    sigma_min = [1e-16 * spans[i] + 1e-18 for i in range(dim)]
    sigma_max = [0.80 * spans[i] for i in range(dim)]

    # success adaptation
    win = 18
    succ = 0
    tri = 0
    target = 0.22
    shrink = 0.87

    # momentum-ish path
    path = [0.0] * dim
    path_decay = 0.85

    # population size
    lam = 10 if dim <= 12 else (14 if dim <= 30 else 18)
    lam = min(max(6, lam), 28)

    # pattern search state: per-dim step sizes (trust radii)
    # start moderately small relative to spans; adapt based on improvements
    tr = [0.08 * spans[i] for i in range(dim)]
    tr_min = [1e-16 * spans[i] + 1e-18 for i in range(dim)]
    tr_max = [0.50 * spans[i] for i in range(dim)]

    # stagnation / restart
    gen = 0
    no_improve = 0
    last_best = best
    last_best_time = time.time()

    # schedule pattern search more often late
    def sigma_hint():
        # typical sigma/span in [0, ~1]
        s = 0.0
        for i in range(dim):
            s += sigma[i] / spans[i]
        return max(1e-18, s / float(dim))

    def pattern_refine(center_x, center_f, budget_dims):
        """Coordinate pattern search with adaptive per-dimension steps + optional 1D quadratic fit."""
        nonlocal best, best_x, x, fx, last_best_time

        # choose subset of dimensions to keep cost bounded
        dims = list(range(dim))
        random.shuffle(dims)
        dims = dims[:budget_dims]

        improved_any = False
        for i in dims:
            if time.time() >= deadline:
                return improved_any

            step = tr[i]
            if step <= 0.0:
                continue

            base = center_x
            basef = center_f

            # try +/- step
            y1 = base[:]
            y1[i] += step
            clip_inplace(y1)
            f1 = eval_cached(y1, sigma_hint())

            if time.time() >= deadline:
                return improved_any

            y2 = base[:]
            y2[i] -= step
            clip_inplace(y2)
            f2 = eval_cached(y2, sigma_hint())

            # accept best among base, y1, y2
            best_local_f = basef
            best_local_x = base
            if f1 < best_local_f:
                best_local_f = f1
                best_local_x = y1
            if f2 < best_local_f:
                best_local_f = f2
                best_local_x = y2

            # optional 1D quadratic refinement if both sides are evaluated
            # Fit parabola through (-h,f2), (0,basef), (+h,f1)
            # x* = h*(f2 - f1) / (2*(f2 - 2f0 + f1))
            denom = (f2 - 2.0 * basef + f1)
            if denom != 0.0 and time.time() < deadline:
                xstar = step * (f2 - f1) / (2.0 * denom)
                # clamp to a reasonable bracket to avoid wild jumps
                if xstar > 2.5 * step:
                    xstar = 2.5 * step
                elif xstar < -2.5 * step:
                    xstar = -2.5 * step
                y3 = base[:]
                y3[i] += xstar
                clip_inplace(y3)
                f3 = eval_cached(y3, sigma_hint())
                if f3 < best_local_f:
                    best_local_f = f3
                    best_local_x = y3

            if best_local_f + 1e-15 < basef:
                # improvement on this coordinate: expand step slightly
                improved_any = True
                tr[i] = min(tr_max[i], tr[i] * 1.18 + 1e-18)

                # update global best if needed
                if best_local_f < best:
                    best = best_local_f
                    best_x = best_local_x[:]
                    last_best_time = time.time()

                # also move current state towards improvement
                x = best_local_x[:]
                fx = best_local_f
                center_x = x
                center_f = fx
            else:
                # no improvement: shrink step
                tr[i] = max(tr_min[i], tr[i] * 0.82)

        return improved_any

    while True:
        if time.time() >= deadline:
            return best
        gen += 1

        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # (1+λ)-ES around current x, with occasional heavy-tail around global best
        p_heavy = 0.12 * (1.0 - 0.75 * frac) + (0.03 if dim > 25 else 0.0)
        k_mut = dim if dim <= 10 else max(3, dim // 4)

        best_off_f = float("inf")
        best_off_x = None

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            y = x[:]
            if random.random() < p_heavy:
                # Cauchy kick around best
                center = best_x
                kk = 1 if dim <= 5 else max(1, dim // 3)
                idxs = random.sample(range(dim), kk) if kk < dim else list(range(dim))
                for i in idxs:
                    y[i] = center[i] + cauchy() * (2.2 * sigma[i] + 1e-12)
            else:
                idxs = random.sample(range(dim), k_mut) if k_mut < dim else list(range(dim))
                for i in idxs:
                    y[i] += 0.65 * path[i] + randn() * sigma[i]

            clip_inplace(y)
            fy = eval_cached(y, sigma_hint())
            if fy < best_off_f:
                best_off_f = fy
                best_off_x = y

        tri += 1
        if best_off_f < fx:
            # accept
            for i in range(dim):
                step_i = best_off_x[i] - x[i]
                path[i] = path_decay * path[i] + (1.0 - path_decay) * step_i

            x = best_off_x[:]
            fx = best_off_f
            succ += 1
            no_improve = 0

            if fx < best:
                best = fx
                best_x = x[:]
                last_best_time = time.time()
        else:
            no_improve += 1
            # mild path damping when rejecting
            for i in range(dim):
                path[i] *= 0.97

        # sigma success rule
        if tri >= win:
            rate = succ / float(tri)
            mult = (1.0 / shrink) if rate > target else shrink
            for i in range(dim):
                s = sigma[i] * mult
                if s < sigma_min[i]: s = sigma_min[i]
                if s > sigma_max[i]: s = sigma_max[i]
                sigma[i] = s
            tri = 0
            succ = 0

        # late-stage polishing: pattern search around best (or current early)
        # increase frequency as time progresses
        if time.time() >= deadline:
            return best
        do_polish = (random.random() < (0.10 + 0.45 * frac))
        if do_polish:
            center = best_x[:] if frac > 0.35 else x[:]
            center_f = best if frac > 0.35 else fx
            budget_dims = min(dim, 8 if dim <= 40 else 12)
            pattern_refine(center, center_f, budget_dims)

        # stagnation escape
        if no_improve > 6 * max(1, dim // 3):
            # inflate sigmas, shrink trust radii slightly, reset path
            for i in range(dim):
                sigma[i] = min(sigma_max[i], sigma[i] * 1.35 + 1e-18)
                tr[i] = max(tr_min[i], min(tr_max[i], tr[i] * 0.90 + 1e-18))
                path[i] *= 0.2
            no_improve = 0

        # restart if long stall (time-aware)
        if gen % (35 + dim // 2) == 0:
            if best >= last_best - 1e-15:
                # radius shrinks with time remaining
                rem = max(0.0, deadline - time.time())
                rem_frac = rem / max_time
                rad = (0.60 * rem_frac + 0.05)
                x = best_x[:]
                for i in range(dim):
                    x[i] += (2.0 * random.random() - 1.0) * (rad * spans[i])
                clip_inplace(x)
                fx = eval_cached(x, sigma_hint())

                # reset dynamics
                path = [0.0] * dim
                for i in range(dim):
                    sigma[i] = max(sigma_min[i], min(sigma_max[i], (0.20 * rem_frac + 0.06) * spans[i]))
                    tr[i] = max(tr_min[i], min(tr_max[i], (0.10 * rem_frac + 0.02) * spans[i]))
            last_best = best

        # if no global improvement for a long time, force a broader restart
        if time.time() - last_best_time > 0.55 * max_time and (time.time() + 0.02) < deadline:
            x = rand_vec()
            fx = eval_cached(x, sigma_hint())
            # slightly broaden sigmas
            for i in range(dim):
                sigma[i] = min(sigma_max[i], max(sigma[i], 0.22 * spans[i]))
                tr[i] = min(tr_max[i], max(tr[i], 0.08 * spans[i]))
            path = [0.0] * dim
            last_best_time = time.time()

        # occasional opposition probe of best (cheap extra)
        if random.random() < 0.02 and time.time() < deadline:
            opp = [lows[i] + highs[i] - best_x[i] for i in range(dim)]
            clip_inplace(opp)
            fo = eval_cached(opp, sigma_hint())
            if fo < best:
                best = fo
                best_x = opp[:]
                x = best_x[:]
                fx = best
                path = [0.0] * dim
                last_best_time = time.time()
