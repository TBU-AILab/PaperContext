import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no external libs).

    Design goals:
      - Strong anytime performance under tight max_time.
      - Robust on noisy/ill-conditioned/box-bounded objectives.
      - Better than simple ES by adding:
          * low-discrepancy + LHS seeding (diversity),
          * (1+λ) adaptive diagonal search with two-point gradient estimate (SPG),
          * coordinate/pattern refinement near the end,
          * restarts with budget-aware radius,
          * evaluation cache with time-adaptive quantization.

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0:
        return float("inf")
    deadline = t0 + float(max_time)

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

    # time-adaptive cache quantization: coarse early, finer late
    cache = {}
    def quant_key(x, q):
        k = []
        for i in range(dim):
            u = (x[i] - lows[i]) / spans[i]
            if u < 0.0: u = 0.0
            if u > 1.0: u = 1.0
            k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0
        # q from 2^15 .. 2^27 (wide to avoid harming late precision)
        q = 1 << (15 + int(12 * frac))
        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # cheap N(0,1) (CLT)
    def randn():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0) * 0.7071067811865475

    def cauchy():
        u = random.random()
        if u <= 1e-12: u = 1e-12
        if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # scrambled Halton for early global coverage
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

    # ---------------- init / seeding ----------------
    best_x = rand_vec()
    best = eval_cached(best_x)

    # budget-aware seeding
    seed_n = max(24, 10 * dim)
    seed_n = min(seed_n, 600)
    if max_time < 0.05:
        seed_n = max(3, min(seed_n, 12))

    # add a light LHS component to reduce projection clustering
    strata = seed_n
    lhs_perms = []
    for i in range(dim):
        p = list(range(strata))
        random.shuffle(p)
        lhs_perms.append(p)

    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best

        # Mix: mostly Halton early, some LHS, some pure random
        r = random.random()
        if r < 0.55:
            x = halton_point(k)
        elif r < 0.85:
            kk = (k - 1) % strata
            x = [0.0] * dim
            for i in range(dim):
                a = lhs_perms[i][kk] / float(strata)
                b = (lhs_perms[i][kk] + 1) / float(strata)
                u = a + (b - a) * random.random()
                x[i] = lows[i] + u * spans[i]
        else:
            x = rand_vec()

        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition point (cheap second guess)
        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - x[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

    # ---------------- main optimizer ----------------
    x = best_x[:]
    fx = best

    # diagonal step sizes
    sigma = [0.20 * spans[i] for i in range(dim)]
    sigma_min = [1e-16 * spans[i] + 1e-18 for i in range(dim)]
    sigma_max = [0.80 * spans[i] for i in range(dim)]

    # momentum-ish direction and SPG direction
    path = [0.0] * dim
    gdir = [0.0] * dim  # estimated descent direction (normalized-ish later)
    path_decay = 0.82
    g_decay = 0.75

    # (1+λ) offspring (fast selection pressure, good anytime)
    lam = 10 if dim <= 10 else (14 if dim <= 30 else 18)
    lam = min(max(6, lam), 28)

    # success-based sigma adaptation
    win = 18
    succ = 0
    tri = 0
    target = 0.20
    shrink = 0.87

    # pattern refinement controls
    pattern_every = 10
    tr = 0.10
    tr_min = 1e-16
    tr_max = 0.50

    # restart controls
    gen = 0
    no_improve = 0
    last_best = best
    restart_every = 45

    # SPG (two-point) controls
    spg_every = 4  # every few generations compute a cheap 2-point estimate
    spg_scale = 0.12  # relative to sigma (per-dim) used for probing

    while True:
        if time.time() >= deadline:
            return best
        gen += 1

        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # ---- occasional two-point "gradient" estimate (SPSA-like, but deterministic pair) ----
        if gen % spg_every == 0 and time.time() < deadline:
            # random +/-1 direction (sparse in high dim)
            k_sparse = dim if dim <= 12 else max(4, dim // 5)
            idxs = random.sample(range(dim), k_sparse) if k_sparse < dim else list(range(dim))

            d = [0.0] * dim
            for i in idxs:
                d[i] = -1.0 if random.random() < 0.5 else 1.0

            # step per dim
            x1 = x[:]
            x2 = x[:]
            for i in idxs:
                h = spg_scale * (sigma[i] + 1e-18)
                x1[i] += h * d[i]
                x2[i] -= h * d[i]
            clip_inplace(x1)
            clip_inplace(x2)

            f1 = eval_cached(x1)
            if time.time() >= deadline:
                return best
            f2 = eval_cached(x2)

            # If either probe improves global best, accept immediately
            if f1 < best:
                best, best_x = f1, x1[:]
            if f2 < best:
                best, best_x = f2, x2[:]

            # build descent direction along d (sign from difference)
            # g ~ (f1 - f2) * d
            df = (f1 - f2)
            # update gdir with decay
            for i in idxs:
                gdir[i] = g_decay * gdir[i] + (1.0 - g_decay) * (df * d[i])

        # ---- generate offspring ----
        pop = []
        # heavy tail early / when high-dim; cool down over time
        p_heavy = 0.10 * (1.0 - 0.75 * frac) + (0.03 if dim > 25 else 0.0)

        # sparse perturbations in high dim
        k_mut = dim if dim <= 10 else max(3, dim // 4)

        # normalize gdir cheaply for use (only if it has energy)
        gnorm = math.sqrt(sum(gi * gi for gi in gdir))
        use_g = (gnorm > 1e-18)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            y = x[:]

            if random.random() < p_heavy:
                # Cauchy kick around global best in subset dims
                center = best_x
                kk = 1 if dim <= 5 else max(1, dim // 3)
                sel = random.sample(range(dim), kk) if kk < dim else list(range(dim))
                for i in sel:
                    y[i] = center[i] + cauchy() * (2.4 * sigma[i] + 1e-12)
            else:
                sel = random.sample(range(dim), k_mut) if k_mut < dim else list(range(dim))

                # combine: momentum path + (optional) estimated descent direction + gaussian noise
                for i in sel:
                    drift = 0.60 * path[i]
                    if use_g:
                        # move opposite gdir (descent), scaled to sigma units
                        drift += (-0.18 * (gdir[i] / gnorm)) * (sigma[i] + 1e-18)
                    y[i] += drift + randn() * sigma[i]

            clip_inplace(y)
            fy = eval_cached(y)
            pop.append((fy, y))

        pop.sort(key=lambda t: t[0])
        best_off_f, best_off_x = pop[0]

        tri += 1

        # accept best offspring if it improves current
        if best_off_f < fx:
            # update momentum path
            for i in range(dim):
                step_i = best_off_x[i] - x[i]
                path[i] = path_decay * path[i] + (1.0 - path_decay) * step_i

            x, fx = best_off_x[:], best_off_f
            succ += 1
            no_improve = 0

            if fx < best:
                best, best_x = fx, x[:]
        else:
            no_improve += 1

        # success rule for sigma
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

        # ---- occasional pattern refinement (coordinate steps) ----
        if gen % pattern_every == 0 and time.time() < deadline:
            center = best_x if frac > 0.45 else x
            # try only a subset of dimensions for speed
            dims = list(range(dim))
            random.shuffle(dims)
            nd = min(dim, 10 if dim <= 40 else 14)
            dims = dims[:nd]

            step_shrink = (0.80 * (1.0 - frac) + 0.20)
            improved = False
            for i in dims:
                if time.time() >= deadline:
                    return best
                step = tr * step_shrink * spans[i]
                if step <= 0.0:
                    continue
                for sgn in (-1.0, 1.0):
                    y = center[:]
                    y[i] += sgn * step
                    clip_inplace(y)
                    fy = eval_cached(y)
                    if fy < best:
                        best, best_x = fy, y[:]
                        x, fx = best_x[:], best
                        improved = True
                        # bias path a bit toward the improvement
                        for j in range(dim):
                            path[j] = 0.75 * path[j] + 0.25 * (best_x[j] - center[j])
                        break

            if improved:
                tr = min(tr_max, tr * 1.10)
            else:
                tr = max(tr_min, tr * 0.82)

        # ---- stagnation handling ----
        if no_improve > 6 * max(1, dim // 3):
            # inflate sigma + damp paths to escape
            for i in range(dim):
                sigma[i] = min(sigma_max[i], sigma[i] * 1.35 + 1e-18)
                path[i] *= 0.2
                gdir[i] *= 0.4
            tr = min(tr_max, tr * 1.15)
            no_improve = 0

        # ---- periodic restart around global best ----
        if gen % restart_every == 0:
            if best >= last_best - 1e-15:
                rad = (0.55 * (1.0 - frac) + 0.05)
                x = best_x[:]
                for i in range(dim):
                    x[i] += (2.0 * random.random() - 1.0) * (rad * spans[i])
                clip_inplace(x)
                fx = eval_cached(x)

                # reset dynamics
                path = [0.0] * dim
                gdir = [0.0] * dim
                for i in range(dim):
                    sigma[i] = max(sigma_min[i], min(sigma_max[i], (0.22 + 0.10 * (1.0 - frac)) * spans[i]))
                tr = max(tr_min, min(tr_max, 0.14 * (1.0 - frac) + 0.03))

            last_best = best

        # cheap opposition probe of best occasionally
        if random.random() < 0.02 and time.time() < deadline:
            opp = [lows[i] + highs[i] - best_x[i] for i in range(dim)]
            clip_inplace(opp)
            fo = eval_cached(opp)
            if fo < best:
                best, best_x = fo, opp[:]
                x, fx = best_x[:], best
                path = [0.0] * dim
                gdir = [0.0] * dim
