import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Key upgrades vs previous CMA-like code:
      - Much lower overhead per evaluation (critical when func is cheap / time is tight)
      - (1+1)-ES with 1/5th success rule + occasional heavy-tail jumps (Cauchy)
      - Alternating search directions: isotropic, coordinate, and random subspace
      - Periodic inexpensive pattern-search polish on current best
      - Multi-start via low-discrepancy sampling (Halton) + random injection
      - Strict bound handling via reflection (no wasted samples)

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        return float("inf")

    # -------- bounds / transforms --------
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]
    # handle degenerate bounds
    for i in range(dim):
        if not (math.isfinite(lo[i]) and math.isfinite(hi[i])) or span[i] < 0:
            return float("inf")
        if span[i] == 0:
            hi[i] = lo[i]

    def reflect(v, a, b):
        # reflect into [a,b]
        if v < a or v > b:
            w = b - a
            if w <= 0.0:
                return a
            t = (v - a) / w
            t = t % 2.0
            if t > 1.0:
                t = 2.0 - t
            v = a + t * w
        # final clamp (numeric safety)
        if v < a: v = a
        if v > b: v = b
        return v

    # -------- safe evaluation + tiny cache --------
    # Coarse cache to avoid duplicate evals during polish / reflections
    cache = {}
    # cache resolution relative to bounds
    res = []
    for i in range(dim):
        s = span[i]
        if s <= 0.0:
            res.append(0.0)
        else:
            # about 2e5 buckets per dim scale -> coarse but helpful
            res.append(200000.0 / s)

    def key_x(x):
        # quantize
        return tuple(int((x[i] - lo[i]) * res[i]) if res[i] > 0 else 0 for i in range(dim))

    def evaluate(x):
        k = key_x(x)
        v = cache.get(k)
        if v is not None:
            return v
        try:
            y = func(x)
            if y is None:
                y = float("inf")
            y = float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")
        cache[k] = y
        return y

    # -------- low-discrepancy init (Halton) --------
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(dim)

    def halton_single(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton_single(k, primes[i])
            x[i] = lo[i] + u * span[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # -------- mutation operators --------
    def cauchy():
        # heavy tail: tan(pi*(u-0.5))
        u = random.random()
        # avoid infinities
        u = 1e-12 if u < 1e-12 else (1.0 - 1e-12 if u > 1.0 - 1e-12 else u)
        return math.tan(math.pi * (u - 0.5))

    def propose(x, sig, mode, subspace_k):
        # mode:
        #  0 isotropic gaussian
        #  1 coordinate gaussian (one dim)
        #  2 random subspace gaussian (k dims)
        #  3 heavy-tail jump (cauchy) on subspace
        y = x[:]  # copy
        if mode == 1:
            j = random.randrange(dim)
            yj = y[j] + random.gauss(0.0, sig[j])
            y[j] = reflect(yj, lo[j], hi[j])
            return y

        if mode == 2:
            k = subspace_k
            if k >= dim:
                idxs = range(dim)
            else:
                # sample k unique dims cheaply
                idxs = random.sample(range(dim), k)
            for j in idxs:
                yj = y[j] + random.gauss(0.0, sig[j])
                y[j] = reflect(yj, lo[j], hi[j])
            return y

        if mode == 3:
            k = subspace_k
            if k >= dim:
                idxs = range(dim)
            else:
                idxs = random.sample(range(dim), k)
            # scale jump a bit larger than gaussian
            for j in idxs:
                yj = y[j] + 0.85 * sig[j] * cauchy()
                y[j] = reflect(yj, lo[j], hi[j])
            return y

        # mode 0: isotropic gaussian (all dims)
        for j in range(dim):
            yj = y[j] + random.gauss(0.0, sig[j])
            y[j] = reflect(yj, lo[j], hi[j])
        return y

    # -------- optional pattern polish (cheap coordinate pattern search) --------
    def polish(best_x, best_f, step_frac, rounds):
        x = best_x[:]
        fx = best_f
        # randomized coordinate order each round
        for _ in range(rounds):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for j in order:
                if time.time() >= deadline:
                    break
                sj = span[j]
                if sj <= 0.0:
                    continue
                step = step_frac * sj
                if step <= 0.0:
                    continue
                base = x[j]
                # try a small stencil
                candidates = (base + step, base - step, base + 0.5 * step, base - 0.5 * step)
                bestj = base
                bestfj = fx
                for v in candidates:
                    if v < lo[j] or v > hi[j]:
                        continue
                    xt = x[:]
                    xt[j] = v
                    f2 = evaluate(xt)
                    if f2 < bestfj:
                        bestfj = f2
                        bestj = v
                if bestfj < fx:
                    x[j] = bestj
                    fx = bestfj
                    improved = True
            if not improved:
                break
        return x, fx

    # -------- initialization --------
    best = float("inf")
    best_x = None

    # Initial samples: Halton + random
    # Keep modest to leave time for local search; also adapts to dim.
    n0 = max(16, min(120, 8 * dim))
    hstart = 1 + random.randrange(256)
    for i in range(n0):
        if time.time() >= deadline:
            return best
        if i < (n0 * 3) // 4:
            x = halton_point(hstart + i)
        else:
            x = rand_point()
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        x = rand_point()
        best = evaluate(x)
        best_x = x

    # Per-dimension sigma (absolute), start relatively broad then adapt
    sig = [0.25 * s if s > 0.0 else 1e-12 for s in span]
    sig_min = [max(1e-12, 1e-9 * (s if s > 0.0 else 1.0)) for s in span]
    sig_max = [0.9 * (s if s > 0.0 else 1.0) for s in span]

    # Success rule state
    succ_ema = 0.2
    target = 0.2  # 1/5th
    it = 0
    stall = 0

    # -------- main loop --------
    while time.time() < deadline:
        it += 1

        # Offspring count: small but >1 for stability; still low overhead.
        lam = 6 if dim <= 10 else (8 if dim <= 30 else 10)
        # Choose mutation regime
        r = random.random()
        if stall > 60 and r < 0.35:
            mode = 3  # heavy-tail to escape
        elif r < 0.55:
            mode = 2  # subspace
        elif r < 0.80:
            mode = 1  # coordinate
        else:
            mode = 0  # isotropic

        subspace_k = 1
        if mode in (2, 3):
            # adapt subspace size with dim
            subspace_k = 2 if dim <= 8 else (max(2, dim // 6))

        parent_x = best_x
        parent_f = best

        best_off_x = None
        best_off_f = float("inf")

        for _ in range(lam):
            if time.time() >= deadline:
                break
            x2 = propose(parent_x, sig, mode, subspace_k)
            f2 = evaluate(x2)
            if f2 < best_off_f:
                best_off_f = f2
                best_off_x = x2

        if best_off_x is None:
            break

        improved = best_off_f < parent_f
        succ_ema = 0.90 * succ_ema + 0.10 * (1.0 if improved else 0.0)

        if improved:
            best_x = best_off_x
            best = best_off_f
            stall = 0
        else:
            stall += 1

        # 1/5th-like step-size adaptation (smooth, per-dimension scaled together)
        if succ_ema > target:
            mult = 1.06
        else:
            mult = 0.94

        # If stalling, gently shrink; if heavily stalling, inject exploration
        if stall > 120:
            mult *= 0.92
        elif stall > 40 and random.random() < 0.15:
            mult *= 1.15

        for j in range(dim):
            sj = sig[j] * mult
            if sj < sig_min[j]:
                sj = sig_min[j]
            elif sj > sig_max[j]:
                sj = sig_max[j]
            sig[j] = sj

        # Periodic polish (very cheap local improvements)
        if it % 25 == 0 and time.time() < deadline:
            step_frac = 0.01 if stall < 30 else 0.02
            x3, f3 = polish(best_x, best, step_frac=step_frac, rounds=1)
            if f3 < best:
                best = f3
                best_x = x3
                stall = 0

        # Soft restart injection when badly stuck: replace parent with a new global point
        if stall > 220 and time.time() < deadline:
            xg = halton_point(hstart + n0 + it)
            fg = evaluate(xg)
            if fg < best:
                best = fg
                best_x = xg
            # reset sigmas moderately
            for j in range(dim):
                sig[j] = max(sig_min[j], min(sig_max[j], 0.30 * (span[j] if span[j] > 0.0 else 1.0)))
            stall = 0
            succ_ema = 0.2

    return best
