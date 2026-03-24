import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Key changes vs previous:
      - Best-first multi-start local search (keeps several good incumbents, not just one).
      - Coordinate/local pattern search (fast improvement on separable-ish landscapes).
      - Adaptive per-dimension step sizes with success-based growth/shrink.
      - Occasional Cauchy (heavy-tail) jumps to escape local minima.
      - Quasi-opposition moves for cheaper global exploration.
      - Always respects time limit and bounds; robust to bad func values.

    Returns:
      best (float): best objective value found.
    """

    # ---------------- helpers ----------------
    eps = 1e-12

    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def safe_eval(x):
        try:
            y = func(x)
            if y is None:
                return float("inf")
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf")
            return y
        except Exception:
            return float("inf")

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def center_vec():
        return [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]

    def opposition_vec(x):
        # opposite point in box
        return [bounds[i][0] + bounds[i][1] - x[i] for i in range(dim)]

    def quasi_opposition_vec(x):
        # "quasi opposite": between center and opposite
        c = center_vec()
        o = opposition_vec(x)
        return [c[i] + random.random() * (o[i] - c[i]) for i in range(dim)]

    def spans():
        return [max(bounds[i][1] - bounds[i][0], 0.0) for i in range(dim)]

    def in_bounds(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    # ---------------- time control ----------------
    start = time.time()
    deadline = start + max(0.0, float(max_time))

    def time_left():
        return time.time() < deadline

    if dim <= 0:
        # Degenerate: no parameters.
        return safe_eval([])

    sp = spans()
    avg_span = sum(max(s, eps) for s in sp) / float(dim)

    # ---------------- initialization: diversified sampling ----------------
    # Try center, randoms, plus quasi-opposition of good points.
    best = float("inf")
    x_best = center_vec()
    x_best = in_bounds(x_best)
    best = safe_eval(x_best)

    # Build initial candidate pool
    init_n = max(12, 8 * dim)
    pool = []  # list of (f, x)
    pool.append((best, list(x_best)))

    for _ in range(init_n):
        if not time_left():
            return best
        x = rand_uniform_vec()
        fx = safe_eval(x)
        pool.append((fx, x))
        if fx < best:
            best, x_best = fx, list(x)

    # Add quasi-opposition points from current best few
    pool.sort(key=lambda t: t[0])
    for j in range(min(5, len(pool))):
        if not time_left():
            return best
        x = pool[j][1]
        xo = quasi_opposition_vec(x)
        xo = in_bounds(xo)
        fo = safe_eval(xo)
        pool.append((fo, xo))
        if fo < best:
            best, x_best = fo, list(xo)

    # Keep top-K starts (best-first multi-start)
    K = max(3, min(12, 2 * dim))
    pool.sort(key=lambda t: t[0])
    starts = [list(pool[i][1]) for i in range(min(K, len(pool)))]
    start_fs = [pool[i][0] for i in range(min(K, len(pool)))]

    # ---------------- local search settings ----------------
    # Per-dimension step sizes (start moderate)
    steps0 = []
    for i in range(dim):
        s = sp[i]
        if s <= eps:
            steps0.append(0.0)
        else:
            steps0.append(0.15 * s)  # a bit larger than before to move faster initially

    # pattern search parameters
    grow = 1.25
    shrink = 0.55
    min_step = [1e-9 * max(sp[i], 1.0) for i in range(dim)]

    # small "polish" gaussian scale relative to step
    def gauss_perturb(x, steps, scale=0.35):
        xn = x[:]
        for i in range(dim):
            if steps[i] > 0.0:
                xn[i] = clip(xn[i] + random.gauss(0.0, scale * steps[i]),
                             bounds[i][0], bounds[i][1])
        return xn

    # heavy-tail jump (Cauchy-like) for escape
    def cauchy_jump(x, strength=0.35):
        xn = x[:]
        for i in range(dim):
            if sp[i] <= eps:
                continue
            # tan(pi*(u-0.5)) is Cauchy(0,1)
            u = random.random()
            c = math.tan(math.pi * (u - 0.5))
            xn[i] = clip(xn[i] + strength * sp[i] * c, bounds[i][0], bounds[i][1])
        return xn

    # ---------------- main optimization loop ----------------
    # We'll cycle through the best starts and improve each; keep refreshing starts
    # with newly found good points.
    idx = 0

    while time_left():
        # pick an incumbent (best-first cycling)
        if len(starts) == 0:
            starts = [rand_uniform_vec()]
            start_fs = [safe_eval(starts[0])]

        x = starts[idx % len(starts)][:]
        fx = start_fs[idx % len(starts)]
        idx += 1

        # fresh per-incumbent steps
        steps = steps0[:]

        # If we don't have fx (or it is inf), re-evaluate (rare)
        if not math.isfinite(fx):
            fx = safe_eval(x)

        # Local improvement budget (dimension-dependent)
        # Use many cheap pattern moves + occasional stochastic probes.
        iters = max(60, 35 * dim)

        no_gain = 0
        for t in range(iters):
            if not time_left():
                return best

            improved = False

            # ---- (1) coordinate pattern search (fast) ----
            # Randomized coordinate order each sweep
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if not time_left():
                    return best
                if steps[i] <= 0.0:
                    continue

                # try +step and -step
                xi = x[i]

                x1 = x[:]
                x1[i] = clip(xi + steps[i], bounds[i][0], bounds[i][1])
                f1 = safe_eval(x1)

                x2 = x[:]
                x2[i] = clip(xi - steps[i], bounds[i][0], bounds[i][1])
                f2 = safe_eval(x2)

                if f1 < fx or f2 < fx:
                    if f1 <= f2:
                        x, fx = x1, f1
                    else:
                        x, fx = x2, f2
                    improved = True

                    # modest grow on success (per-dim)
                    steps[i] = min(steps[i] * grow, 0.5 * sp[i] if sp[i] > 0 else steps[i])

                    if fx < best:
                        best, x_best = fx, x[:]
                else:
                    # shrink this coordinate if it keeps failing
                    steps[i] = max(steps[i] * shrink, min_step[i])

            # ---- (2) stochastic "polish" near current point ----
            if time_left() and random.random() < 0.30:
                xn = gauss_perturb(x, steps, scale=0.25)
                fn = safe_eval(xn)
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    if fx < best:
                        best, x_best = fx, x[:]

            # ---- (3) escape / exploration when stuck ----
            if not improved:
                no_gain += 1
            else:
                no_gain = 0

            # If stagnating, try one of: quasi-opposition, cauchy jump, random restart
            if no_gain >= max(6, 2 * dim):
                no_gain = 0
                r = random.random()
                if r < 0.45:
                    xn = quasi_opposition_vec(x)
                    xn = in_bounds(xn)
                elif r < 0.80:
                    xn = cauchy_jump(x, strength=0.20 + 0.25 * random.random())
                else:
                    xn = rand_uniform_vec()

                fn = safe_eval(xn)
                if fn < fx:
                    x, fx = xn, fn
                    # reset steps a bit larger after a successful jump
                    steps = [max(s, 0.08 * sp[i]) for i, s in enumerate(steps)]
                    if fx < best:
                        best, x_best = fx, x[:]

            # Early stop if steps became tiny across all dims (local optimum)
            if all(steps[i] <= min_step[i] * 2.0 for i in range(dim)):
                break

        # ---- Update multi-start pool with improved incumbent ----
        # Insert (fx,x) into starts if it's among best K
        # Also keep the global best always present.
        candidates = list(zip(start_fs, starts))
        candidates.append((fx, x[:]))
        candidates.append((best, x_best[:]))

        # Deduplicate loosely by rounding (cheap)
        seen = set()
        uniq = []
        for fval, xv in sorted(candidates, key=lambda t: t[0]):
            key = tuple(int(1e6 * (xv[i] - bounds[i][0]) / (sp[i] + eps)) for i in range(dim))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((fval, xv))
            if len(uniq) >= K:
                break

        start_fs = [u[0] for u in uniq]
        starts = [u[1] for u in uniq]

        # Add a bit of fresh diversity occasionally
        if time_left() and random.random() < 0.20 and len(starts) < K + 2:
            xr = rand_uniform_vec()
            fr = safe_eval(xr)
            if math.isfinite(fr):
                start_fs.append(fr)
                starts.append(xr)

    return best
