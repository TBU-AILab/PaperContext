import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimization.

    Hybrid strategy:
      1) Low-discrepancy (Halton) global exploration to quickly cover space
      2) Multi-start local refinement with:
         - mirrored Gaussian mutation (tests +/- step) to reduce wasted moves
         - per-dimension step sizes with 1/5 success rule adaptation
         - occasional coordinate/local pattern moves for cheap extra improvement
      3) Stagnation-triggered restarts; best-so-far always preserved

    Returns:
      best (float): best objective value found within max_time
    """

    # ----------------------- helpers -----------------------
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def evaluate(x):
        try:
            v = float(func(x))
        except Exception:
            return float("inf")
        if v != v:  # NaN
            return float("inf")
        return v

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # first primes for Halton bases (enough for common dims; fallback if larger)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def _next_prime(n):
        def is_prime(k):
            if k < 2: return False
            if k % 2 == 0: return k == 2
            r = int(math.isqrt(k))
            p = 3
            while p <= r:
                if k % p == 0:
                    return False
                p += 2
            return True
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        # index >= 1 recommended
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        # returns a point within bounds using Halton sequence + random Cranley-Patterson shift
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else _next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    # ----------------------- setup -----------------------
    start = time.time()
    deadline = start + max(0.0, float(max_time))
    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # Best-so-far
    best = float("inf")
    best_x = None

    # ----------------------- phase 1: global exploration -----------------------
    # Use Halton points; number tuned to be small but effective.
    # Also include a few pure random points to avoid pathological structure.
    shift = [random.random() for _ in range(dim)]
    global_budget = max(16, 12 * dim)

    k = 1
    for i in range(global_budget):
        if time.time() >= deadline:
            return best
        if (i % 5) == 0:
            x = rand_uniform_vec()
        else:
            x = halton_point(k, shift)
            k += 1
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        x = rand_uniform_vec()
        best = evaluate(x)
        best_x = x

    # ----------------------- phase 2: multi-start local refinement -----------------------
    # Local search parameters
    # Per-dimension sigma (step). Start moderately small; will adapt.
    def init_sigma():
        return [0.18 * s for s in span]

    min_sigma = [1e-12 * s for s in span]
    max_sigma = [0.6 * s for s in span]

    # multiple restarts seeded near best, plus occasional global jumps
    x = list(best_x)
    fx = float(best)
    sigma = init_sigma()

    # 1/5 success rule bookkeeping
    window = 30
    succ = 0
    trials = 0

    # stagnation controls
    last_best_time = time.time()
    stagnation_seconds = max(0.35, 0.12 * max_time)
    # probability of global jump per iteration (kept small)
    global_jump_prob = 0.01

    # Small helper: propose gaussian using Box-Muller
    def gaussian():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Coordinate pattern move: try a few coords with small step if cheap
    def coordinate_tweak(x0, f0, sig):
        # try up to a few coordinates each call
        xbest = x0
        fbest = f0
        # choose a small subset of dimensions to test
        m = 1 if dim == 1 else min(dim, 4)
        idxs = random.sample(range(dim), m)
        for j in idxs:
            lo, hi = bounds[j]
            step = 0.6 * sig[j] if sig[j] > 0 else 0.0
            if step <= 0:
                continue
            # test plus and minus deterministically
            xp = list(xbest); xp[j] = clamp(xp[j] + step, lo, hi)
            fp = evaluate(xp)
            if fp < fbest:
                xbest, fbest = xp, fp
                continue
            xm = list(xbest); xm[j] = clamp(xm[j] - step, lo, hi)
            fm = evaluate(xm)
            if fm < fbest:
                xbest, fbest = xm, fm
        return xbest, fbest

    while time.time() < deadline:
        now = time.time()

        # Restart / diversify if stagnating or at random
        if (now - last_best_time) > stagnation_seconds or (random.random() < global_jump_prob):
            # Half the time restart near best, half global
            if random.random() < 0.6 and best_x is not None:
                # near-best restart: best + noise
                xr = []
                for j in range(dim):
                    lo, hi = bounds[j]
                    z = gaussian()
                    xr.append(clamp(best_x[j] + z * (0.35 * span[j]), lo, hi))
                x = xr
            else:
                x = rand_uniform_vec()

            fx = evaluate(x)
            if fx < best:
                best = fx
                best_x = list(x)
                last_best_time = now

            sigma = init_sigma()
            succ = 0
            trials = 0

        # Occasionally do a coordinate tweak around current point (cheap exploit)
        if (trials % 12) == 0:
            xt, ft = coordinate_tweak(x, fx, sigma)
            if ft < fx:
                x, fx = xt, ft
                succ += 1
                if ft < best:
                    best, best_x = ft, list(xt)
                    last_best_time = time.time()

        # Mirrored mutation: sample one gaussian direction, test both + and - steps.
        # This often improves acceptance vs single-sided proposals.
        zvec = [gaussian() for _ in range(dim)]

        x1 = [0.0] * dim
        x2 = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            step = zvec[j] * sigma[j]
            x1[j] = clamp(x[j] + step, lo, hi)
            x2[j] = clamp(x[j] - step, lo, hi)

        f1 = evaluate(x1)
        f2 = evaluate(x2)

        trials += 1
        accepted = False

        if f1 <= f2 and f1 <= fx:
            x, fx = x1, f1
            accepted = True
        elif f2 < f1 and f2 <= fx:
            x, fx = x2, f2
            accepted = True

        if accepted:
            succ += 1
            if fx < best:
                best = fx
                best_x = list(x)
                last_best_time = time.time()

        # Adapt sigma every window with 1/5 rule
        if trials >= window:
            rate = succ / float(trials)
            # more decisive adaptation than before
            if rate > 0.22:
                factor = 1.35
            elif rate < 0.16:
                factor = 0.72
            else:
                factor = 1.0

            if factor != 1.0:
                for j in range(dim):
                    sj = sigma[j] * factor
                    # clamp
                    if sj < min_sigma[j]: sj = min_sigma[j]
                    if sj > max_sigma[j]: sj = max_sigma[j]
                    sigma[j] = sj

            succ = 0
            trials = 0

    return best
