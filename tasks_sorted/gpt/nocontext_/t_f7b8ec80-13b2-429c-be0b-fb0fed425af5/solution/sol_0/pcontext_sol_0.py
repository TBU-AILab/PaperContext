import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Quasi-random (Halton) sampling for global coverage
    - Adaptive local search (coordinate + gaussian steps)
    - Multiple restarts and step-size control
    Returns: best (float) = best objective value found
    """

    # ---- helpers ----
    def clamp(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi: 
            return hi
        return x

    def clamp_vec(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        return bounds[i][1] - bounds[i][0]

    # Halton sequence (bases: first primes)
    def primes_upto(n):
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

    def van_der_corput(index, base):
        # index >= 1
        vdc = 0.0
        denom = 1.0
        while index > 0:
            index, rem = divmod(index, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(k, bases):
        # k >= 1
        u = [van_der_corput(k, bases[i]) for i in range(dim)]
        return [bounds[i][0] + u[i] * (bounds[i][1] - bounds[i][0]) for i in range(dim)]

    # Normal via Box-Muller
    _have_spare = False
    _spare = 0.0
    def randn():
        nonlocal _have_spare, _spare
        if _have_spare:
            _have_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-12)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        _spare = z1
        _have_spare = True
        return z0

    # Evaluate safely; if func errors, treat as bad point
    def evaluate(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            # guard NaN/inf
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return float("inf")
            return float(v)
        except Exception:
            return float("inf")

    # ---- main algorithm ----
    start = time.time()
    deadline = start + float(max_time)

    # Initialize
    best = float("inf")
    best_x = None

    bases = primes_upto(dim)
    halton_k = 1

    # Search state
    # initial step sizes per dimension
    steps = [0.25 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]
    min_steps = [1e-12 * (span(i) if span(i) > 0 else 1.0) for i in range(dim)]

    # Parameters (kept simple and robust)
    stall_limit = 40            # iterations without improvement before restart/adapt
    local_iters_per_start = 200  # local tries per start (time-bounded anyway)
    coord_frac = 0.5            # probability to do coordinate move vs gaussian move

    # get initial candidate
    x = rand_uniform_vec()
    fx = evaluate(x)
    if fx < best:
        best, best_x = fx, x[:]

    stall = 0
    restarts = 0

    while time.time() < deadline:
        # Periodically restart from a global point (Halton) to avoid local traps
        if best_x is None or stall >= stall_limit:
            # Alternate between Halton and random restarts
            if restarts % 2 == 0:
                x = halton_point(halton_k, bases)
                halton_k += 1
            else:
                x = rand_uniform_vec()

            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x[:]

            # reset/adapt steps on restart: large then will shrink again
            steps = [max(steps[i] * 0.7, 0.15 * span(i)) for i in range(dim)]
            stall = 0
            restarts += 1

        improved_any = False

        # Local improvement loop (bounded by time)
        for _ in range(local_iters_per_start):
            if time.time() >= deadline:
                return best

            # Propose neighbor
            xn = x[:]

            if random.random() < coord_frac:
                # Coordinate step (one dimension)
                j = random.randrange(dim)
                s = steps[j]
                if s <= 0:
                    s = 1.0
                direction = -1.0 if random.random() < 0.5 else 1.0
                xn[j] = clamp(xn[j] + direction * s, bounds[j][0], bounds[j][1])
            else:
                # Gaussian step in all dimensions (scaled)
                for j in range(dim):
                    s = steps[j]
                    if s <= 0:
                        continue
                    xn[j] = clamp(xn[j] + randn() * 0.5 * s, bounds[j][0], bounds[j][1])

            fn = evaluate(xn)

            # Accept if improves current; also keep global best
            if fn < fx:
                x, fx = xn, fn
                improved_any = True
                stall = 0

                if fn < best:
                    best, best_x = fn, xn[:]
            else:
                stall += 1

            # Gentle step-size adaptation
            # If we improved, slightly expand; else shrink slowly
            if improved_any:
                for j in range(dim):
                    steps[j] = max(min_steps[j], steps[j] * 1.05)
            else:
                for j in range(dim):
                    steps[j] = max(min_steps[j], steps[j] * 0.98)

            # If steps are extremely small and no progress, force restart
            if stall >= stall_limit:
                break

    return best
