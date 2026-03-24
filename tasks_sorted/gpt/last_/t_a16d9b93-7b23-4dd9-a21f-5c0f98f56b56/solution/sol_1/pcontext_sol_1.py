import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Key upgrades vs previous:
      - Better global exploration: low-discrepancy (Halton) initial sampling
      - Stronger local search: multi-start Nelder–Mead (derivative-free simplex)
      - Robust bounded handling via reflection + clipping
      - Occasional adaptive coordinate/pattern search around incumbent
      - Time-aware scheduling (keeps returning best-so-far)

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ------------------------- helpers -------------------------
    def is_finite(v):
        return not (math.isnan(v) or math.isinf(v))

    def clip(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if v < lo: v = lo
            elif v > hi: v = hi
            y[i] = v
        return y

    def reflect_into_bounds(x):
        # Reflect to handle out-of-bounds more smoothly than hard clipping
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if hi <= lo:
                y[i] = lo
                continue
            span = hi - lo
            # map to [0, 2*span) then reflect
            v = (v - lo) % (2.0 * span)
            if v > span:
                v = 2.0 * span - v
            y[i] = lo + v
        return y

    def evaluate(x):
        try:
            v = float(func(x))
            if not is_finite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Halton sequence for quasi-random sampling (no external libs)
    def _primes(n):
        ps = []
        k = 2
        while len(ps) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in ps:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(k)
            k += 1
        return ps

    primes = _primes(max(1, dim))

    def halton(idx, base):
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_vec(k):
        x = [0.0] * dim
        for i in range(dim):
            u = halton(k, primes[i])
            lo, hi = bounds[i]
            x[i] = lo + u * (hi - lo)
        return x

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def avg_span():
        s = 0.0
        for i in range(dim):
            lo, hi = bounds[i]
            s += (hi - lo)
        return s / max(1, dim)

    # coordinate/pattern search step
    def pattern_refine(x, fx, step):
        # Try +/- along each coordinate; accept first improvement (greedy)
        bestx, bestf = x, fx
        for i in range(dim):
            if time.time() >= deadline:
                return bestx, bestf
            for sign in (-1.0, 1.0):
                y = bestx[:]  # use incumbent each time (first-improvement)
                y[i] += sign * step
                y = reflect_into_bounds(y)
                fy = evaluate(y)
                if fy < bestf:
                    bestx, bestf = y, fy
                    break
        return bestx, bestf

    # ------------------------- global initialization -------------------------
    best = float("inf")
    best_x = None

    # Spend a small but meaningful fraction of time/budget on global probing
    # Use Halton + random to avoid worst-cases.
    init_points = max(40, 15 * dim)
    k = 1
    for j in range(init_points):
        if time.time() >= deadline:
            return best
        if j % 3 == 0:
            x = rand_vec()
        else:
            x = halton_vec(k)
            k += 1
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        best_x = rand_vec()
        best = evaluate(best_x)

    # ------------------------- Nelder–Mead local search -------------------------
    # Standard coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho   = 0.5   # contraction
    sigma_shrink = 0.5  # shrink

    # Build an initial simplex around x0 with scale relative to bounds
    def make_simplex(x0, scale):
        simplex = [x0[:]]
        for i in range(dim):
            v = x0[:]
            v[i] += scale
            v = reflect_into_bounds(v)
            simplex.append(v)
        return simplex

    def nm_optimize(x0, f0, scale, max_evals):
        # returns (bestx, bestf, evals_used)
        simplex = make_simplex(x0, scale)
        fs = [evaluate(p) for p in simplex]
        evals = len(fs)

        # If x0 evaluation provided, use it (avoid one call)
        fs[0] = f0

        while evals < max_evals and time.time() < deadline:
            # order
            idx = list(range(dim + 1))
            idx.sort(key=lambda i: fs[i])
            simplex = [simplex[i] for i in idx]
            fs = [fs[i] for i in idx]

            bestp, bestf = simplex[0], fs[0]
            worstp, worstf = simplex[-1], fs[-1]
            second_worstf = fs[-2]

            # centroid of all but worst
            centroid = [0.0] * dim
            for p in simplex[:-1]:
                for d in range(dim):
                    centroid[d] += p[d]
            inv = 1.0 / dim
            for d in range(dim):
                centroid[d] *= inv

            def comb(a, pa, b, pb):
                # a*pa + b*pb
                return [a * pa[d] + b * pb[d] for d in range(dim)]

            # reflection
            xr = comb(1.0 + alpha, centroid, -alpha, worstp)
            xr = reflect_into_bounds(xr)
            fr = evaluate(xr); evals += 1
            if fr < bestf:
                # expansion
                xe = comb(1.0 + gamma, centroid, -gamma, worstp)
                xe = reflect_into_bounds(xe)
                fe = evaluate(xe); evals += 1
                if fe < fr:
                    simplex[-1], fs[-1] = xe, fe
                else:
                    simplex[-1], fs[-1] = xr, fr
            elif fr < second_worstf:
                simplex[-1], fs[-1] = xr, fr
            else:
                # contraction
                if fr < worstf:
                    # outside contraction
                    xc = comb(1.0 + rho, centroid, -rho, worstp)
                else:
                    # inside contraction
                    xc = comb(1.0 - rho, centroid, rho, worstp)
                xc = reflect_into_bounds(xc)
                fc = evaluate(xc); evals += 1
                if fc < worstf:
                    simplex[-1], fs[-1] = xc, fc
                else:
                    # shrink
                    b = simplex[0]
                    for i in range(1, dim + 1):
                        simplex[i] = reflect_into_bounds(comb(1.0 - sigma_shrink, b, sigma_shrink, simplex[i]))
                        fs[i] = evaluate(simplex[i])
                    evals += dim

            # Optional early exit if simplex is tiny in objective
            # (keeps robustness; avoids wasting time on flat noise)
            fspread = fs[-1] - fs[0]
            if fspread < 1e-12:
                break

        # return best in simplex
        bi = min(range(dim + 1), key=lambda i: fs[i])
        return simplex[bi], fs[bi], evals

    # ------------------------- multistart schedule -------------------------
    # Alternate: NM local search + pattern refine + occasional random restart
    base = avg_span()
    # initial NM scale: a fraction of domain
    init_scale = 0.08 * base if base > 0 else 1.0

    # Manage restarts around best and from new global points
    halton_k = k
    last_improve_time = time.time()
    attempts = 0

    while time.time() < deadline:
        attempts += 1

        # Choose a start point:
        #  - often the current best with a small jitter
        #  - sometimes a new Halton point (global restart)
        if attempts % 4 == 0:
            x0 = halton_vec(halton_k); halton_k += 1
            f0 = evaluate(x0)
        else:
            # jitter best
            x0 = best_x[:]
            # decreasing jitter when improving recently; otherwise larger
            dt = max(0.0, time.time() - last_improve_time)
            jitter_scale = init_scale * (0.5 if dt < 0.3 * max_time else 1.3)
            for i in range(dim):
                # gaussian via Box-Muller
                u1 = max(1e-12, random.random())
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                x0[i] += z * jitter_scale
            x0 = reflect_into_bounds(x0)
            f0 = evaluate(x0)

        if f0 < best:
            best, best_x = f0, x0
            last_improve_time = time.time()

        # Decide NM evaluation budget per attempt (time-aware)
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        # keep per-call bounded; scale with dimension
        nm_budget = max(40 + 20 * dim, 120)
        # if little time left, reduce
        if remaining < 0.15 * max_time:
            nm_budget = max(30 + 10 * dim, 60)

        # Run NM with a scale adapted to how long since improvement
        dt = max(0.0, time.time() - last_improve_time)
        scale = init_scale * (0.6 if dt < 0.25 * max_time else 1.4)

        x1, f1, _ = nm_optimize(x0, f0, scale, nm_budget)
        if f1 < best:
            best, best_x = f1, x1
            last_improve_time = time.time()

        # Quick pattern refinement around incumbent (cheap)
        if time.time() < deadline:
            step = 0.02 * base if base > 0 else 0.1
            # adapt step: smaller when recently improving
            if (time.time() - last_improve_time) < 0.2 * max_time:
                step *= 0.5
            x2, f2 = pattern_refine(best_x, best, step)
            if f2 < best:
                best, best_x = f2, x2
                last_improve_time = time.time()

    return best
