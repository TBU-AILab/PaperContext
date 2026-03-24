import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a simple, self-contained hybrid:
    - Sobol-like low-discrepancy sequence (Halton) for broad coverage (no numpy)
    - Local adaptive coordinate search around the current best
    - Occasional random restarts

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/array of length dim
    dim : int
        dimensionality
    bounds : list of (low, high)
        bounds per dimension
    max_time : int or float
        max runtime in seconds

    Returns
    -------
    best : float
        best (minimum) objective value found
    """

    # --------- helpers (no external libs) ----------
    def clamp(x):
        y = []
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if v < lo: v = lo
            if v > hi: v = hi
            y.append(v)
        return y

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def reflect_bounds(x):
        # Reflect step overshoots (often better than clamp for local moves)
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            v = y[i]
            if hi <= lo:
                y[i] = lo
                continue
            # reflect repeatedly if far out
            span = hi - lo
            while v < lo or v > hi:
                if v < lo:
                    v = lo + (lo - v)
                if v > hi:
                    v = hi - (v - hi)
            # numerical safety
            if v < lo: v = lo
            if v > hi: v = hi
            y[i] = v
        return y

    def halton_value(index, base):
        # radical inverse in base
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def first_primes(n):
        primes = []
        candidate = 2
        while len(primes) < n:
            is_p = True
            limit = int(math.isqrt(candidate))
            for p in primes:
                if p > limit:
                    break
                if candidate % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(candidate)
            candidate += 1
        return primes

    # --------- initialization ----------
    start = time.time()
    deadline = start + float(max_time)

    primes = first_primes(dim)
    # Start with a random point to avoid any pathological alignment
    best_x = rand_point()
    try:
        best = float(func(best_x))
    except Exception:
        # if func fails on some points, keep searching
        best = float("inf")

    # step sizes as fraction of range
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_steps = [0.25 * (r if r > 0 else 1.0) for r in ranges]
    steps = base_steps[:]

    # bookkeeping
    halton_index = 1
    it = 0

    # --------- main loop ----------
    while time.time() < deadline:
        it += 1
        now = time.time()
        remaining = deadline - now

        # Phase selection: more global early, more local later
        # Use remaining time to bias behavior
        frac_left = remaining / (float(max_time) + 1e-12)

        # Occasionally do a (quasi) global sample
        do_global = (it % 7 == 0) or (frac_left > 0.6 and it % 2 == 0) or (random.random() < 0.05)

        if do_global:
            # Halton sample mapped to bounds
            u = [halton_value(halton_index, primes[i]) for i in range(dim)]
            halton_index += 1
            x = [bounds[i][0] + u[i] * (bounds[i][1] - bounds[i][0]) for i in range(dim)]

            # small jitter to avoid strict determinism / function discontinuities
            for i in range(dim):
                span = ranges[i] if ranges[i] > 0 else 1.0
                x[i] += (random.random() - 0.5) * 0.01 * span
            x = reflect_bounds(x)

            try:
                fx = float(func(x))
                if fx < best:
                    best = fx
                    best_x = x
                    # reset steps when improvement found
                    steps = [max(1e-12, s * 0.9) for s in steps]
            except Exception:
                pass
            continue

        # Local search: adaptive coordinate exploration around best_x
        i = random.randrange(dim)
        step = steps[i]

        # If step becomes too small, restart steps or do random restart
        span_i = ranges[i] if ranges[i] > 0 else 1.0
        min_step = 1e-9 * span_i + 1e-12

        if step < min_step:
            # random restart around the best or global restart
            if random.random() < 0.5:
                # broaden search around best
                steps = [max(bs, 0.05 * (r if r > 0 else 1.0)) for bs, r in zip(base_steps, ranges)]
            else:
                # full restart
                x = rand_point()
                try:
                    fx = float(func(x))
                    if fx < best:
                        best = fx
                        best_x = x
                        steps = base_steps[:]
                except Exception:
                    pass
            continue

        # Try plus/minus moves on one coordinate, and accept best improvement
        x_plus = best_x[:]
        x_minus = best_x[:]
        x_plus[i] += step
        x_minus[i] -= step
        x_plus = reflect_bounds(x_plus)
        x_minus = reflect_bounds(x_minus)

        improved = False
        best_candidate_x = None
        best_candidate_f = None

        try:
            f_plus = float(func(x_plus))
            best_candidate_x, best_candidate_f = x_plus, f_plus
        except Exception:
            pass

        try:
            f_minus = float(func(x_minus))
            if best_candidate_f is None or f_minus < best_candidate_f:
                best_candidate_x, best_candidate_f = x_minus, f_minus
        except Exception:
            pass

        if best_candidate_f is not None and best_candidate_f < best:
            best = best_candidate_f
            best_x = best_candidate_x
            improved = True

        # Adapt step size: shrink on failure, slightly expand on success
        if improved:
            steps[i] = min(0.5 * span_i + 1e-12, steps[i] * 1.05)
        else:
            steps[i] = steps[i] * 0.7

        # Rare random perturbation (helps escape shallow local minima)
        if random.random() < 0.02:
            x = best_x[:]
            for j in range(dim):
                span = ranges[j] if ranges[j] > 0 else 1.0
                x[j] += random.gauss(0.0, 0.05 * span * frac_left)
            x = reflect_bounds(x)
            try:
                fx = float(func(x))
                if fx < best:
                    best = fx
                    best_x = x
            except Exception:
                pass

    # return fitness of the best found solution
    return best
