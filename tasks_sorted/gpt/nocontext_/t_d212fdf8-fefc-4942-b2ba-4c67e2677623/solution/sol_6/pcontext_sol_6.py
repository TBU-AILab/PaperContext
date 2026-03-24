import random
import time
import math


def run(func, dim, bounds, max_time):
    # Self-contained bounded optimizer: Adaptive coordinate pattern search + random restarts
    # - Works without numpy/scipy
    # - Uses only func(params)->float, dim, bounds, max_time
    # - Returns best (minimum) fitness found within max_time

    t0 = time.time()
    deadline = t0 + max_time

    # Helpers
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def clip_vec(x):
        return [clamp(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        lo, hi = bounds[i]
        return hi - lo

    def safe_eval(x):
        # Make evaluation robust to occasional func failures
        try:
            v = func(x)
            # If returns NaN/inf, treat as bad
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    return float("inf")
                return float(v)
            # If func returns something odd, coerce if possible
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Initial best: a few random samples
    best = float("inf")
    best_x = None

    # Budget split implicitly by time; keep overhead small
    init_samples = max(5, 2 * dim)
    for _ in range(init_samples):
        if time.time() >= deadline:
            return best
        x = rand_vec()
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # If everything failed, still return inf
        return best

    # Main optimization loop
    # Step sizes start as a fraction of bounds and shrink when no improvement.
    base_steps = [0.2 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]
    min_steps = [1e-12 * (span(i) if span(i) > 0 else 1.0) for i in range(dim)]

    # Parameters
    shrink = 0.5          # step reduction factor
    expand = 1.2          # mild expansion when improving
    stall_restart = 60    # iterations without improvement before restart (time-based loop, so just a counter)
    max_restarts = 10**9  # effectively unlimited; controlled by time

    steps = base_steps[:]
    no_improve = 0
    restarts = 0

    while time.time() < deadline and restarts < max_restarts:
        improved_any = False

        # Coordinate-wise pattern search: try +/- step on each dimension
        for i in range(dim):
            if time.time() >= deadline:
                return best

            if steps[i] <= min_steps[i]:
                continue

            xi = best_x[i]

            # Try plus
            cand = best_x[:]
            cand[i] = clamp(xi + steps[i], bounds[i][0], bounds[i][1])
            f1 = safe_eval(cand)
            if f1 < best:
                best, best_x = f1, cand
                improved_any = True
                steps[i] *= expand
                continue

            # Try minus
            cand = best_x[:]
            cand[i] = clamp(xi - steps[i], bounds[i][0], bounds[i][1])
            f2 = safe_eval(cand)
            if f2 < best:
                best, best_x = f2, cand
                improved_any = True
                steps[i] *= expand
                continue

            # No improvement on this coordinate: optionally shrink a bit
            steps[i] *= shrink

        if improved_any:
            no_improve = 0
        else:
            no_improve += 1

        # If stalled or steps are tiny, do a randomized restart near the best and globally
        tiny_steps = all(steps[i] <= min_steps[i] * 1000 for i in range(dim))
        if no_improve >= stall_restart or tiny_steps:
            if time.time() >= deadline:
                return best

            restarts += 1
            no_improve = 0

            # Mix: local perturbation around best + occasional global random
            if random.random() < 0.7:
                # Local: gaussian-like perturbation using sum of uniforms (no external libs)
                x = best_x[:]
                for i in range(dim):
                    s = span(i)
                    if s <= 0:
                        continue
                    # Approx gaussian in [-1,1] scaled by 0.1*range
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random()) / 6.0
                    g = (g - 0.5) * 2.0  # roughly in [-1,1]
                    x[i] = clamp(x[i] + g * 0.1 * s, bounds[i][0], bounds[i][1])
            else:
                # Global
                x = rand_vec()

            fx = safe_eval(x)
            if fx < best:
                best, best_x = fx, x
                steps = base_steps[:]
            else:
                # Even if not better, reset steps sometimes to escape
                steps = [max(min_steps[i] * 1e6, base_steps[i] * 0.5) for i in range(dim)]

    return best
