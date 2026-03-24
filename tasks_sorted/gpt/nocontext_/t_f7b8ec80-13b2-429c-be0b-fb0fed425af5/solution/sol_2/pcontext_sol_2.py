import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      1) Latin-hypercube-like stratified initialization
      2) Local coordinate search (pattern search)
      3) Occasional random restarts and shrinking step sizes

    Returns:
      best (float): fitness of the best found solution
    """
    t0 = time.time()

    def time_left():
        return max_time - (time.time() - t0)

    # --- utilities ---
    lows = [bounds[i][0] for i in range(dim)]
    highs = [bounds[i][1] for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        # Be robust to exceptions / NaN / inf from func
        try:
            v = func(x)
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    return float("inf")
                return float(v)
            # If func returns something unexpected, treat as bad.
            return float("inf")
        except Exception:
            return float("inf")

    # --- initial design: stratified sampling per dimension (LHS-style) ---
    # Keep it small to respect time, but enough to seed a good region.
    # If max_time is very small, it will still gracefully return best found so far.
    init_n = max(8, 4 * dim)
    init_n = min(init_n, 60)  # cap

    # Build permutations of bins for each dimension
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    best = float("inf")
    best_x = None

    # Evaluate initial points
    for k in range(init_n):
        if time_left() <= 0:
            return best
        x = []
        for i in range(dim):
            # sample within the bin [p/init_n, (p+1)/init_n]
            b = perms[i][k]
            u = (b + random.random()) / init_n
            x.append(lows[i] + u * spans[i])
        v = safe_eval(x)
        if v < best:
            best, best_x = v, x

    if best_x is None:
        # Fallback if everything failed
        best_x = rand_point()
        best = safe_eval(best_x)

    # --- local search parameters ---
    # Start with a step size ~ 20% of span (per-dimension), then adapt.
    step = [0.2 * s if s > 0 else 1.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # Stagnation/restart control
    no_improve_iters = 0
    restart_after = 25 + 5 * dim

    # Coordinate order shuffling to reduce bias
    coords = list(range(dim))

    # --- main loop ---
    while time_left() > 0:
        improved = False
        random.shuffle(coords)

        # Pattern/coordinate search around current best
        for i in coords:
            if time_left() <= 0:
                return best

            xi = best_x[i]

            # try +step and -step
            for direction in (+1.0, -1.0):
                if time_left() <= 0:
                    return best

                cand = list(best_x)
                cand[i] = xi + direction * step[i]
                clamp(cand)
                v = safe_eval(cand)

                if v < best:
                    best, best_x = v, cand
                    improved = True
                    break  # move to next coordinate from the new point

            # If improved on this coordinate, continue with next coordinate
            # otherwise, keep trying others.

        if improved:
            no_improve_iters = 0
            # slightly increase steps when improving (capped by span)
            for i in range(dim):
                if spans[i] > 0:
                    step[i] = min(step[i] * 1.15, spans[i])
            continue

        # No improvement: shrink steps
        for i in range(dim):
            step[i] *= 0.5

        no_improve_iters += 1

        # If steps are tiny or we stagnated: restart near best or globally
        tiny = all(step[i] <= min_step[i] for i in range(dim))
        if tiny or no_improve_iters >= restart_after:
            no_improve_iters = 0

            # With some probability do a global restart, else local jitter
            if random.random() < 0.35:
                x = rand_point()
            else:
                # local jitter around best_x scaled by current span
                x = []
                for i in range(dim):
                    # jitter up to 10% of span
                    r = (random.random() * 2.0 - 1.0) * (0.10 * spans[i] if spans[i] > 0 else 1.0)
                    x.append(best_x[i] + r)
                clamp(x)

            v = safe_eval(x)
            if v < best:
                best, best_x = v, x

            # reset step sizes after restart
            for i in range(dim):
                step[i] = 0.2 * spans[i] if spans[i] > 0 else 1.0

    return best
