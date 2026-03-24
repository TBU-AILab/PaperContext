import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: hybrid of
      - Latin-hypercube-like seeding (stratified random per-dimension),
      - local coordinate search (pattern search),
      - adaptive step-size reduction,
      - occasional random restarts.

    Returns:
      best (float): best (minimum) function value found within max_time seconds.

    Notes:
      - No external libraries required.
      - Assumes func(x) accepts a list (or sequence) of length dim and returns a float.
      - Enforces bounds by clipping.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        # func may throw for some inputs; treat as very bad.
        try:
            v = func(x)
            # Handle NaN/inf robustly.
            if v is None:
                return float("inf")
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    return float("inf")
                return float(v)
            # If func returns something non-numeric, treat as bad.
            return float("inf")
        except Exception:
            return float("inf")

    # --- configuration (kept simple, robust) ---
    # Initial step sizes as a fraction of span; not too tiny.
    init_step_frac = 0.2
    min_step_frac = 1e-8
    step_shrink = 0.5

    # Seeding: number of initial stratified samples (bounded by time anyway).
    seed_points = max(8, 4 * dim)

    # Random restart probability during local search
    restart_prob = 0.02

    # --- initialization: stratified seeding per dimension ---
    # We approximate Latin hypercube by permuting strata independently per dimension.
    best_x = None
    best = float("inf")

    # If max_time is extremely small, still try at least one evaluation.
    if time.time() >= deadline:
        return best

    strata = seed_points
    perms = []
    for i in range(dim):
        p = list(range(strata))
        random.shuffle(p)
        perms.append(p)

    for k in range(strata):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            # sample uniformly inside stratum
            a = perms[i][k] / float(strata)
            b = (perms[i][k] + 1) / float(strata)
            u = a + (b - a) * random.random()
            x[i] = lows[i] + u * spans[i]
        fx = safe_eval(x)
        if fx < best:
            best = fx
            best_x = x[:]

    # Fallback if somehow no evaluation improved (shouldn't happen)
    if best_x is None:
        best_x = rand_vec()
        best = safe_eval(best_x)

    # --- main loop: coordinate/pattern local search with adaptive steps + restarts ---
    step = [max(spans[i] * init_step_frac, 1e-12) for i in range(dim)]
    min_step = [max(spans[i] * min_step_frac, 1e-15) for i in range(dim)]

    x = best_x[:]
    fx = best

    while time.time() < deadline:
        # occasional restart to escape local minima / plateaus
        if random.random() < restart_prob:
            x = rand_vec()
            fx = safe_eval(x)
            if fx < best:
                best, best_x = fx, x[:]
            # reset steps a bit on restart
            step = [max(spans[i] * init_step_frac, step[i]) for i in range(dim)]

        improved = False

        # Randomized coordinate order helps avoid cycles
        coords = list(range(dim))
        random.shuffle(coords)

        for i in coords:
            if time.time() >= deadline:
                return best

            if step[i] <= min_step[i]:
                continue

            # Try plus and minus moves along coordinate i
            xi = x[i]

            # plus
            xp = x[:]
            xp[i] = xi + step[i]
            clip_vec(xp)
            fp = safe_eval(xp)

            # minus
            xm = x[:]
            xm[i] = xi - step[i]
            clip_vec(xm)
            fm = safe_eval(xm)

            # Greedy accept the best among current/plus/minus
            if fp < fx or fm < fx:
                if fp <= fm:
                    x, fx = xp, fp
                else:
                    x, fx = xm, fm
                improved = True

                if fx < best:
                    best, best_x = fx, x[:]

        if not improved:
            # reduce step sizes when no coordinate move helps
            all_tiny = True
            for i in range(dim):
                step[i] *= step_shrink
                if step[i] > min_step[i]:
                    all_tiny = False

            # If steps are tiny, do a mild "shake" around best_x and reset steps
            if all_tiny:
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                # small random perturbation
                for i in range(dim):
                    jitter = (random.random() * 2.0 - 1.0) * (spans[i] * 0.01)
                    x[i] += jitter
                clip_vec(x)
                fx = safe_eval(x)
                if fx < best:
                    best, best_x = fx, x[:]
                # reset steps to moderate size to continue searching
                step = [max(spans[i] * init_step_frac * 0.5, min_step[i]) for i in range(dim)]

    return best
