import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using: Sobol-ish LHS seeding + adaptive local search
    (coordinate steps with decaying step size) + occasional random restarts.
    No external libraries required.

    Args:
        func: callable(params:list[float]) -> float
        dim: int
        bounds: list of (low, high) for each dimension
        max_time: seconds (int/float)

    Returns:
        best fitness (float) found within the time limit
    """
    t0 = time.time()

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        lo, hi = lows[i], highs[i]
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # defend against NaN/inf/exception: treat as very bad
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

    # Latin Hypercube sampling for initial coverage
    def lhs_points(n):
        # For each dimension, create n strata and permute them
        per_dim = []
        for i in range(dim):
            idx = list(range(n))
            random.shuffle(idx)
            per_dim.append(idx)

        pts = []
        for k in range(n):
            x = []
            for i in range(dim):
                # sample uniformly within stratum
                u = (per_dim[i][k] + random.random()) / n
                x.append(lows[i] + u * spans[i])
            pts.append(x)
        return pts

    # --- initialization ---
    best_x = None
    best = float("inf")

    # choose a small, fast initial design
    init_n = max(10, min(50, 10 * dim))
    for x in lhs_points(init_n):
        if time.time() - t0 >= max_time:
            return best
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        best_x = rand_point()
        best = eval_f(best_x)

    # --- main search parameters ---
    # initial step sizes as a fraction of range
    base_steps = [0.25 * s if s > 0 else 1.0 for s in spans]
    min_steps = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # local search state
    x = best_x[:]
    fx = best
    steps = base_steps[:]

    # stagnation counters and restart control
    no_improve_iters = 0
    last_improve_time = time.time()
    restart_wait = max(0.25, 0.05 * max_time)  # if no improvement for this long, restart

    # --- optimization loop ---
    while True:
        if time.time() - t0 >= max_time:
            return best

        improved = False

        # Try coordinate-wise moves (both directions), greedy acceptance
        # Randomize coordinate order to avoid bias
        order = list(range(dim))
        random.shuffle(order)

        for i in order:
            if time.time() - t0 >= max_time:
                return best

            si = steps[i]
            if si < min_steps[i]:
                continue

            # propose two candidates
            x1 = x[:]
            x1[i] = clamp(x1[i] + si, i)
            f1 = eval_f(x1)

            x2 = x[:]
            x2[i] = clamp(x2[i] - si, i)
            f2 = eval_f(x2)

            # pick best move
            if f1 < fx or f2 < fx:
                if f1 <= f2:
                    x, fx = x1, f1
                else:
                    x, fx = x2, f2
                improved = True

                # track global best
                if fx < best:
                    best = fx
                    best_x = x[:]
                    last_improve_time = time.time()
            else:
                # small probability to accept a worse move early (mild hill-climbing escape)
                # temperature tied to remaining time
                remaining = max(1e-9, max_time - (time.time() - t0))
                T = 0.1 * (remaining / max_time)  # decays to 0
                if T > 0:
                    # choose one random direction candidate
                    cand, fc = (x1, f1) if random.random() < 0.5 else (x2, f2)
                    delta = fc - fx
                    if delta > 0 and random.random() < math.exp(-delta / max(T, 1e-12)):
                        x, fx = cand, fc  # accept worse

        if improved:
            no_improve_iters = 0
            # slightly increase steps on success (keeps momentum)
            for i in range(dim):
                steps[i] = min(steps[i] * 1.05, spans[i] if spans[i] > 0 else steps[i])
        else:
            no_improve_iters += 1
            # reduce step sizes when stuck
            for i in range(dim):
                steps[i] *= 0.5

        # Restart logic: if steps are tiny or no improvement for a while, restart near best or random
        steps_tiny = all(steps[i] < (1e-9 * (spans[i] if spans[i] > 0 else 1.0) + min_steps[i]) for i in range(dim))
        no_improve_long = (time.time() - last_improve_time) > restart_wait

        if steps_tiny or (no_improve_iters > 10 and no_improve_long):
            # restart: sample a point either globally or in a neighborhood around best_x
            if best_x is None:
                x = rand_point()
            else:
                if random.random() < 0.7:
                    # local perturbation around best
                    x = best_x[:]
                    for i in range(dim):
                        # perturb scale decreases with time (more exploitation later)
                        frac = 0.2 * max(0.05, (max_time - (time.time() - t0)) / max_time)
                        r = (random.random() * 2.0 - 1.0) * frac * spans[i]
                        x[i] = clamp(x[i] + r, i)
                else:
                    # global restart
                    x = rand_point()

            fx = eval_f(x)
            if fx < best:
                best = fx
                best_x = x[:]
                last_improve_time = time.time()

            # reset step sizes for new local search
            steps = base_steps[:]
            no_improve_iters = 0
