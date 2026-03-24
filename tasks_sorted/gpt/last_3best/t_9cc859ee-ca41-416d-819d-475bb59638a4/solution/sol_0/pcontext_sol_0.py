import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization
    - Local coordinate search (pattern search)
    - Occasional random restarts
    - Adaptive step sizes per dimension

    Args:
        func: callable(params_list)->float
        dim: int
        bounds: list of (low, high) pairs, length dim
        max_time: seconds (int/float)

    Returns:
        best fitness (float) found within time
    """

    # ---------- helpers ----------
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clip(x, i):
        if x < lo[i]:
            return lo[i]
        if x > hi[i]:
            return hi[i]
        return x

    def eval_f(x):
        # func expects an array-like; we pass a plain list for compatibility.
        return float(func(list(x)))

    def random_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    def lhs_points(n):
        # Simple per-dimension stratification with independent permutations
        # (not full LHS correlation handling, but works well and is cheap).
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        pts = []
        for i in range(n):
            x = []
            for j in range(dim):
                # sample uniformly within stratum
                u = (perms[j][i] + random.random()) / n
                x.append(lo[j] + u * span[j])
            pts.append(x)
        return pts

    # ---------- time control ----------
    t_end = time.time() + float(max_time)

    # ---------- initialization ----------
    best_x = None
    best = float("inf")

    # Initial batch size: small but grows with dimension (kept modest for time).
    n0 = max(8, min(40, 8 + 2 * dim))
    for x in lhs_points(n0):
        if time.time() >= t_end:
            return best
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # Extreme edge case: no time to evaluate
        return best

    # ---------- optimization loop ----------
    # Step sizes start as a fraction of range per dimension.
    step = [max(1e-12, 0.25 * span[i]) for i in range(dim)]
    min_step = [max(1e-12, 1e-9 * span[i] if span[i] > 0 else 1e-12) for i in range(dim)]

    # Restart control
    no_improve_iters = 0
    restart_threshold = 40 + 5 * dim

    # Coordinate ordering (shuffled each sweep)
    dims = list(range(dim))

    while time.time() < t_end:
        improved_any = False
        random.shuffle(dims)

        # One coordinate-search sweep around current best
        for j in dims:
            if time.time() >= t_end:
                return best

            x0 = best_x
            f0 = best

            # Try +/- step on this coordinate
            s = step[j]
            if s <= min_step[j]:
                continue

            candidates = []

            xp = list(x0)
            xp[j] = clip(xp[j] + s, j)
            if xp[j] != x0[j]:
                candidates.append(xp)

            xm = list(x0)
            xm[j] = clip(xm[j] - s, j)
            if xm[j] != x0[j]:
                candidates.append(xm)

            # Evaluate candidates (best-first not known, just check both)
            for xc in candidates:
                if time.time() >= t_end:
                    return best
                fc = eval_f(xc)
                if fc < best:
                    best = fc
                    best_x = xc
                    improved_any = True

        if improved_any:
            no_improve_iters = 0
            # Slightly expand steps after improvements to move faster
            for j in range(dim):
                step[j] = min(span[j], step[j] * 1.2) if span[j] > 0 else step[j]
        else:
            no_improve_iters += 1
            # Contract steps when stuck
            for j in range(dim):
                step[j] *= 0.5

        # Random restart / perturbation if stuck too long or steps too small
        if no_improve_iters >= restart_threshold or all(step[j] <= min_step[j] for j in range(dim)):
            no_improve_iters = 0

            # With some probability, do a local "shake" near best; otherwise full restart
            if random.random() < 0.6:
                x = list(best_x)
                for j in range(dim):
                    # shake proportional to range but smaller than initial step
                    amp = 0.1 * span[j]
                    if amp > 0:
                        x[j] = clip(x[j] + (random.random() * 2 - 1) * amp, j)
            else:
                x = random_point()

            if time.time() >= t_end:
                return best
            fx = eval_f(x)
            if fx < best:
                best = fx
                best_x = x
                # Reset steps after a good restart find
                step = [max(1e-12, 0.25 * span[i]) for i in range(dim)]
            else:
                # If restart didn't help, still reset steps moderately
                step = [max(min_step[i], 0.2 * span[i]) for i in range(dim)]

    return best
