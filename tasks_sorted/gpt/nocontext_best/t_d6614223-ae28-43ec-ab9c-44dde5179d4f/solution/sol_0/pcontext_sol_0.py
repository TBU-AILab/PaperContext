import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained adaptive pattern search:
    - Global exploration via Latin-hypercube-like stratified sampling (no numpy).
    - Local exploitation via coordinate/pattern search with adaptive step sizes.
    - Occasional random restarts + shrinking steps to refine.
    Returns: best (float) = minimum fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lows = [bounds[i][0] for i in range(dim)]
    highs = [bounds[i][1] for i in range(dim)]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def now():
        return time.time()

    def clip_vec(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; list is acceptable per template.
        return float(func(x))

    # Latin-hypercube-like stratified sample (1 point per "bin" per dimension)
    def lhs_sample(n):
        # For each dimension, create a random permutation of bins [0..n-1]
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)

        pts = []
        for i in range(n):
            x = []
            for d in range(dim):
                # pick uniformly inside bin
                u = (perms[d][i] + random.random()) / n
                x.append(lows[d] + u * spans[d])
            pts.append(x)
        return pts

    # --- initialization ---
    best = float("inf")
    best_x = None

    # initial step scale per dimension
    # start with 10% of range, but not zero
    step0 = [0.1 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Choose initial budget for stratified exploration
    # Keep it small and time-aware.
    n0 = max(4, int(8 + 2 * math.sqrt(dim)))
    # Try a few exploratory points quickly
    for x in lhs_sample(n0):
        if now() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # fallback
        best_x = rand_vec()
        best = eval_f(best_x)
        if now() >= deadline:
            return best

    # --- main loop: adaptive local search with random restarts ---
    # state for current search point
    x = best_x[:]
    fx = best
    step = step0[:]

    # restart control
    last_improve_time = now()
    stall_seconds = max(0.05, 0.15 * max_time)  # if no improvement for this long -> restart

    # pattern search parameters
    shrink = 0.65
    grow = 1.15
    max_grow = 0.5  # cap step to 50% of span

    while now() < deadline:
        improved = False

        # Coordinate search around x
        for d in range(dim):
            if now() >= deadline:
                return best

            sd = step[d]
            if sd <= min_step[d]:
                continue

            # try + direction
            xp = x[:]
            xp[d] += sd
            xp = clip_vec(xp)
            fp = eval_f(xp)
            if fp < fx:
                x, fx = xp, fp
                improved = True
                continue

            # try - direction
            xm = x[:]
            xm[d] -= sd
            xm = clip_vec(xm)
            fm = eval_f(xm)
            if fm < fx:
                x, fx = xm, fm
                improved = True
                continue

        # Update best
        if fx < best:
            best = fx
            best_x = x[:]
            last_improve_time = now()

        # Adapt steps
        if improved:
            # slightly increase steps where possible to move faster
            for d in range(dim):
                cap = max_grow * spans[d] if spans[d] > 0 else step[d]
                step[d] = min(cap if cap > 0 else step[d], step[d] * grow)
        else:
            # no improvement: shrink steps
            for d in range(dim):
                step[d] = max(min_step[d], step[d] * shrink)

        # If steps are tiny or stalled too long, restart near best or globally
        tiny = True
        for d in range(dim):
            if step[d] > 10 * min_step[d]:
                tiny = False
                break

        if tiny or (now() - last_improve_time) > stall_seconds:
            # Mix: local restart near best with gaussian-like jitter + occasional global restart
            if random.random() < 0.25:
                x = rand_vec()
            else:
                # jitter around best_x
                x = best_x[:]
                for d in range(dim):
                    # jitter scale based on current step and span
                    s = max(step[d], 0.02 * spans[d])
                    # triangular noise (no external libs) centered at 0
                    noise = (random.random() - random.random()) * s
                    x[d] = x[d] + noise
                x = clip_vec(x)

            fx = eval_f(x)

            # reset steps moderately
            step = [max(min_step[i], 0.1 * spans[i] if spans[i] > 0 else step0[i]) for i in range(dim)]
            last_improve_time = now()

            if fx < best:
                best = fx
                best_x = x[:]

    return best
