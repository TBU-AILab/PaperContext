import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: hybrid of
      - Latin-hypercube-like stratified initialization
      - Local coordinate search (pattern search)
      - Occasional adaptive random restarts
    Returns: best (float) = best objective value found.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- helpers ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    span = [highs[i] - lows[i] for i in range(dim)]

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_uniform_vec():
        return [lows[i] + random.random() * span[i] for i in range(dim)]

    def lhs_points(n):
        """
        Lightweight LHS-like sampler (no numpy).
        For each dimension, use n stratified bins with random jitter,
        then permute independently per dimension.
        """
        # create per-dimension stratified coordinates
        coords_per_dim = []
        for d in range(dim):
            # n bins in [0,1): (k + U)/n
            u = [(k + random.random()) / n for k in range(n)]
            random.shuffle(u)
            coords_per_dim.append(u)

        pts = []
        for k in range(n):
            x = [lows[d] + coords_per_dim[d][k] * span[d] for d in range(dim)]
            pts.append(x)
        return pts

    def safe_eval(x):
        # func expects an "array-like"; list is fine for typical use.
        # Guard against exceptions; treat failures as very bad.
        try:
            v = func(x)
            # ensure float
            return float(v)
        except Exception:
            return float("inf")

    # ---- initial sampling budget (small, time-aware) ----
    best_val = float("inf")
    best_x = None

    # Determine an initial sample size based on dimension, but keep small.
    init_n = max(8, min(40, 4 * dim + 8))
    for x in lhs_points(init_n):
        if time.time() >= deadline:
            return best_val
        v = safe_eval(x)
        if v < best_val:
            best_val, best_x = v, x

    # Fallback if everything failed
    if best_x is None:
        best_x = rand_uniform_vec()
        best_val = safe_eval(best_x)

    # ---- main loop: coordinate/pattern search with adaptive step ----
    # Initial step sizes relative to range
    step = [0.2 * s if s > 0 else 1.0 for s in span]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # Restart controls
    no_improve_iters = 0
    shrink_factor = 0.5
    grow_factor = 1.2

    # To add some exploration, occasionally perturb multiple dims
    def random_perturb(x, scale=0.1):
        y = x[:]
        for i in range(dim):
            # symmetric perturbation
            y[i] += (random.random() * 2.0 - 1.0) * scale * span[i]
        return clip_vec(y)

    # Main optimization loop
    while time.time() < deadline:
        improved = False

        # Randomize coordinate order to avoid bias
        order = list(range(dim))
        random.shuffle(order)

        # Try coordinate moves (+step, -step) for each dimension
        x0 = best_x
        v0 = best_val

        for i in order:
            if time.time() >= deadline:
                return best_val

            if step[i] <= min_step[i]:
                continue

            # + step
            xp = x0[:]
            xp[i] += step[i]
            clip_vec(xp)
            vp = safe_eval(xp)

            # - step
            xm = x0[:]
            xm[i] -= step[i]
            clip_vec(xm)
            vm = safe_eval(xm)

            # Pick best among current and the two trials
            if vp < v0 or vm < v0:
                if vp <= vm:
                    x0, v0 = xp, vp
                else:
                    x0, v0 = xm, vm
                improved = True

        if improved:
            best_x, best_val = x0, v0
            no_improve_iters = 0
            # Slightly grow steps to speed movement on smooth landscapes
            for i in range(dim):
                step[i] *= grow_factor
                # cap to range
                if step[i] > span[i]:
                    step[i] = span[i]
        else:
            no_improve_iters += 1
            # Shrink steps to refine
            for i in range(dim):
                step[i] *= shrink_factor
                if step[i] < min_step[i]:
                    step[i] = min_step[i]

            # If stuck for a while, do a restart/perturbation
            if no_improve_iters >= 6:
                no_improve_iters = 0

                # 50%: local perturbation; 50%: full random restart
                if random.random() < 0.5:
                    cand = random_perturb(best_x, scale=0.15)
                else:
                    cand = rand_uniform_vec()

                v = safe_eval(cand)
                if v < best_val:
                    best_val, best_x = v, cand
                    # reset steps when improvement occurs
                    step = [0.2 * s if s > 0 else 1.0 for s in span]

        # Extra exploration when time remains: occasional single random sample
        # (cheap insurance against local minima)
        if time.time() < deadline and random.random() < 0.2:
            x = rand_uniform_vec()
            v = safe_eval(x)
            if v < best_val:
                best_val, best_x = v, x
                step = [0.2 * s if s > 0 else 1.0 for s in span]

    return best_val
