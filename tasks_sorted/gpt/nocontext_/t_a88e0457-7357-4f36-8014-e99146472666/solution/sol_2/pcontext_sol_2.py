import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified sampling for initialization
    - Local coordinate search around the best
    - Adaptive step-size + random restarts
    Works with: func(params) where params is a list/array-like of length dim.
    Returns: best (float) minimum fitness found within max_time seconds.
    """
    start = time.time()
    deadline = start + max_time

    # ---- helpers ----
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_params(p):
        # func expects array-like; list is fine per template examples
        try:
            v = float(func(p))
        except Exception:
            v = float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # scale per dimension (range)
    span = [max(1e-12, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    # ---- initialization: stratified samples (LHS-like) ----
    # Choose sample count based on dim, but keep it small and time-aware.
    init_n = max(8, min(40, 10 * dim))
    # Precompute per-dimension strata order
    strata = []
    for i in range(dim):
        perm = list(range(init_n))
        random.shuffle(perm)
        strata.append(perm)

    best = float("inf")
    best_x = None

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            # sample within stratum
            a = (strata[i][k] + random.random()) / init_n
            x.append(lo + a * (hi - lo))
        f = eval_params(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        # fallback
        best_x = rand_in_bounds()
        best = eval_params(best_x)

    # ---- main loop: adaptive local search + restarts ----
    # Step sizes start moderately large then shrink on stagnation.
    step = [0.15 * s for s in span]
    min_step = [1e-9 * s for s in span]

    # parameters controlling adaptation
    no_improve = 0
    shrink_every = 30 + 5 * dim
    restart_every = 200 + 20 * dim

    # Keep a working point (typically the best)
    x = best_x[:]
    fx = best

    while time.time() < deadline:
        improved = False

        # Coordinate-wise probe around current best
        # Random order to reduce bias
        idxs = list(range(dim))
        random.shuffle(idxs)

        for i in idxs:
            if time.time() >= deadline:
                return best

            # try +step and -step
            for direction in (1.0, -1.0):
                cand = x[:]
                cand[i] = clip(cand[i] + direction * step[i], bounds[i][0], bounds[i][1])
                fc = eval_params(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
                    if fx < best:
                        best, best_x = fx, x[:]
                    break  # move to next coordinate after improvement

        # Occasional small random perturbation around best to escape shallow basins
        if time.time() < deadline:
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

            # Adapt step sizes
            if no_improve > 0 and (no_improve % shrink_every == 0):
                for i in range(dim):
                    step[i] *= 0.5
                    if step[i] < min_step[i]:
                        step[i] = min_step[i]

            # Random restart if stuck too long or steps got tiny
            if (no_improve % restart_every == 0 and no_improve > 0) or all(step[i] <= 5 * min_step[i] for i in range(dim)):
                # restart near global best with some probability, else uniform
                if random.random() < 0.7 and best_x is not None:
                    # Gaussian-like perturbation using sum of uniforms (no external libs)
                    cand = []
                    for i in range(dim):
                        lo, hi = bounds[i]
                        # approx normal(0,1) via 12 uniforms - 6
                        z = sum(random.random() for _ in range(12)) - 6.0
                        cand.append(clip(best_x[i] + z * 0.10 * span[i], lo, hi))
                    x = cand
                else:
                    x = rand_in_bounds()

                fx = eval_params(x)
                if fx < best:
                    best, best_x = fx, x[:]
                # reset steps to a moderate size
                step = [max(0.05 * span[i], min_step[i]) for i in range(dim)]
                no_improve = 0

        # Also try a pure random sample occasionally (global exploration)
        if time.time() >= deadline:
            break
        if random.random() < 0.05:
            cand = rand_in_bounds()
            fc = eval_params(cand)
            if fc < best:
                best, best_x = fc, cand
                x, fx = cand[:], fc

    return best
