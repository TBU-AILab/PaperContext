import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
      - Latin-hypercube-like stratified seeding
      - (1+λ)-Evolution Strategy with step-size adaptation (success-based)
      - Occasional coordinate pattern search for local refinement
    Returns: best (float) = minimum fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------------------------- helpers ----------------------------

    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def clamp_vec(v):
        return [clamp(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; we pass list (common compatible)
        return float(func(x))

    widths = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # scale baseline sigma to search space size
    base_sigma = 0.2 * (sum(widths) / max(1, dim))
    min_sigma = 1e-12 * (sum(widths) / max(1, dim) + 1.0)

    # -------------------------- initialization -----------------------

    best_x = None
    best = float("inf")

    # Stratified (LHS-like) seeding: per dimension, shuffle strata
    # Keep it small; we are time-bounded.
    seed_n = max(8, 4 * dim)
    strata = []
    for i in range(dim):
        idx = list(range(seed_n))
        random.shuffle(idx)
        strata.append(idx)

    # Evaluate seeds
    for k in range(seed_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            lo, hi = bounds[i]
            a = strata[i][k]
            # sample within stratum
            u = (a + random.random()) / seed_n
            x.append(lo + u * (hi - lo))
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    # If something went wrong (e.g., func errors), fall back to random
    if best_x is None:
        best_x = rand_uniform_vec()
        best = eval_f(best_x)

    # ES parameters
    lam = max(8, 4 * dim)          # offspring count
    # success rule target rate ~ 1/5
    target_success = 0.20
    sigma = base_sigma
    no_improve = 0

    # Local pattern search step sizes (per-dim), relative to width
    coord_step = [0.05 * w for w in widths]
    min_coord_step = [1e-12 * (w + 1.0) for w in widths]

    # ------------------------------ loop ----------------------------

    while time.time() < deadline:
        # (1+λ)-ES: sample around current best
        parent = best_x
        parent_f = best

        improved = False

        # Generate offspring; keep best
        best_child = None
        best_child_f = parent_f

        # Antithetic sampling to reduce variance a bit
        half = lam // 2
        for j in range(half):
            if time.time() >= deadline:
                return best

            # Gaussian step
            step = [random.gauss(0.0, sigma) for _ in range(dim)]
            child1 = clamp_vec([parent[i] + step[i] for i in range(dim)])
            f1 = eval_f(child1)
            if f1 < best_child_f:
                best_child_f = f1
                best_child = child1

            # antithetic
            child2 = clamp_vec([parent[i] - step[i] for i in range(dim)])
            f2 = eval_f(child2)
            if f2 < best_child_f:
                best_child_f = f2
                best_child = child2

        # If lam is odd, one extra
        if lam % 2 == 1 and time.time() < deadline:
            step = [random.gauss(0.0, sigma) for _ in range(dim)]
            child = clamp_vec([parent[i] + step[i] for i in range(dim)])
            f = eval_f(child)
            if f < best_child_f:
                best_child_f = f
                best_child = child

        # Selection
        if best_child is not None and best_child_f < parent_f:
            best_x, best = best_child, best_child_f
            improved = True
            no_improve = 0
        else:
            no_improve += 1

        # Success-based step-size adaptation (log-space)
        # If improved: slightly increase sigma; else decrease
        if improved:
            sigma *= math.exp(0.15)  # modest growth
        else:
            # shrink more often to focus
            sigma *= math.exp(-0.10)

        # Keep sigma within sensible bounds
        # upper bound: a fraction of average width
        avgw = sum(widths) / max(1, dim)
        sigma = max(min_sigma, min(sigma, 0.5 * avgw if avgw > 0 else sigma))

        # Occasionally do a coordinate pattern search when stuck
        if no_improve >= 10:
            no_improve = 0
            # Try improving along each coordinate with +/- step; shrink if no gain
            for i in range(dim):
                if time.time() >= deadline:
                    return best

                step_i = coord_step[i]
                if step_i < min_coord_step[i]:
                    continue

                base = best_x
                base_f = best

                # plus
                xp = base[:]
                xp[i] = clamp(xp[i] + step_i, bounds[i][0], bounds[i][1])
                fp = eval_f(xp)

                # minus
                xm = base[:]
                xm[i] = clamp(xm[i] - step_i, bounds[i][0], bounds[i][1])
                fm = eval_f(xm)

                if fp < base_f or fm < base_f:
                    if fp <= fm:
                        best_x, best = xp, fp
                    else:
                        best_x, best = xm, fm
                    # if we found improvement, consider increasing that coordinate step a bit
                    coord_step[i] = min(coord_step[i] * 1.2, widths[i] if widths[i] > 0 else coord_step[i])
                    # also gently increase global sigma to allow escape
                    sigma = min(sigma * 1.1, 0.5 * avgw if avgw > 0 else sigma)
                else:
                    # no improvement: reduce step for this coordinate
                    coord_step[i] *= 0.5

        # Rare restart to escape local minima (time permitting)
        # Trigger if sigma becomes tiny or repeated stagnation in narrow areas.
        if sigma <= 2.0 * min_sigma and time.time() < deadline:
            # probabilistic restart
            if random.random() < 0.15:
                x = rand_uniform_vec()
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x
                sigma = base_sigma

    return best
