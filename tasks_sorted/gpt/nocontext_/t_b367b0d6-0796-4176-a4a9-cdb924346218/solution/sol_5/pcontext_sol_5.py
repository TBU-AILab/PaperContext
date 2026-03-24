import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded global optimization using a self-contained:
    - Sobol-like (Van der Corput / Halton) low-discrepancy initializer
    - Best-first local refinement (coordinate + pattern search)
    - Adaptive step sizes + random restarts

    Returns:
        best (float): best (minimum) function value found within max_time.
    """

    # ---------- helpers ----------
    def clip(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi: 
            return hi
        return x

    def clamp_vec(x):
        return [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        return bounds[i][1] - bounds[i][0]

    def rand_uniform_vec():
        return [bounds[i][0] + random.random() * span(i) for i in range(dim)]

    # first primes for Halton (enough for typical dims; fallback for larger dims)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        r = int(n ** 0.5)
        f = 3
        while f <= r:
            if n % f == 0:
                return False
            f += 2
        return True

    def nth_prime(k):
        # 0-based: nth_prime(0)=2
        count = -1
        n = 1
        while True:
            n += 1
            if is_prime(n):
                count += 1
                if count == k:
                    return n

    def prime_for_dim(i):
        if i < len(_PRIMES):
            return _PRIMES[i]
        return nth_prime(i)

    def vdc(n, base):
        # Van der Corput radical inverse
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index):
        # index should start from 1 for better dispersion
        x = []
        for i in range(dim):
            base = prime_for_dim(i)
            u = vdc(index, base)
            x.append(bounds[i][0] + u * span(i))
        return x

    def safe_eval(x):
        # robust evaluation: treat NaN/Inf/exceptions as very bad
        try:
            y = func(x)
            if y is None:
                return float("inf")
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf")
            return y
        except Exception:
            return float("inf")

    # ---------- initialization ----------
    start = time.time()
    deadline = start + max(0.0, float(max_time))

    best = float("inf")
    best_x = None

    # initial step sizes (per dimension)
    base_steps = [0.15 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]

    # Evaluate a few low-discrepancy points quickly
    idx = 1
    init_budget = 8 * max(1, dim)  # small but useful
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best
        x = halton_point(idx)
        idx += 1
        y = safe_eval(x)
        if y < best:
            best, best_x = y, x

    if best_x is None:
        # fallback
        best_x = rand_uniform_vec()
        best = safe_eval(best_x)

    # ---------- main loop: multi-start local improvement ----------
    # Parameters tuned for general use without external libs
    min_step_frac = 1e-6
    shrink = 0.5
    expand = 1.25

    # current "center" for local search
    center = list(best_x)
    center_y = best

    # local step sizes adapt over time
    steps = list(base_steps)

    # attempt counters
    it = 0
    stagnation = 0

    while time.time() < deadline:
        it += 1

        improved = False

        # --- pattern/coordinate search around center ---
        # Shuffle dimensions to avoid bias
        dims = list(range(dim))
        random.shuffle(dims)

        for j in dims:
            if time.time() >= deadline:
                return best

            sj = steps[j]
            if sj <= min_step_frac * (span(j) if span(j) > 0 else 1.0):
                continue

            # Try + and - direction
            for direction in (1.0, -1.0):
                cand = list(center)
                cand[j] = clip(cand[j] + direction * sj, bounds[j][0], bounds[j][1])

                # If nothing changes due to clipping, skip
                if cand[j] == center[j]:
                    continue

                y = safe_eval(cand)
                if y < center_y:
                    center, center_y = cand, y
                    improved = True
                    if y < best:
                        best, best_x = y, list(cand)
                    break  # move to next dimension after an improvement
            # small early exit if improved and time is tight
            if improved and (deadline - time.time()) < 0.001:
                return best

        # --- adaptive step size update ---
        if improved:
            stagnation = 0
            # slightly increase steps to move faster on slopes
            for j in range(dim):
                steps[j] = min(steps[j] * expand, 0.5 * span(j) if span(j) > 0 else steps[j])
        else:
            stagnation += 1
            # shrink steps to refine locally
            for j in range(dim):
                steps[j] *= shrink

        # --- occasional random/global restart to escape local minima ---
        # Trigger restarts if stagnating or steps are very small
        very_small = True
        for j in range(dim):
            if steps[j] > min_step_frac * (span(j) if span(j) > 0 else 1.0):
                very_small = False
                break

        if stagnation >= (5 + dim) or very_small:
            # Mix: best-guided perturbation + Halton + pure random
            if time.time() >= deadline:
                return best

            mode = it % 3
            if mode == 0:
                # best-guided perturbation
                cand = list(best_x)
                for j in range(dim):
                    # perturb within ~10% of range
                    r = (random.random() * 2.0 - 1.0) * 0.10 * span(j)
                    cand[j] = clip(cand[j] + r, bounds[j][0], bounds[j][1])
            elif mode == 1:
                cand = halton_point(idx)
                idx += 1
            else:
                cand = rand_uniform_vec()

            y = safe_eval(cand)
            if y < best:
                best, best_x = y, list(cand)

            center, center_y = (list(best_x), best) if best_x is not None else (cand, y)

            # reset steps (slightly smaller over time for refinement)
            decay = 1.0 / (1.0 + 0.05 * (it / max(1, dim)))
            steps = [max(min_step_frac * (span(j) if span(j) > 0 else 1.0),
                         base_steps[j] * decay)
                     for j in range(dim)]
            stagnation = 0

    return best
