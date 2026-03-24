import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
      - Sobol-like (van der Corput/Halton) low-discrepancy initialization
      - Iterated local improvement with adaptive step sizes (pattern search / hill-climb)
      - Occasional random/LDS restarts to escape local minima

    No external libraries required.

    Args:
        func: callable(x)->float where x is a list/sequence of length dim
        dim: int, dimensionality
        bounds: list of (low, high) pairs of length dim
        max_time: seconds (int/float)

    Returns:
        best (float): best objective value found within time limit
    """

    # ------------------------- helpers ------------------------- #
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def sanitize_bounds(bounds):
        b = []
        for (lo, hi) in bounds:
            if hi < lo:
                lo, hi = hi, lo
            # Degenerate bounds are allowed
            b.append((float(lo), float(hi)))
        return b

    def safe_eval(x):
        # Ensure x is inside bounds and handle bad function evaluations.
        xx = [clamp(x[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
        try:
            y = func(xx)
            if y is None:
                return float("inf"), xx
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf"), xx
            return y, xx
        except Exception:
            return float("inf"), xx

    # Halton sequence for deterministic-ish space-filling samples
    # (good initial coverage without numpy)
    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def first_primes(k):
        primes = []
        candidate = 2
        while len(primes) < k:
            is_p = True
            r = int(candidate ** 0.5)
            for p in primes:
                if p > r:
                    break
                if candidate % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(candidate)
            candidate += 1
        return primes

    def halton_point(index, bases):
        # index should start from 1 typically
        return [vdc(index, b) for b in bases]

    def scale_to_bounds(u):
        x = []
        for i in range(dim):
            lo, hi = bnds[i]
            if hi == lo:
                x.append(lo)
            else:
                x.append(lo + u[i] * (hi - lo))
        return x

    def random_point():
        return [random.uniform(bnds[i][0], bnds[i][1]) for i in range(dim)]

    # Local search: coordinate + random directions with adaptive step
    def local_improve(x0, f0, time_deadline):
        x = x0[:]
        f = f0

        # Initial step sizes proportional to range
        ranges = [bnds[i][1] - bnds[i][0] for i in range(dim)]
        # If a range is 0, keep step 0 for that dim
        step = [0.2 * r for r in ranges]
        min_step = [max(1e-12, 1e-6 * r) if r > 0 else 0.0 for r in ranges]

        # Limit inner evaluations per call to keep responsiveness
        # (still adaptive to time remaining)
        eval_budget = 40 + 10 * dim

        # Precompute a small set of random direction vectors
        # to complement coordinate moves
        dirs = []
        for _ in range(6):
            v = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            norm = math.sqrt(sum(t*t for t in v)) or 1.0
            dirs.append([t / norm for t in v])

        tries = 0
        while tries < eval_budget and time.time() < time_deadline:
            improved = False

            # 1) Coordinate pattern moves
            for i in range(dim):
                if time.time() >= time_deadline:
                    return x, f

                if step[i] <= 0:
                    continue

                # Try + step
                xp = x[:]
                xp[i] = clamp(xp[i] + step[i], bnds[i][0], bnds[i][1])
                fp, xp = safe_eval(xp)
                tries += 1
                if fp < f:
                    x, f = xp, fp
                    improved = True
                    continue

                if time.time() >= time_deadline or tries >= eval_budget:
                    break

                # Try - step
                xm = x[:]
                xm[i] = clamp(xm[i] - step[i], bnds[i][0], bnds[i][1])
                fm, xm = safe_eval(xm)
                tries += 1
                if fm < f:
                    x, f = xm, fm
                    improved = True

                if tries >= eval_budget:
                    break

            if time.time() >= time_deadline or tries >= eval_budget:
                break

            # 2) Random direction moves (a few)
            for d in dirs:
                if time.time() >= time_deadline or tries >= eval_budget:
                    break
                # Use average step scale
                s = 0.0
                cnt = 0
                for i in range(dim):
                    if step[i] > 0:
                        s += step[i]
                        cnt += 1
                if cnt == 0:
                    break
                s /= cnt

                xt = [clamp(x[i] + s * d[i], bnds[i][0], bnds[i][1]) for i in range(dim)]
                ft, xt = safe_eval(xt)
                tries += 1
                if ft < f:
                    x, f = xt, ft
                    improved = True

            # Adapt steps
            if improved:
                # Slightly increase steps to move faster along improvement
                for i in range(dim):
                    step[i] *= 1.15
                    # cap step to range
                    if ranges[i] > 0:
                        step[i] = min(step[i], 0.5 * ranges[i])
            else:
                # Reduce steps when stuck
                all_small = True
                for i in range(dim):
                    step[i] *= 0.5
                    if step[i] > min_step[i]:
                        all_small = False
                if all_small:
                    break

        return x, f

    # ------------------------- main ------------------------- #
    bnds = sanitize_bounds(bounds)
    start = time.time()
    deadline = start + float(max_time)

    # Edge cases
    if dim <= 0:
        return float("inf")
    if len(bnds) != dim:
        raise ValueError("bounds length must match dim")

    # Prepare Halton bases (first primes)
    bases = first_primes(dim)

    best = float("inf")
    best_x = None

    # Phase scheduling based on time: allocate some initial global samples, then loop
    # Ensure at least a few points even for tiny time budgets.
    init_points = max(10, 5 * dim)

    # Initial low-discrepancy sweep
    idx = 1
    while idx <= init_points and time.time() < deadline:
        u = halton_point(idx, bases)
        x = scale_to_bounds(u)
        f, x = safe_eval(x)
        if f < best:
            best, best_x = f, x
        idx += 1

    # Main loop: restart + local improve
    # Alternate between:
    #  - local search from best (intensify)
    #  - local search from new LDS/random (diversify)
    restarts = 0
    while time.time() < deadline:
        now = time.time()
        remaining = deadline - now
        if remaining <= 0:
            break

        # Decide seed point
        # Every few iterations, explore a fresh region; otherwise exploit best.
        if best_x is None or restarts % 3 == 2:
            # mix Halton and random
            if restarts % 2 == 0:
                u = halton_point(idx, bases)
                idx += 1
                x0 = scale_to_bounds(u)
            else:
                x0 = random_point()
            f0, x0 = safe_eval(x0)
        else:
            # slight perturbation around best
            x0 = best_x[:]
            for i in range(dim):
                lo, hi = bnds[i]
                r = hi - lo
                if r > 0:
                    x0[i] = clamp(x0[i] + random.uniform(-0.05, 0.05) * r, lo, hi)
            f0, x0 = safe_eval(x0)

        if f0 < best:
            best, best_x = f0, x0

        # Local improvement with a sub-deadline to preserve time for restarts
        # Give at most ~25% of remaining time per local run (bounded).
        sub = min(deadline, time.time() + max(0.01, 0.25 * remaining))
        x1, f1 = local_improve(x0, f0, sub)
        if f1 < best:
            best, best_x = f1, x1

        restarts += 1

    return best
