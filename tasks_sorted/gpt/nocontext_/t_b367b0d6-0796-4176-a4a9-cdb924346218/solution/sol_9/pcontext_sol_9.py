import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
      - Sobol-like (low-discrepancy) sequence for initial space coverage (Halton sequence)
      - Local refinement via adaptive coordinate search (pattern search)
      - Occasional random restarts to escape local minima

    Returns:
        best (float): best (minimum) objective value found within max_time.
    """

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def clip_vec(x):
        return [clamp(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        return bounds[i][1] - bounds[i][0]

    def eval_f(x):
        # func expects an "array-like"; keep it simple as list
        return float(func(x))

    # First primes for Halton bases (enough for typical dims; extend if needed)
    PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113
    ]

    def van_der_corput(n, base):
        vdc, denom = 0.0, 1.0
        while n > 0:
            n, remainder = divmod(n, base)
            denom *= base
            vdc += remainder / denom
        return vdc

    def halton_point(index):
        # index should start at 1 for better distribution
        x = []
        for d in range(dim):
            base = PRIMES[d] if d < len(PRIMES) else (2 * d + 3)
            u = van_der_corput(index, base)
            lo, hi = bounds[d]
            x.append(lo + u * (hi - lo))
        return x

    # ---------- initialization ----------
    start = time.time()
    deadline = start + max_time

    # best solution tracking
    best = float("inf")
    best_x = None

    # Adaptive step sizes for coordinate search
    # start with ~10% of range per dimension (fallback to 1.0 if range is zero)
    steps = [0.1 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]
    min_steps = [1e-12 * (span(i) if span(i) > 0 else 1.0) for i in range(dim)]

    # Control parameters (kept simple and robust)
    halton_budget = 50 * max(1, dim)          # initial global sampling
    no_improve_shrink_after = 2 * max(1, dim) # shrink steps after some failures
    restart_period = 200 * max(1, dim)        # force occasional restart attempts

    # ---------- phase 1: global exploration via Halton ----------
    idx = 1
    while time.time() < deadline and idx <= halton_budget:
        x = halton_point(idx)
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x
        idx += 1

    if best_x is None:
        # as a fallback (shouldn't happen), sample a single random point
        x = [random.uniform(lo, hi) for (lo, hi) in bounds]
        best_x = x
        best = eval_f(x)

    # ---------- phase 2: iterative local search with restarts ----------
    it = 0
    failures = 0

    while time.time() < deadline:
        it += 1

        # Restart logic: occasionally jump to a new region
        # Blend: random point + best point (keeps some exploitation)
        if it % restart_period == 0:
            alpha = random.random()
            cand = []
            for i in range(dim):
                lo, hi = bounds[i]
                r = random.uniform(lo, hi)
                cand.append(alpha * best_x[i] + (1.0 - alpha) * r)
            cand = clip_vec(cand)
            fc = eval_f(cand)
            if fc < best:
                best, best_x = fc, cand
                failures = 0

        improved = False

        # Coordinate-wise exploratory moves (+step and -step), with random order
        order = list(range(dim))
        random.shuffle(order)

        for j in order:
            if time.time() >= deadline:
                return best

            step = steps[j]
            if step <= min_steps[j]:
                continue

            x0 = best_x

            # try +step
            xp = x0[:]
            xp[j] = clamp(xp[j] + step, bounds[j][0], bounds[j][1])
            fp = eval_f(xp)
            if fp < best:
                best, best_x = fp, xp
                improved = True
                continue

            # try -step
            xm = x0[:]
            xm[j] = clamp(xm[j] - step, bounds[j][0], bounds[j][1])
            fm = eval_f(xm)
            if fm < best:
                best, best_x = fm, xm
                improved = True
                continue

        if improved:
            failures = 0
            # modestly increase steps in dimensions that are still meaningful
            for j in range(dim):
                if steps[j] > min_steps[j]:
                    steps[j] *= 1.05
        else:
            failures += 1
            # if stuck, shrink step sizes (classic pattern search behavior)
            if failures >= no_improve_shrink_after:
                for j in range(dim):
                    steps[j] *= 0.5
                failures = 0

            # If steps are tiny in all dims, do a small random perturbation
            # to avoid wasting remaining time in flat regions.
            all_tiny = True
            for j in range(dim):
                if steps[j] > min_steps[j]:
                    all_tiny = False
                    break
            if all_tiny:
                cand = best_x[:]
                for j in range(dim):
                    # perturb by a small fraction of range
                    s = 0.01 * (span(j) if span(j) > 0 else 1.0)
                    cand[j] = clamp(cand[j] + random.uniform(-s, s), bounds[j][0], bounds[j][1])
                fc = eval_f(cand)
                if fc < best:
                    best, best_x = fc, cand

    return best
