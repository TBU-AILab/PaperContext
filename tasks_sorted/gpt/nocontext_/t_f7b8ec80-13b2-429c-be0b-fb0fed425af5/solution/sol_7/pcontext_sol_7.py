import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - LHS-like diversified initial sampling
      - Best-so-far tracking
      - Coordinate/local search around incumbent
      - Occasional random/global restarts
      - Adaptive step size reduction on improvements/stagnation

    Returns:
      best (float): fitness of the best found solution
    """

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def sample_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def reflect_into_bounds(v):
        # reflect value into [lo, hi] (helps keep gaussian steps inside bounds)
        lo, hi = v[1], v[2]

    def eval_candidate(x):
        # func expects an array-like; we pass a Python list
        try:
            fx = func(x)
        except Exception:
            # If function fails on some input, treat as very bad.
            fx = float("inf")
        # If it returns NaN or non-finite, treat as bad.
        if fx is None or not isinstance(fx, (int, float)) or math.isnan(fx) or math.isinf(fx):
            return float("inf")
        return float(fx)

    def latin_hypercube_batch(n):
        # Simple LHS-style batch: per dimension, use n stratified bins and shuffle
        # Produces n points in dim dimensions.
        strata = []
        for d in range(dim):
            lo, hi = bounds[d]
            width = (hi - lo) / n if n > 0 else 0.0
            bins = [lo + (i + random.random()) * width for i in range(n)]
            random.shuffle(bins)
            strata.append(bins)
        pts = []
        for i in range(n):
            pts.append([strata[d][i] for d in range(dim)])
        return pts

    # ---------- initialization ----------
    t0 = time.time()
    deadline = t0 + max_time

    # Handle degenerate cases
    if dim <= 0:
        return eval_candidate([])

    # Precompute ranges and a scale for steps
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # Avoid zero ranges in step computations
    safe_ranges = [r if r > 0 else 1.0 for r in ranges]

    best = float("inf")
    best_x = None

    # Initial diversified sampling budget (small, time-safe)
    init_n = max(8, min(64, 8 * dim))
    init_pts = latin_hypercube_batch(init_n)

    for x in init_pts:
        if time.time() >= deadline:
            return best
        fx = eval_candidate(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        # fallback if all failed
        best_x = sample_uniform()
        best = eval_candidate(best_x)

    # ---------- main loop ----------
    # Step sizes per dimension start at ~20% of range
    steps = [0.2 * sr for sr in safe_ranges]
    min_steps = [1e-12 * sr for sr in safe_ranges]

    # Stagnation / restart control
    no_improve = 0
    restart_after = 30 + 5 * dim  # heuristic
    shrink_on_no_improve = 0.85
    grow_on_improve = 1.05

    # Probability of global random sample vs local move
    p_global = 0.15

    # coordinate order for local search
    coord = list(range(dim))
    random.shuffle(coord)
    coord_idx = 0

    while True:
        if time.time() >= deadline:
            return best

        # Decide between global exploration and local exploitation
        if random.random() < p_global or no_improve >= restart_after:
            # Global exploration / restart:
            # - sometimes sample uniform
            # - sometimes sample around best with larger noise
            if random.random() < 0.6:
                x = sample_uniform()
            else:
                x = best_x[:]
                for d in range(dim):
                    lo, hi = bounds[d]
                    sigma = 0.5 * safe_ranges[d]
                    x[d] = clamp(x[d] + random.gauss(0.0, sigma), lo, hi)

            fx = eval_candidate(x)
            if fx < best:
                best, best_x = fx, x[:]
                no_improve = 0
                # modestly increase steps to encourage movement after success
                for d in range(dim):
                    steps[d] = max(min_steps[d], min(steps[d] * grow_on_improve, safe_ranges[d]))
            else:
                no_improve += 1
                # shrink steps when not improving
                for d in range(dim):
                    steps[d] = max(min_steps[d], steps[d] * shrink_on_no_improve)

            # reset restart counter if we forced a restart
            if no_improve >= restart_after:
                no_improve = 0
                # reset steps to medium values
                steps = [0.2 * sr for sr in safe_ranges]
            continue

        # Local search: coordinate perturbation around incumbent
        d = coord[coord_idx]
        coord_idx = (coord_idx + 1) % dim
        if coord_idx == 0:
            random.shuffle(coord)

        lo, hi = bounds[d]
        base = best_x[d]
        step = steps[d]

        # Try a few directional proposals (pattern search)
        candidates = []
        candidates.append(base + step)
        candidates.append(base - step)
        # Occasionally try a smaller step too
        candidates.append(base + 0.5 * step)
        candidates.append(base - 0.5 * step)

        improved = False
        for val in candidates:
            if time.time() >= deadline:
                return best
            x = best_x[:]
            x[d] = clamp(val, lo, hi)
            fx = eval_candidate(x)
            if fx < best:
                best, best_x = fx, x[:]
                improved = True
                no_improve = 0
                # grow step slightly for this coordinate on improvement
                steps[d] = max(min_steps[d], min(steps[d] * grow_on_improve, safe_ranges[d]))
                break

        if not improved:
            no_improve += 1
            # shrink this coordinate's step if no improvement
            steps[d] = max(min_steps[d], steps[d] * shrink_on_no_improve)
            # If all steps are tiny, inject exploration
            if all(steps[i] <= (min_steps[i] * 1000.0) for i in range(dim)) and random.random() < 0.5:
                # reinitialize steps and do a global sample soon
                steps = [0.2 * sr for sr in safe_ranges]
                p_global = min(0.5, p_global + 0.05)
            else:
                # slowly anneal exploration probability down/up
                p_global = max(0.05, min(0.35, p_global * 0.999))
