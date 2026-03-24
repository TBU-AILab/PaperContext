import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization algorithm: hybrid of
    - Latin-hypercube-like stratified initialization
    - Adaptive local search (coordinate steps)
    - Simulated annealing-style occasional acceptance of worse moves
    No external libraries required.

    Returns:
        best (float): best (minimum) function value found within max_time.
    """

    # --- helpers ---
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def reflect_into_bounds(x):
        # reflect if out of bounds (helps preserve step size near edges)
        y = x[:]
        for i in range(dim):
            lo, hi = bounds[i]
            if y[i] < lo:
                y[i] = lo + (lo - y[i])
                if y[i] > hi:  # still out due to huge step
                    y[i] = lo
            elif y[i] > hi:
                y[i] = hi - (y[i] - hi)
                if y[i] < lo:
                    y[i] = hi
            # final clamp
            y[i] = clamp(y[i], lo, hi)
        return y

    def safe_eval(x):
        # Evaluate func robustly; treat exceptions/NaN/Inf as very bad.
        try:
            v = func(x)
            if v is None:
                return float("inf")
            # handle NaN without numpy
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return float("inf")
            return float(v)
        except Exception:
            return float("inf")

    def lhs_like_points(n):
        # Simple stratified sampling per-dimension (approx LHS without numpy)
        # Each dimension uses n strata; we permute strata per dim.
        strata = list(range(n))
        perms = []
        for _ in range(dim):
            p = strata[:]
            random.shuffle(p)
            perms.append(p)

        pts = []
        for k in range(n):
            x = []
            for i in range(dim):
                lo, hi = bounds[i]
                a = perms[i][k]
                # sample uniformly inside stratum [a/n, (a+1)/n]
                u = (a + random.random()) / n
                x.append(lo + u * (hi - lo))
            pts.append(x)
        return pts

    # --- time setup ---
    start = time.time()
    deadline = start + max_time

    # --- initialization ---
    # initial step sizes relative to range
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    step = [0.15 * (r if r > 0 else 1.0) for r in ranges]  # coordinate-wise step
    min_step = [1e-12 * (r if r > 0 else 1.0) for r in ranges]

    # Evaluate a small stratified batch, then keep best as the starting point
    init_n = max(8, min(40, 4 * dim))
    best_x = None
    best = float("inf")

    for x in lhs_like_points(init_n):
        if time.time() >= deadline:
            return best
        fx = safe_eval(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        best_x = rand_in_bounds()
        best = safe_eval(best_x)

    # --- main loop: adaptive coordinate search + SA acceptance ---
    x = best_x[:]
    fx = best

    # temperature schedule parameters
    T0 = 1.0
    # scale temperature by typical magnitude of fx when possible
    scale = abs(fx) if (isinstance(fx, float) and math.isfinite(fx) and abs(fx) > 1e-9) else 1.0
    T0 *= scale

    it = 0
    no_improve = 0

    while time.time() < deadline:
        it += 1
        t = time.time()
        frac = (t - start) / max(1e-12, max_time)
        # Exponential-ish cooling; stays >0
        T = T0 * (0.05 ** frac) + 1e-12

        # Choose a coordinate and direction; occasionally do a full random perturbation
        if random.random() < 0.15:
            # random perturbation (helps escape local minima)
            cand = x[:]
            for i in range(dim):
                # perturb proportionally to range, moderated by temperature
                sigma = (0.25 * ranges[i] if ranges[i] > 0 else 1.0) * (0.2 + 0.8 * (1.0 - frac))
                cand[i] += random.uniform(-sigma, sigma)
            cand = reflect_into_bounds(cand)
        else:
            i = random.randrange(dim)
            s = step[i]
            if s < min_step[i]:
                s = min_step[i]
            direction = -1.0 if random.random() < 0.5 else 1.0
            cand = x[:]
            cand[i] = cand[i] + direction * s
            cand = reflect_into_bounds(cand)

        fc = safe_eval(cand)
        delta = fc - fx

        accept = False
        if fc <= fx:
            accept = True
        else:
            # simulated annealing acceptance
            # If delta is huge, acceptance will be ~0
            p = math.exp(-delta / max(1e-12, T))
            if random.random() < p:
                accept = True

        if accept:
            x, fx = cand, fc

        # Track global best
        if fx < best:
            best = fx
            best_x = x[:]
            no_improve = 0
            # modestly increase steps after improvement (exploit momentum)
            for j in range(dim):
                step[j] = min(step[j] * 1.05 + 1e-15, 0.5 * (ranges[j] if ranges[j] > 0 else 1.0))
        else:
            no_improve += 1

        # If stuck, reduce step sizes (local refinement)
        if no_improve >= 30:
            for j in range(dim):
                step[j] = max(step[j] * 0.7, min_step[j])
            no_improve = 0

        # Safety exit if extremely fine steps everywhere and near end of time
        if frac > 0.95 and all(step[j] <= 1.001 * min_step[j] for j in range(dim)):
            break

    # return fitness of the best found solution
    return best
