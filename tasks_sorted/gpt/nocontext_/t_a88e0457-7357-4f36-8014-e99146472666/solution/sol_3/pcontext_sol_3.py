import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-limited minimization using a self-contained hybrid:
    - Latin-hypercube-like seeding (stratified per-dimension sampling)
    - (1+1)-ES style local search with adaptive step-size (success-based)
    - occasional random restarts

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """
    # ----------------- helpers -----------------
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; we pass Python list
        return float(func(x))

    def make_lhs_points(n):
        """Simple per-dimension stratified sampling (approx LHS without numpy)."""
        pts = [[0.0] * dim for _ in range(n)]
        for d in range(dim):
            lo, hi = bounds[d]
            width = hi - lo
            # strata indices shuffled
            idx = list(range(n))
            random.shuffle(idx)
            for k in range(n):
                # sample within stratum
                u = (idx[k] + random.random()) / n
                pts[k][d] = lo + u * width
        return pts

    # ----------------- initialization -----------------
    start = time.time()
    deadline = start + max_time

    # best-so-far
    best = float("inf")

    # Determine a small budget of initial seeds (time-safe)
    # Keep modest to avoid over-spending time if func is expensive.
    seed_count = max(4, min(20, 2 * dim))

    # initial step size per dimension: fraction of range
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_sigma = [0.15 * r if r > 0 else 1.0 for r in ranges]

    # Seed candidates (stratified + a few pure random)
    seeds = make_lhs_points(seed_count)
    for _ in range(max(2, seed_count // 3)):
        seeds.append(rand_uniform_vec())

    # Evaluate seeds quickly
    best_x = None
    for x in seeds:
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x[:]

    # If somehow no evaluation happened (shouldn't), return inf
    if best_x is None:
        return best

    # ----------------- main loop: adaptive (1+1)-ES with restarts -----------------
    x = best_x[:]
    fx = best
    sigma = base_sigma[:]  # per-dimension mutation scale

    # success tracking for 1/5th rule-like adaptation
    succ = 0
    tried = 0

    # restart control
    no_improve = 0
    restart_after = 50 + 10 * dim  # attempts without improvement -> restart

    while time.time() < deadline:
        # Mutation: gaussian step, per-dimension sigma
        cand = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            step = random.gauss(0.0, sigma[i])
            cand[i] = clamp(x[i] + step, lo, hi)

        f_cand = eval_f(cand)
        tried += 1

        if f_cand <= fx:
            # accept
            x, fx = cand, f_cand
            succ += 1
            no_improve = 0
            if fx < best:
                best = fx
                best_x = x[:]
        else:
            no_improve += 1

        # Adapt step sizes occasionally
        if tried >= 20:
            rate = succ / tried
            # If success rate too low -> decrease sigma, too high -> increase
            if rate < 0.18:
                factor = 0.82
            elif rate > 0.28:
                factor = 1.22
            else:
                factor = 1.0

            if factor != 1.0:
                for i in range(dim):
                    # keep sigma within reasonable bounds
                    r = ranges[i] if ranges[i] > 0 else 1.0
                    sigma[i] = max(1e-12 * r, min(0.5 * r, sigma[i] * factor))

            succ = 0
            tried = 0

        # Random restart if stuck (keeps exploration)
        if no_improve >= restart_after:
            # restart around global best with moderate sigma, or random
            if random.random() < 0.7:
                x = best_x[:]
                for i in range(dim):
                    r = ranges[i] if ranges[i] > 0 else 1.0
                    sigma[i] = 0.12 * r
            else:
                x = rand_uniform_vec()
                for i in range(dim):
                    r = ranges[i] if ranges[i] > 0 else 1.0
                    sigma[i] = 0.20 * r

            fx = eval_f(x)
            if fx < best:
                best = fx
                best_x = x[:]
            no_improve = 0

    return best
