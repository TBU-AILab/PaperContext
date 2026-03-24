import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - Sobol-ish / Halton-like initialization (van der Corput bases)
      - Adaptive local search with coordinate + random directions
      - Step-size control with occasional restarts
    Returns best (minimum) fitness found.
    """

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def project(vec):
        return [clamp(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects an "array-like"; list is fine for most uses
        return float(func(x))

    # Van der Corput / Halton sequence for better initial coverage than pure random
    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point(index, bases):
        p = []
        for i in range(dim):
            lo, hi = bounds[i]
            u = vdc(index, bases[i])
            p.append(lo + (hi - lo) * u)
        return p

    def primes_up_to(k):
        primes = []
        x = 2
        while len(primes) < k:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def l2_norm(v):
        return math.sqrt(sum(t * t for t in v))

    def rand_unit_vec():
        # Gaussian-free: use uniform(-1,1) then normalize
        v = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        n = l2_norm(v)
        if n == 0.0:
            v[0] = 1.0
            n = 1.0
        return [t / n for t in v]

    # ---------- timing ----------
    start = time.time()
    deadline = start + float(max_time)

    # ---------- initialization ----------
    bases = primes_up_to(dim)

    best = float("inf")
    best_x = None

    # Characteristic scale per dimension
    spans = [max(1e-12, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    # Start with a modest number of space-filling samples, then local search dominates.
    # Keep it small to ensure quick first improvement under short time limits.
    init_n = max(8, 4 * dim)
    idx = 1

    while idx <= init_n and time.time() < deadline:
        x = halton_point(idx, bases)
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x
        idx += 1

    if best_x is None:
        # Extremely small max_time edge case
        return best

    # ---------- adaptive local search ----------
    # Initial step: ~10% of span (capped), then adapt down on failures
    step = [0.10 * s for s in spans]
    min_step = [1e-12 * s + 1e-15 for s in spans]
    max_step = [0.50 * s for s in spans]

    # Track stagnation; trigger restarts to escape local minima
    no_improve = 0
    stagnation_limit = 30 + 5 * dim

    # Evaluate current best (already evaluated in initialization)
    current_x = best_x[:]
    current_f = best

    # Local search loop until time is up
    while time.time() < deadline:
        improved = False

        # 1) Coordinate pattern search (plus/minus along each axis)
        order = list(range(dim))
        random.shuffle(order)
        for i in order:
            if time.time() >= deadline:
                break

            for sign in (-1.0, 1.0):
                cand = current_x[:]
                cand[i] = cand[i] + sign * step[i]
                cand = project(cand)
                fc = eval_f(cand)
                if fc < current_f:
                    current_x, current_f = cand, fc
                    if fc < best:
                        best, best_x = fc, cand
                    improved = True
                    break
            if improved:
                # Greedy: restart coordinate sweep from new point
                break

        if time.time() >= deadline:
            break

        # 2) Random direction tries (useful when axes are not aligned with optimum)
        if not improved:
            tries = 2 + dim  # small, time-friendly
            for _ in range(tries):
                if time.time() >= deadline:
                    break
                d = rand_unit_vec()
                # scale by average step size
                avg_step = sum(step) / max(1, dim)
                alpha = avg_step * (0.5 + random.random())  # in [0.5,1.5)*avg
                cand = [current_x[j] + alpha * d[j] for j in range(dim)]
                cand = project(cand)
                fc = eval_f(cand)
                if fc < current_f:
                    current_x, current_f = cand, fc
                    if fc < best:
                        best, best_x = fc, cand
                    improved = True
                    break

        # 3) Step-size adaptation / restart
        if improved:
            no_improve = 0
            # gently increase step sizes (up to cap) to move faster when improving
            for i in range(dim):
                step[i] = min(max_step[i], step[i] * 1.15)
        else:
            no_improve += 1
            # decrease step sizes to refine search
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.65)

            # If we are stuck for a while, restart around global best with larger steps
            if no_improve >= stagnation_limit:
                no_improve = 0
                # restart center at best_x; randomize steps to escape basin
                current_x = best_x[:]
                current_f = best
                for i in range(dim):
                    # reset step to a moderate size with randomness
                    step[i] = min(max_step[i], max(min_step[i], spans[i] * (0.05 + 0.20 * random.random())))
                # also try a few global random samples quickly
                for _ in range(2 + dim):
                    if time.time() >= deadline:
                        break
                    x = rand_uniform_vec()
                    fx = eval_f(x)
                    if fx < best:
                        best, best_x = fx, x
                        current_x, current_f = x, fx

    # return fitness of the best found solution
    return best
