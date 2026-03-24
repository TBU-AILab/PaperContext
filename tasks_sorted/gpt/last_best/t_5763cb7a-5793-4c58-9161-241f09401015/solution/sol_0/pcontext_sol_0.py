import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a self-contained hybrid:
    - Latin-hypercube-like initial coverage (stratified per dimension)
    - Local coordinate search around the incumbent
    - Occasional random-restart sampling (escaping local minima)
    - Step-size adaptation (success -> enlarge, failure -> shrink)

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # Basic validation / normalization
    if dim <= 0:
        return float("inf")
    if bounds is None or len(bounds) != dim:
        raise ValueError("bounds must be a list of (low, high) pairs, one per dimension")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if not (s > 0.0):
            raise ValueError("Each bound must satisfy high > low")

    def clip(x):
        # clip in-place and return
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_point(x):
        # func expects an array-like; list is fine per prompt
        return float(func(x))

    # Track best
    best_x = None
    best = float("inf")

    # --- Initial space-filling sampling (stratified per dimension) ---
    # Choose an initial sample count that scales mildly with dim and available time.
    # Keep it modest to remain time-safe for expensive funcs.
    init_n = max(8, min(40, 8 + 2 * dim))
    # Precompute permutations for each dimension to create stratification
    perms = []
    for i in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # Sample within stratum perms[i][k]
            a = perms[i][k]
            u = (a + random.random()) / init_n
            x.append(lows[i] + u * spans[i])
        fx = eval_point(x)
        if fx < best:
            best = fx
            best_x = x[:]  # copy

    if best_x is None:
        # Should not happen, but be safe
        return best

    # --- Main optimization loop ---
    # Coordinate/local search step size per dimension
    # Start with 10% of range; never below a tiny fraction of range.
    step = [0.10 * s for s in spans]
    min_step = [1e-9 * s for s in spans]  # scale to bounds

    # Bookkeeping
    no_improve_iters = 0
    # Probability of random restart grows when stuck
    base_restart_prob = 0.03

    while True:
        if time.time() >= deadline:
            return best

        # Random restart / exploration
        # Increase restart probability when many iterations without improvement.
        restart_prob = min(0.35, base_restart_prob + 0.002 * no_improve_iters)
        if random.random() < restart_prob:
            x = rand_point()
            fx = eval_point(x)
            if fx < best:
                best = fx
                best_x = x[:]
                # On success, reset local search step a bit (encourage exploration around new basin)
                step = [max(0.08 * spans[i], step[i]) for i in range(dim)]
                no_improve_iters = 0
            else:
                no_improve_iters += 1
            continue

        # Local coordinate search around incumbent
        improved = False

        # Randomize coordinate order to avoid bias
        coords = list(range(dim))
        random.shuffle(coords)

        # Try plus/minus steps along each coordinate
        for i in coords:
            if time.time() >= deadline:
                return best

            xi = best_x[:]  # start from incumbent

            # Propose +step
            xi[i] = xi[i] + step[i]
            clip(xi)
            f1 = eval_point(xi)

            if f1 < best:
                best = f1
                best_x = xi[:]
                improved = True
                continue

            # Propose -step
            xi = best_x[:]
            xi[i] = xi[i] - step[i]
            clip(xi)
            f2 = eval_point(xi)

            if f2 < best:
                best = f2
                best_x = xi[:]
                improved = True
                continue

        # Step-size adaptation
        if improved:
            no_improve_iters = 0
            # If improving, slightly expand step sizes (but not beyond range)
            for i in range(dim):
                step[i] = min(0.25 * spans[i], step[i] * 1.15)
        else:
            no_improve_iters += 1
            # If not improving, shrink step sizes
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.7)

            # If steps are extremely small and still stuck, do a stronger restart
            tiny = True
            for i in range(dim):
                if step[i] > 1e-6 * spans[i]:
                    tiny = False
                    break
            if tiny and time.time() < deadline:
                x = rand_point()
                fx = eval_point(x)
                if fx < best:
                    best = fx
                    best_x = x[:]
                    # reset steps to moderate size
                    step = [0.08 * s for s in spans]
                    no_improve_iters = 0

    # Unreachable, but keep structure explicit
    # return best
