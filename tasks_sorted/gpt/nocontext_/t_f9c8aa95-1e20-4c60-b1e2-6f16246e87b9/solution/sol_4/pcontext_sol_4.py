import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Initial Latin-hypercube-like stratified sampling
    - Local refinement with adaptive Gaussian steps (1/5 success rule style)
    - Occasional random restarts to escape local minima

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list (or sequence) of length dim.
    dim : int
        Dimension of search space.
    bounds : list of (low, high)
        Bounds per dimension.
    max_time : int or float
        Max time in seconds.

    Returns
    -------
    best : float
        Best (minimum) fitness found.
    """

    # ---------- helpers ----------
    def clip(x):
        # clip to bounds
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            y[i] = v
        return y

    def rand_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def evaluate(x):
        # func is assumed to accept a list; if it accepts other types, Python will still pass list.
        return float(func(x))

    # Basic scale per dimension (used for step sizes)
    span = [max(1e-12, bounds[i][1] - bounds[i][0]) for i in range(dim)]

    # ---------- initialization ----------
    start_time = time.time()
    deadline = start_time + float(max_time)

    best_x = None
    best = float("inf")

    # Determine a small budget for stratified initial sampling
    # Keep it light to preserve time for refinement
    init_n = max(10, 6 * dim)

    # "LHS-like": for each dimension, permute bins then sample within each bin
    # We combine them by taking the k-th bin sample for all dims.
    bins = list(range(init_n))
    per_dim_bins = []
    for _ in range(dim):
        b = bins[:]
        random.shuffle(b)
        per_dim_bins.append(b)

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            # pick bin index for this dim
            bi = per_dim_bins[i][k]
            # sample uniformly inside that bin
            a = lo + (span[i] * bi) / init_n
            b = lo + (span[i] * (bi + 1)) / init_n
            x[i] = random.uniform(a, b)
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # fallback, should not happen
        best_x = rand_uniform()
        best = evaluate(best_x)

    # ---------- adaptive local search with restarts ----------
    # Step size fraction per dimension (start moderately sized)
    sigma = [0.15 * s for s in span]  # per-dimension std dev
    min_sigma = [1e-6 * s for s in span]
    max_sigma = [0.5 * s for s in span]

    # Control parameters
    p_restart = 0.03          # chance to restart on each iteration
    success_window = 30       # adapt sigma based on recent successes
    successes = 0
    trials = 0

    # Keep a "current" point for local refinement; occasionally jump elsewhere
    cur_x = best_x[:]
    cur_f = best

    while time.time() < deadline:
        # Random restart (global exploration)
        if random.random() < p_restart:
            x = rand_uniform()
            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x
                cur_x, cur_f = x[:], fx
            else:
                # also allow restart to become current sometimes
                cur_x, cur_f = x[:], fx
            # reset adaptation a bit
            successes = 0
            trials = 0
            # reset sigma moderately
            sigma = [max(min_sigma[i], min(max_sigma[i], 0.15 * span[i])) for i in range(dim)]
            continue

        # Propose a local move (Gaussian around current)
        # Occasionally do a larger jump ("heavy tail")
        heavy = (random.random() < 0.15)
        x = [0.0] * dim
        for i in range(dim):
            step = random.gauss(0.0, sigma[i])
            if heavy:
                step *= 3.0
            x[i] = cur_x[i] + step
        x = clip(x)

        fx = evaluate(x)
        trials += 1

        # Accept if improves current; always update global best if improved
        if fx < cur_f:
            cur_x, cur_f = x, fx
            successes += 1
            if fx < best:
                best, best_x = fx, x

        # Adapt sigma every window of trials (1/5-ish rule)
        if trials >= success_window:
            rate = successes / float(trials)
            # If too successful, increase step; if not, decrease
            if rate > 0.25:
                factor = 1.3
            elif rate < 0.15:
                factor = 0.7
            else:
                factor = 1.0

            if factor != 1.0:
                for i in range(dim):
                    sigma[i] = max(min_sigma[i], min(max_sigma[i], sigma[i] * factor))

            # If we are stagnating with tiny steps, force a restart-like jump
            if rate == 0.0:
                # increase chance of escape by enlarging sigma slightly and moving current to best
                cur_x, cur_f = best_x[:], best
                for i in range(dim):
                    sigma[i] = max(sigma[i], 0.05 * span[i])

            successes = 0
            trials = 0

    return best
