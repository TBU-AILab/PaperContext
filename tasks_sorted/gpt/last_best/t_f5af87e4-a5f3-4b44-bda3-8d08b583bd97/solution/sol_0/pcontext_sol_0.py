import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
      - Latin-hypercube-like initial sampling (stratified per dimension)
      - Local refinement around the current best (adaptive radius)
      - Occasional global resampling to escape local minima

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a sequence/list of length dim.
    dim : int
        Dimension of the search space.
    bounds : list of (low, high)
        Bounds for each dimension.
    max_time : int or float
        Time budget in seconds.

    Returns
    -------
    best : float
        Best (minimum) objective value found within the time budget.
    """

    # --- helpers (no external libs) ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        y = list(x)
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func is expected to accept list/array-like
        return float(func(x))

    # --- time control ---
    deadline = time.perf_counter() + float(max_time)

    # Best-so-far
    best = float("inf")
    best_x = None

    # --- Phase 1: stratified initialization (cheap space-filling) ---
    # Choose number of initial samples relative to dimension
    init_n = max(10, 8 * dim)

    # For each dimension, create permuted strata indices 0..init_n-1
    strata = []
    for _ in range(dim):
        idx = list(range(init_n))
        random.shuffle(idx)
        strata.append(idx)

    for j in range(init_n):
        if time.perf_counter() >= deadline:
            return best
        x = []
        for i in range(dim):
            # sample inside stratum
            u = (strata[i][j] + random.random()) / init_n
            x.append(lows[i] + u * spans[i])
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # Fallback (shouldn't happen unless func fails)
        return best

    # --- Phase 2: adaptive local search with occasional global jumps ---
    # Step radius as fraction of span; start moderately wide, shrink on stagnation
    radius = [0.25 * s for s in spans]
    min_radius = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # Success/stagnation tracking
    no_improve = 0
    iter_count = 0

    # Controls
    # probability of global jump increases with prolonged stagnation
    base_global_p = 0.05

    while True:
        if time.perf_counter() >= deadline:
            return best

        iter_count += 1

        # Decide between local perturbation vs global sample
        # Increase global exploration if stuck
        global_p = min(0.5, base_global_p + 0.002 * no_improve)
        do_global = (random.random() < global_p)

        if do_global:
            x = rand_uniform_point()
        else:
            # Local: gaussian-like step via sum of uniforms (approx normal)
            x = list(best_x)
            for i in range(dim):
                if spans[i] == 0:
                    continue
                # approx normal in [-1,1] by averaging uniforms, then scale
                z = (random.random() + random.random() + random.random() + random.random() - 2.0) / 2.0
                step = z * radius[i]
                x[i] += step
            x = clamp(x)

        fx = eval_f(x)

        if fx < best:
            best = fx
            best_x = x
            no_improve = 0
            # Slightly expand radius on success (but keep bounded)
            for i in range(dim):
                radius[i] = min(0.5 * spans[i], radius[i] * 1.05)
        else:
            no_improve += 1
            # Shrink radius if stagnating
            if no_improve % (20 + 2 * dim) == 0:
                for i in range(dim):
                    radius[i] = max(min_radius[i], radius[i] * 0.7)

            # If extremely small radius and still no progress, restart locally/globally
            if no_improve % (200 + 10 * dim) == 0:
                # soft restart: keep best, but reset radius and probe nearby
                radius = [max(1e-6 * spans[i], 0.25 * spans[i]) for i in range(dim)]

    # Unreachable; loop returns on time budget.
