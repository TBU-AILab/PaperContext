import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization
    - (1+1)-ES style local search with adaptive step size (success rule)
    - Occasional random restarts to escape local minima

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/array of length dim
    dim : int
    bounds : list of (low, high)
    max_time : int/float
        seconds

    Returns
    -------
    best : float
        best (minimum) objective value found within time budget
    """
    t0 = time.time()

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    # Helpers
    def clamp(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func may accept list or array-like; keep it simple/self-contained
        return float(func(x))

    # ---- Initial sampling: stratified across each dimension ----
    # Choose a modest number so we don't burn time on initialization.
    n_init = max(8, min(40, 6 * dim))
    # Pre-generate stratified bins per dimension, then shuffle independently.
    strata = []
    for i in range(dim):
        idxs = list(range(n_init))
        random.shuffle(idxs)
        strata.append(idxs)

    best_x = None
    best = float("inf")

    for k in range(n_init):
        if time.time() - t0 >= max_time:
            return best
        x = []
        for i in range(dim):
            # sample within the k-th stratum for dim i
            s = strata[i][k]
            u = (s + random.random()) / n_init
            x.append(lows[i] + u * spans[i])
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        # degenerate case
        best_x = rand_point()
        best = evaluate(best_x)

    # ---- Local search with adaptive Gaussian steps + restarts ----
    # Start step size as a fraction of range.
    base_sigma = [0.15 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    sigma = base_sigma[:]

    # Success-based adaptation parameters
    success_window = 30
    successes = 0
    trials = 0

    # Restart policy
    last_improve_time = time.time()
    stagnation_limit = max(0.2, min(2.0, 0.15 * max_time))  # seconds without improvement before restart
    restart_prob = 0.02  # small chance to restart anyway

    x = best_x[:]
    fx = best

    while True:
        if time.time() - t0 >= max_time:
            return best

        # Optional restart if stagnating or by chance
        if (time.time() - last_improve_time) > stagnation_limit or random.random() < restart_prob:
            # Random restart, but keep a bias toward best known region by mixing
            xr = rand_point()
            mix = 0.5 * random.random()
            x = [mix * x[i] + (1.0 - mix) * xr[i] for i in range(dim)]
            x = clamp(x)
            fx = evaluate(x)
            # Reset step size to explore
            sigma = base_sigma[:]
            successes = 0
            trials = 0

            if fx < best:
                best = fx
                best_x = x[:]
                last_improve_time = time.time()

        # Propose mutation around current x
        cand = x[:]
        for i in range(dim):
            # Gaussian step via Box-Muller transform (no external libs)
            u1 = random.random()
            u2 = random.random()
            # avoid log(0)
            u1 = max(u1, 1e-12)
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            cand[i] = cand[i] + sigma[i] * z
        cand = clamp(cand)

        fc = evaluate(cand)

        trials += 1
        if fc < fx:
            x, fx = cand, fc
            successes += 1
            if fc < best:
                best = fc
                best_x = cand[:]
                last_improve_time = time.time()

        # Adapt sigma every window using a 1/5 success rule style heuristic
        if trials >= success_window:
            rate = successes / float(trials)
            # If too successful, increase step; if not, decrease.
            if rate > 0.25:
                factor = 1.2
            elif rate < 0.15:
                factor = 0.7
            else:
                factor = 1.0

            for i in range(dim):
                sigma[i] *= factor
                # Keep sigma within reasonable bounds
                min_sigma = 1e-12 + 1e-6 * spans[i]
                max_sigma = 0.5 * spans[i] + 1e-12
                if sigma[i] < min_sigma:
                    sigma[i] = min_sigma
                if sigma[i] > max_sigma:
                    sigma[i] = max_sigma

            successes = 0
            trials = 0
