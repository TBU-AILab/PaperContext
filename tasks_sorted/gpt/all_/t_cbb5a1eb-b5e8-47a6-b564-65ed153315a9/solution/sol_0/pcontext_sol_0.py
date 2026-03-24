import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded global optimization using a self-contained hybrid of:
      - Latin-hypercube-like stratified initialization
      - (1+1)-ES style local search with adaptive step size (success rule)
      - Occasional random/global restarts

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/sequence of length dim
    dim : int
        Dimensionality
    bounds : list of (low, high)
        Bounds per dimension
    max_time : int/float
        Time limit in seconds

    Returns
    -------
    best : float
        Best (minimum) objective value found
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [hi - lo for lo, hi in zip(lows, highs)]

    def clip_to_bounds(x):
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if x[i] < lo:
                x[i] = lo
            elif x[i] > hi:
                x[i] = hi
        return x

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Normal(0,1) without numpy
    def randn():
        # Box-Muller
        u1 = random.random()
        u2 = random.random()
        u1 = max(u1, 1e-12)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def evaluate(x):
        # func expects "array of input parameter values"; a Python list is acceptable
        # for most objective implementations; keep it self-contained.
        return float(func(x))

    # --- initial stratified sampling (LHS-like) ---
    # Build m stratified samples per dimension then permute to form m points.
    # Choose m based on dim and time budget conservatively.
    # (If time is tiny, still produces at least 1-2 evaluations.)
    m = max(4, min(40, int(8 + 2 * math.sqrt(max(dim, 1)))))

    strata = []
    for i in range(dim):
        lo = lows[i]
        span = spans[i] if spans[i] > 0 else 1.0
        # m strata in [0,1): [k/m,(k+1)/m)
        vals = [lo + ((k + random.random()) / m) * span for k in range(m)]
        random.shuffle(vals)
        strata.append(vals)

    best = float("inf")
    best_x = None

    # Evaluate initial points
    for k in range(m):
        if time.time() >= deadline:
            return best
        x = [strata[i][k] for i in range(dim)]
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

    # If we somehow didn't evaluate anything (extreme time constraints)
    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]

    # --- adaptive local search with restarts ---
    # Step sizes are relative to span; start moderately, adapt with success rate.
    # Use a (1+1)-ES: propose x' = x + sigma * N(0,1) per component.
    # Adapt sigma using a simple success rule.
    x = best_x[:]
    fx = best

    sigma = 0.2  # relative scale factor
    sigma_min = 1e-12
    sigma_max = 1.0

    # Track success over a sliding window
    window = max(10, 5 * dim)
    successes = 0
    trials = 0

    # Restart control
    no_improve = 0
    restart_after = 60 + 15 * dim

    while time.time() < deadline:
        # Occasionally do a global jump/restart to avoid stagnation
        if no_improve >= restart_after:
            # restart from best with random perturbation or random point
            if random.random() < 0.7:
                x = best_x[:]
                # broaden a bit on restart
                sigma = min(sigma_max, max(0.2, sigma * 2.0))
            else:
                x = rand_uniform_point()
                sigma = 0.3
            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x[:]
            no_improve = 0
            successes = 0
            trials = 0
            continue

        # Propose neighbor
        x_new = x[:]
        # scale by span so dimensions with bigger ranges move more
        for i in range(dim):
            step = sigma * spans[i] * randn()
            x_new[i] += step
        clip_to_bounds(x_new)

        f_new = evaluate(x_new)

        trials += 1
        improved = (f_new <= fx)

        # Accept if improves current; also update global best
        if improved:
            x, fx = x_new, f_new
            successes += 1
            no_improve = 0
            if f_new < best:
                best, best_x = f_new, x_new[:]
        else:
            no_improve += 1

        # Adapt sigma by success rate every "window" trials
        if trials >= window:
            rate = successes / float(trials)
            # target success rate ~ 1/5
            if rate > 0.25:
                sigma *= 1.25
            elif rate < 0.15:
                sigma *= 0.8
            sigma = min(sigma_max, max(sigma_min, sigma))
            successes = 0
            trials = 0

    return best
