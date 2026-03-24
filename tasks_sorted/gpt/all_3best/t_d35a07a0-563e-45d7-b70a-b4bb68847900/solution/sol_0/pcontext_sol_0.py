import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a self-contained
    hybrid of:
      - Latin-hypercube-like stratified initialization
      - (1+1)-ES style local search with 1/5th success rule step adaptation
      - occasional random/global restarts

    Returns:
      best (float): best objective value found within max_time
    """
    # ---- helpers ----
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; keep it simple (list)
        try:
            val = float(func(x))
        except Exception:
            # If evaluation fails, treat as very bad
            val = float("inf")
        if math.isnan(val):
            val = float("inf")
        return val

    # distance scale per dimension (for step sizing)
    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # ---- initial sampling (stratified) ----
    # Build a small initial design; sized to be quick.
    init_n = max(8, 4 * dim)
    # For each dimension, create shuffled strata points in [0,1]
    strata = []
    for j in range(dim):
        perm = list(range(init_n))
        random.shuffle(perm)
        strata.append(perm)

    best = float("inf")
    best_x = None

    for i in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for j in range(dim):
            lo, hi = bounds[j]
            u = (strata[j][i] + random.random()) / init_n  # in (0,1)
            x.append(lo + u * (hi - lo))
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        # fallback
        best_x = rand_uniform_vec()
        best = evaluate(best_x)

    # ---- local search with adaptive step + restarts ----
    # Initial step as fraction of spans
    sigma = [0.15 * s for s in span]  # per-dimension step scale
    min_sigma = [1e-12 * s for s in span]
    max_sigma = [0.5 * s for s in span]

    x = list(best_x)
    fx = float(best)

    # 1/5th success rule bookkeeping
    window = 20
    success = 0
    trials = 0

    # restart controls
    last_improve_t = time.time()
    stagnation_seconds = max(0.25, 0.15 * max_time)  # allow some time before restarting
    restart_prob = 0.02  # small chance each iteration to jump globally

    while time.time() < deadline:
        # occasional random restart (diversification)
        if random.random() < restart_prob or (time.time() - last_improve_t) > stagnation_seconds:
            x = rand_uniform_vec()
            fx = evaluate(x)
            if fx < best:
                best = fx
                best_x = list(x)
                last_improve_t = time.time()
            # reset step a bit on restart
            sigma = [0.2 * s for s in span]
            success = 0
            trials = 0

        # propose a new point around current x using gaussian-like perturbation
        # (Box-Muller)
        xn = []
        for j in range(dim):
            # gaussian sample
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(max(1e-16, u1))) * math.cos(2.0 * math.pi * u2)
            step = z * sigma[j]
            lo, hi = bounds[j]
            xn.append(clamp(x[j] + step, lo, hi))

        fn = evaluate(xn)

        trials += 1
        if fn <= fx:  # accept
            x, fx = xn, fn
            success += 1
            if fn < best:
                best = fn
                best_x = list(xn)
                last_improve_t = time.time()

        # Adapt step sizes every window using 1/5 success rule
        if trials >= window:
            rate = success / float(trials)
            # if success rate > 0.2 increase, else decrease
            if rate > 0.2:
                factor = 1.25
            else:
                factor = 0.82
            sigma = [clamp(s * factor, min_sigma[j], max_sigma[j]) for j, s in enumerate(sigma)]
            success = 0
            trials = 0

    return best
