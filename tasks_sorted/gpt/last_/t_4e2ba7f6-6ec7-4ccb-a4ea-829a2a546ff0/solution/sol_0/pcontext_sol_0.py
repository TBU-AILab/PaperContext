import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      1) Latin-hypercube-like stratified initialization (per-dimension shuffles)
      2) Adaptive local search around the current best (Gaussian steps)
      3) Occasional global re-samples (diversification)
      4) Lightweight step-size control based on success rate
    Returns: best fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        return float(func(x))

    # --- initial stratified sampling (cheap LHS-style) ---
    # Choose an initialization budget relative to dim but keep it modest.
    init_n = max(8, min(40 * dim, 400))
    # Build per-dimension shuffled strata indices 0..init_n-1
    perm = [list(range(init_n)) for _ in range(dim)]
    for j in range(dim):
        random.shuffle(perm[j])

    best_x = None
    best = float("inf")

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for j in range(dim):
            # sample uniformly within stratum
            u = (perm[j][k] + random.random()) / init_n
            x.append(lows[j] + u * spans[j])
        f = eval_f(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        # Fallback (shouldn't happen)
        x = rand_uniform_vec()
        best = eval_f(x)
        best_x = x

    # --- adaptive local/global search loop ---
    # Step sizes start as a fraction of the domain span.
    sigmas = [0.15 * s if s > 0 else 1.0 for s in spans]
    min_sigma = [1e-12 * (s if s > 0 else 1.0) for s in spans]
    max_sigma = [0.50 * (s if s > 0 else 1.0) for s in spans]

    # Control parameters
    p_global = 0.15  # probability of global exploration
    window = 30      # success-rate adaptation window
    success = 0
    trials = 0

    # A secondary "temperature" for occasional large jumps around best
    jump_scale = 0.35

    while time.time() < deadline:
        trials += 1

        # Decide exploration mode
        r = random.random()
        if r < p_global:
            # Global sample
            cand = rand_uniform_vec()
        else:
            # Local sample around best with per-dimension Gaussian steps
            cand = []
            for j in range(dim):
                # random.gauss(mu, sigma)
                step = random.gauss(0.0, sigmas[j])
                cand.append(clamp(best_x[j] + step, j))

            # Occasionally try a bigger jump around best to escape local minima
            if random.random() < 0.10:
                for j in range(dim):
                    step = random.gauss(0.0, jump_scale * spans[j])
                    cand[j] = clamp(cand[j] + step, j)

        f = eval_f(cand)

        if f < best:
            best = f
            best_x = cand
            success += 1

        # Adapt step sizes based on recent success rate
        if trials >= window:
            rate = success / float(trials)
            # If too few improvements, shrink steps; if many, expand slightly
            if rate < 0.15:
                factor = 0.82
                p_global = min(0.35, p_global + 0.02)  # diversify a bit more
            elif rate > 0.30:
                factor = 1.12
                p_global = max(0.05, p_global - 0.01)  # exploit more
            else:
                factor = 1.0

            if factor != 1.0:
                for j in range(dim):
                    s = sigmas[j] * factor
                    if s < min_sigma[j]:
                        s = min_sigma[j]
                    elif s > max_sigma[j]:
                        s = max_sigma[j]
                    sigmas[j] = s

            success = 0
            trials = 0

    return best
