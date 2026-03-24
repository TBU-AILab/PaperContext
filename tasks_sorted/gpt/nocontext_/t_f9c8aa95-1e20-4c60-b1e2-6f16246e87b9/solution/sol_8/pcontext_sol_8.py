import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like seeding (stratified per dimension)
    - Local coordinate search (pattern search)
    - Occasional random restarts with adaptive step size

    Args:
        func: callable(params_list_or_array) -> float
        dim: int
        bounds: list of (low, high) pairs, length=dim
        max_time: seconds (int/float)

    Returns:
        best: float, fitness of the best found solution
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --- helpers ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func may accept list or array-like; we pass list (no external deps)
        return float(func(x))

    # Best-so-far
    best_x = None
    best = float("inf")

    # Determine a reasonable evaluation budget for initial seeding
    # (kept small; we are time-limited, not eval-limited)
    seed_n = max(8, min(64, 4 * dim))

    # --- stratified seeding (approx. Latin Hypercube Sampling) ---
    # For each dimension, create stratified bins and permute them.
    # Then combine i-th bin sample across dimensions.
    strata = []
    for j in range(dim):
        n = seed_n
        width = spans[j] / n if n > 0 else 0.0
        bins = []
        for i in range(n):
            a = lows[j] + i * width
            b = lows[j] + (i + 1) * width
            # sample uniformly within bin
            u = random.random()
            bins.append(a + u * (b - a))
        random.shuffle(bins)
        strata.append(bins)

    # Evaluate seeds
    for i in range(seed_n):
        if time.time() >= deadline:
            return best
        x = [strata[j][i] for j in range(dim)]
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # Fallback (should not happen)
        x = rand_point()
        best = eval_f(x)
        best_x = x

    # --- Local search parameters ---
    # Initial step: a fraction of span (handles scaling)
    step0 = [0.2 * s if s > 0 else 1.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # Search state
    x = best_x[:]
    fx = best
    step = step0[:]

    # Restart control
    no_improve_iters = 0
    restart_after = 20 + 5 * dim

    # --- main loop: coordinate pattern search + adaptive restarts ---
    while time.time() < deadline:
        improved = False

        # Try coordinate moves in random dimension order
        order = list(range(dim))
        random.shuffle(order)

        for j in order:
            if time.time() >= deadline:
                return best

            if spans[j] == 0.0:
                continue

            # probe both directions
            for direction in (-1.0, 1.0):
                cand = x[:]
                cand[j] = cand[j] + direction * step[j]
                cand = clamp(cand)
                f_cand = eval_f(cand)

                if f_cand < fx:
                    x, fx = cand, f_cand
                    improved = True
                    if f_cand < best:
                        best, best_x = f_cand, cand[:]
                    break  # move to next dimension
            # continue scanning dimensions regardless

        if improved:
            no_improve_iters = 0
            # Mildly increase steps on success (but keep bounded)
            for j in range(dim):
                step[j] = min(step0[j], step[j] * 1.2)
        else:
            no_improve_iters += 1
            # Reduce steps on failure
            for j in range(dim):
                step[j] *= 0.5
                if step[j] < min_step[j]:
                    step[j] = min_step[j]

        # If stuck for a while, restart near the best point or random point
        if no_improve_iters >= restart_after:
            no_improve_iters = 0

            if time.time() >= deadline:
                return best

            # With higher probability, restart near best; otherwise global random
            if random.random() < 0.75:
                # Gaussian perturbation around best_x with scale based on current step0
                cand = []
                for j in range(dim):
                    sigma = 0.15 * spans[j] if spans[j] > 0 else 1.0
                    # Box-Muller for normal sample (avoid external libs)
                    u1 = max(1e-12, random.random())
                    u2 = random.random()
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                    cand.append(best_x[j] + sigma * z)
                cand = clamp(cand)
            else:
                cand = rand_point()

            f_cand = eval_f(cand)
            if f_cand < fx:
                x, fx = cand, f_cand
            if f_cand < best:
                best, best_x = f_cand, cand[:]

            # Reset step sizes after restart
            step = step0[:]

    return best
