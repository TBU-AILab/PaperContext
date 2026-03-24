import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like stratified initialization
      - Local coordinate search (pattern search)
      - Occasional adaptive random restarts
    No external libraries required.

    Returns:
        best (float): best (minimum) fitness value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        lo, hi = lows[i], highs[i]
        if x < lo: return lo
        if x > hi: return hi
        return x

    def eval_f(x):
        # Ensure plain list; user func expects array-like
        return float(func(list(x)))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # --- quick path if time is extremely small ---
    if max_time <= 0:
        return float("inf")

    # --- Stratified initial sampling (LHS-like) ---
    # Build K candidates where each dimension is stratified into K bins, then shuffled.
    # This gives better coverage than pure random for the initial phase.
    # Choose K based on time and dimension; keep it modest.
    K = max(4, min(30, int(8 + 2 * math.sqrt(dim))))
    # Create permutations per dimension
    perms = []
    for i in range(dim):
        idxs = list(range(K))
        random.shuffle(idxs)
        perms.append(idxs)

    best_x = None
    best = float("inf")

    for k in range(K):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # sample within the stratified bin
            bin_idx = perms[i][k]
            u = (bin_idx + random.random()) / K
            x.append(lows[i] + u * spans[i])
        f = eval_f(x)
        if f < best:
            best = f
            best_x = x

    # If for some reason we didn't evaluate anything yet
    if best_x is None:
        x = rand_point()
        best_x = x
        best = eval_f(x)

    # --- Local search: adaptive coordinate/pattern search ---
    # Step size starts as a fraction of span and shrinks on stalls.
    # Occasional random restart around the best (and sometimes global) to escape local minima.
    step = [0.2 * s if s > 0 else 0.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # Counters controlling restarts / shrinkage
    no_improve_iters = 0
    shrink_factor = 0.5
    expand_factor = 1.2

    # To reduce bias, we will randomize coordinate order each sweep
    while time.time() < deadline:
        improved = False

        # Coordinate order
        coords = list(range(dim))
        random.shuffle(coords)

        # Try coordinate moves around current best_x
        for i in coords:
            if time.time() >= deadline:
                return best

            if step[i] <= min_step[i]:
                continue

            x0 = best_x

            # Try positive and negative steps (with a tiny random scale to avoid cycles)
            sc = 0.8 + 0.4 * random.random()
            delta = step[i] * sc

            # + move
            xp = list(x0)
            xp[i] = clamp(xp[i] + delta, i)
            fp = eval_f(xp)
            if fp < best:
                best = fp
                best_x = xp
                improved = True
                step[i] *= expand_factor
                continue  # move to next coordinate from new point

            # - move
            xn = list(x0)
            xn[i] = clamp(xn[i] - delta, i)
            fn = eval_f(xn)
            if fn < best:
                best = fn
                best_x = xn
                improved = True
                step[i] *= expand_factor
                continue

            # No improvement on this coordinate: slightly shrink that coordinate step
            step[i] *= 0.9

        if improved:
            no_improve_iters = 0
            continue

        no_improve_iters += 1

        # If stalled, shrink all steps
        if no_improve_iters % 5 == 0:
            for i in range(dim):
                step[i] *= shrink_factor

        # Random restart logic:
        # - Mostly sample in a shrinking neighborhood around current best
        # - Occasionally sample globally to escape deep local minima
        if no_improve_iters >= 3:
            if time.time() >= deadline:
                return best

            # Probability of global restart increases with longer stalls
            p_global = min(0.35, 0.05 + 0.02 * no_improve_iters)

            if random.random() < p_global:
                x = rand_point()
            else:
                # Local perturbation around best_x
                x = list(best_x)
                for i in range(dim):
                    # Neighborhood scale tied to current step and span
                    neigh = max(step[i], 0.02 * spans[i])
                    r = (random.random() * 2 - 1)  # [-1, 1]
                    x[i] = clamp(x[i] + r * neigh, i)

            f = eval_f(x)
            if f < best:
                best = f
                best_x = x
                # modestly increase steps after finding better basin
                for i in range(dim):
                    step[i] = max(step[i], 0.05 * spans[i])
                no_improve_iters = 0
            else:
                # If restart didn't help, keep shrinking over time
                for i in range(dim):
                    step[i] *= 0.95

        # If all steps are tiny, do occasional global sampling until time ends
        if all(step[i] <= min_step[i] for i in range(dim)):
            if time.time() >= deadline:
                return best
            x = rand_point()
            f = eval_f(x)
            if f < best:
                best = f
                best_x = x
                # reset steps a bit
                step = [0.1 * s if s > 0 else 0.0 for s in spans]

    return best
