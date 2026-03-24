import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid strategy:
    - Latin-hypercube-like seeding (stratified random per-dimension)
    - Local refinement with adaptive coordinate search
    - Occasional random restarts to escape local minima

    Args:
        func: callable(list_or_array_like)-> float
        dim: int
        bounds: list of (low, high) pairs, length = dim
        max_time: seconds (int/float)

    Returns:
        best: best (minimum) fitness found (float)
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
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

    def evaluate(x):
        # func in the prompt mentions np.array, but we avoid numpy;
        # most objective functions will accept a plain list as well.
        return float(func(x))

    # --- initialization: stratified random samples (LHS-like) ---
    # Choose sample count based on dimension and time budget.
    # Keep it modest so we leave time for local search.
    # If max_time is tiny, fall back to a few samples.
    base = 8 * dim
    extra = int(20 * math.log(dim + 2))
    n_init = max(10, min(200, base + extra))

    # Build per-dimension shuffled strata
    strata = []
    for d in range(dim):
        idx = list(range(n_init))
        random.shuffle(idx)
        strata.append(idx)

    best_x = None
    best = float("inf")

    # Evaluate initial points
    for k in range(n_init):
        if time.time() >= deadline:
            return best
        x = []
        for d in range(dim):
            # sample uniformly within stratum
            u = (strata[d][k] + random.random()) / n_init
            x.append(lows[d] + u * spans[d])
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        # Extremely unlikely, but safe
        best_x = rand_uniform_point()
        best = evaluate(best_x)

    # --- local search: adaptive coordinate pattern search ---
    # Start with step sizes proportional to bounds
    step = [0.25 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Restart parameters
    # After a number of consecutive non-improving step reductions, restart.
    no_improve_rounds = 0
    max_no_improve_rounds = 12

    # To reduce bias, randomize coordinate order each round
    while time.time() < deadline:
        improved = False

        order = list(range(dim))
        random.shuffle(order)

        x0 = best_x

        # Try +/- move in each coordinate
        for i in order:
            if time.time() >= deadline:
                return best

            if step[i] <= min_step[i]:
                continue

            # positive move
            xp = list(x0)
            xp[i] += step[i]
            xp = clamp(xp)
            fp = evaluate(xp)
            if fp < best:
                best = fp
                best_x = xp
                x0 = best_x
                improved = True
                continue

            # negative move
            xn = list(x0)
            xn[i] -= step[i]
            xn = clamp(xn)
            fn = evaluate(xn)
            if fn < best:
                best = fn
                best_x = xn
                x0 = best_x
                improved = True
                continue

        if improved:
            # If we are improving, gently increase steps to move faster.
            for i in range(dim):
                step[i] *= 1.2
                # Cap step to span to avoid wasted evaluations
                if step[i] > spans[i]:
                    step[i] = spans[i]
            no_improve_rounds = 0
        else:
            # If no improvement, shrink steps
            for i in range(dim):
                step[i] *= 0.5
            no_improve_rounds += 1

        # If we're stuck (steps small / many failures), do a randomized restart near best
        # plus occasional full random restart.
        if no_improve_rounds >= max_no_improve_rounds:
            if time.time() >= deadline:
                return best

            # 70%: local perturbation around best, 30%: global restart
            if random.random() < 0.7:
                # local perturbation scale based on current step, with floor based on span
                x = list(best_x)
                for i in range(dim):
                    scale = max(step[i], 0.02 * spans[i])
                    x[i] += (random.random() * 2.0 - 1.0) * scale
                x = clamp(x)
            else:
                x = rand_uniform_point()

            f = evaluate(x)
            if f < best:
                best = f
                best_x = x

            # Reset steps after restart
            step = [0.25 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
            no_improve_rounds = 0

    return best
