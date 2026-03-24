import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like seeding (stratified per dimension)
    - Local coordinate search with shrinking step (pattern search)
    - Occasional random restarts / perturbations to escape local minima

    Returns:
        best (float): best (minimum) fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # --------- helpers ----------
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func expects "an array-like"; keep it plain list to avoid external libs
        return float(func(x))

    # --------- initialization ----------
    best = float("inf")
    best_x = None

    # Seed count chosen to be small and fast, scales mildly with dim.
    # (Also caps to avoid spending too much time on seeding.)
    seed_n = max(8, 6 * dim)
    seed_n = min(seed_n, 60)

    # Stratified sampling per dimension (LHS-ish):
    # For each dimension i, create seed_n strata and shuffle them independently.
    strata = []
    for i in range(dim):
        idx = list(range(seed_n))
        random.shuffle(idx)
        strata.append(idx)

    for k in range(seed_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # pick a point uniformly within stratum
            u = (strata[i][k] + random.random()) / seed_n
            x.append(lows[i] + u * spans[i])
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # degenerate case; just return if no evaluations happened
        return best

    # --------- main loop: pattern search + restarts ----------
    # Initial step sizes as fraction of span
    base_step = [0.2 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Search state
    x = best_x[:]
    fx = best
    step = base_step[:]

    # Control parameters
    shrink = 0.65          # shrink factor when no improvement
    expand = 1.15          # expand slightly after improvement
    restart_prob = 0.03    # occasional random restart
    perturb_prob = 0.10    # small random perturbation around incumbent
    max_step_mult = 0.5    # don't let step explode beyond 50% span

    while time.time() < deadline:
        # Occasionally restart from a new random point to avoid stagnation
        if random.random() < restart_prob:
            xr = rand_point()
            fr = eval_f(xr)
            if fr < best:
                best, best_x = fr, xr[:]
                x, fx = xr[:], fr
            else:
                # still allow switching with small probability (diversification)
                if random.random() < 0.15:
                    x, fx = xr[:], fr
            # reset step after restart
            step = base_step[:]

        # Occasional small perturbation around current point (cheap escape)
        if random.random() < perturb_prob:
            xp = x[:]
            for i in range(dim):
                if spans[i] > 0:
                    # gaussian-like using sum of uniforms (no external libs)
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random() - 3.0)
                    xp[i] += 0.15 * step[i] * g
            clip(xp)
            fp = eval_f(xp)
            if fp < fx:
                x, fx = xp, fp
                if fp < best:
                    best, best_x = fp, xp[:]

        improved = False

        # Coordinate-wise exploratory moves (pattern search)
        order = list(range(dim))
        random.shuffle(order)
        for i in order:
            if time.time() >= deadline:
                return best

            si = step[i]
            if si <= min_step[i] or spans[i] == 0:
                continue

            # Try +step and -step
            cand_best_x = None
            cand_best_f = fx

            for direction in (+1.0, -1.0):
                xc = x[:]
                xc[i] += direction * si
                clip(xc)
                fc = eval_f(xc)
                if fc < cand_best_f:
                    cand_best_f = fc
                    cand_best_x = xc

            if cand_best_x is not None:
                x, fx = cand_best_x, cand_best_f
                improved = True
                if fx < best:
                    best, best_x = fx, x[:]

        if improved:
            # Slightly expand step sizes to move faster while improving
            for i in range(dim):
                if spans[i] > 0:
                    step[i] = min(step[i] * expand, max_step_mult * spans[i])
        else:
            # No progress: shrink steps; if too small, jump somewhere new
            too_small = True
            for i in range(dim):
                if spans[i] > 0:
                    step[i] *= shrink
                    if step[i] > min_step[i]:
                        too_small = False

            if too_small:
                # If we've shrunk too far, restart near global best (or random)
                if random.random() < 0.7 and best_x is not None:
                    x = best_x[:]
                    for i in range(dim):
                        if spans[i] > 0:
                            # local restart radius
                            r = 0.25 * spans[i]
                            x[i] += (2.0 * random.random() - 1.0) * r
                    clip(x)
                else:
                    x = rand_point()
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x[:]
                step = base_step[:]

    return best
