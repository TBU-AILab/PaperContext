import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      1) Latin-hypercube-like stratified sampling for initialization
      2) Adaptive local search (coordinate + random directions)
      3) Occasional restarts (to escape local minima)

    Returns:
      best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---- helpers ----
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        # In-place clip to bounds
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; list is acceptable for most user funcs
        return float(func(x))

    # ---- initialization: stratified samples (LHS-style) ----
    # Choose number of initial samples based on time budget and dimension (lightweight).
    # Kept conservative so we leave time for exploitation.
    init_n = max(8, 6 * dim)

    best_x = None
    best = float("inf")

    # Build LHS buckets per dimension
    # We generate init_n points: for each dim, permute strata indices.
    strata = list(range(init_n))
    perms = []
    for _ in range(dim):
        p = strata[:]
        random.shuffle(p)
        perms.append(p)

    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            # sample uniformly within the k-th stratum for dimension i
            u = (perms[i][k] + random.random()) / init_n
            x[i] = lows[i] + u * spans[i]
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x

    # If init did not run, seed with random
    if best_x is None:
        best_x = rand_uniform_vec()
        best = evaluate(best_x)

    # ---- local search parameters ----
    # Step size per dimension (start relatively large, then shrink on stagnation)
    step = [0.25 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Stagnation tracking
    no_improve = 0
    shrink_every = 25  # shrink step after this many non-improving proposals
    restart_every = 250  # restart around best after more stagnation

    # ---- main loop ----
    # Mix of coordinate moves and random-direction moves.
    while time.time() < deadline:
        improved = False

        # Choose move type: coordinate (more often) vs random direction
        if random.random() < 0.7:
            # Coordinate move: pick one dimension, try +/- step
            i = random.randrange(dim)

            # Try both directions (random order)
            dirs = [-1.0, 1.0]
            if random.random() < 0.5:
                dirs.reverse()

            for sgn in dirs:
                if time.time() >= deadline:
                    return best
                if step[i] <= min_step[i]:
                    continue
                cand = best_x[:]  # start from current best
                cand[i] = cand[i] + sgn * step[i]
                clip(cand)
                f = evaluate(cand)
                if f < best:
                    best = f
                    best_x = cand
                    improved = True
                    break
        else:
            # Random direction move: perturb multiple dimensions
            if time.time() >= deadline:
                return best

            # Build random direction vector
            # Use Gaussian-like via sum of uniforms to avoid importing libraries
            dir_vec = [0.0] * dim
            norm2 = 0.0
            for i in range(dim):
                # approx N(0,1): sum of 12 uniforms - 6
                g = sum(random.random() for _ in range(12)) - 6.0
                dir_vec[i] = g
                norm2 += g * g

            if norm2 > 0:
                inv_norm = 1.0 / math.sqrt(norm2)
                dir_vec = [d * inv_norm for d in dir_vec]

            # Scale step by average step size
            avg_step = sum(step) / float(dim) if dim > 0 else 1.0
            scale = avg_step * (0.25 + 0.75 * random.random())

            cand = [best_x[i] + scale * dir_vec[i] for i in range(dim)]
            clip(cand)
            f = evaluate(cand)
            if f < best:
                best = f
                best_x = cand
                improved = True

        if improved:
            no_improve = 0
            # Mildly expand steps in dimensions with room, encouraging progress
            for i in range(dim):
                step[i] = min(step[i] * 1.05, spans[i] if spans[i] > 0 else step[i])
        else:
            no_improve += 1

            # Periodic shrink to refine search
            if no_improve % shrink_every == 0:
                for i in range(dim):
                    step[i] = max(step[i] * 0.5, min_step[i])

            # Occasional restart: sample near best (or globally) to escape local minima
            if no_improve % restart_every == 0:
                if time.time() >= deadline:
                    return best

                # With some probability do a global restart, else a local one
                if random.random() < 0.35:
                    x = rand_uniform_vec()
                else:
                    # local restart around best with current step
                    x = [best_x[i] + (random.random() * 2.0 - 1.0) * step[i] for i in range(dim)]
                    clip(x)
                f = evaluate(x)
                if f < best:
                    best = f
                    best_x = x
                    no_improve = 0

    return best
