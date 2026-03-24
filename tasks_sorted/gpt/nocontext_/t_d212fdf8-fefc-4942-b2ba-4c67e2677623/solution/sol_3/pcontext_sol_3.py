import time
import random
import math

def run(func, dim, bounds, max_time):
    # Self-contained hybrid optimizer:
    # - Latin-hypercube-ish initial sampling
    # - Local coordinate search (pattern search)
    # - Occasional random restarts + shrinking step sizes
    #
    # Returns best (minimum) fitness found within max_time.

    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lo = [bounds[i][0] for i in range(dim)]
    hi = [bounds[i][1] for i in range(dim)]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clip(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    def eval_point(x):
        # func expects an array-like; keep it list to avoid numpy dependency
        return float(func(x))

    # --- best-so-far ---
    best_x = None
    best = float("inf")

    # --- initial sampling (stratified per-dimension) ---
    # Choose number of strata based on dimension and time budget (lightweight).
    # At minimum 8; at maximum 64 to keep overhead low.
    strata = max(8, min(64, int(12 + 6 * math.log(dim + 1, 2))))

    # Precompute per-dimension shuffled strata indices for quasi-LHS
    strata_idx = []
    for i in range(dim):
        idx = list(range(strata))
        random.shuffle(idx)
        strata_idx.append(idx)

    # Evaluate initial candidates
    s = 0
    while time.time() < deadline and s < strata:
        x = []
        for i in range(dim):
            a = strata_idx[i][s]
            # sample within stratum [a/strata, (a+1)/strata)
            u = (a + random.random()) / strata
            x.append(lo[i] + u * span[i])
        f = eval_point(x)
        if f < best:
            best, best_x = f, x
        s += 1

    if best_x is None:
        # max_time might be extremely small
        x = rand_point()
        return eval_point(x)

    # --- local coordinate/pattern search around incumbent ---
    # Step sizes start as a fraction of range; shrink on failure.
    step = [0.25 * span[i] if span[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (span[i] if span[i] > 0 else 1.0) for i in range(dim)]

    # Restart control
    no_improve = 0
    restart_after = 12 * dim + 50

    # Main loop
    while time.time() < deadline:
        improved = False

        # Randomize coordinate order to avoid bias
        order = list(range(dim))
        random.shuffle(order)

        # Coordinate exploration (+/- step)
        for i in order:
            if time.time() >= deadline:
                break
            if step[i] <= min_step[i]:
                continue

            xi = best_x[i]

            # try plus
            x_try = best_x[:]  # copy
            x_try[i] = xi + step[i]
            clip(x_try)
            f_try = eval_point(x_try)
            if f_try < best:
                best, best_x = f_try, x_try
                improved = True
                continue

            # try minus
            x_try = best_x[:]
            x_try[i] = xi - step[i]
            clip(x_try)
            f_try = eval_point(x_try)
            if f_try < best:
                best, best_x = f_try, x_try
                improved = True
                continue

        # If improved, attempt a short "pattern move" in the improvement direction
        if time.time() < deadline and improved:
            no_improve = 0
            # Pattern move: extrapolate from a slightly perturbed base
            # Make a direction by sampling a nearby point and moving away if better.
            base = best_x[:]
            # small random direction scaled by current steps
            direction = []
            for i in range(dim):
                r = (random.random() * 2.0 - 1.0)
                direction.append(r * step[i])
            x_pat = [base[i] + direction[i] for i in range(dim)]
            clip(x_pat)
            f_pat = eval_point(x_pat)
            if f_pat < best:
                best, best_x = f_pat, x_pat
            continue

        # No improvement: shrink steps
        no_improve += 1
        for i in range(dim):
            step[i] *= 0.5

        # If steps too small everywhere, restart around global best with moderate steps
        if no_improve >= restart_after or all(step[i] <= min_step[i] for i in range(dim)):
            no_improve = 0

            # With some probability, global random restart; otherwise local jitter
            if random.random() < 0.35:
                x0 = rand_point()
            else:
                # local restart around best_x with jitter
                x0 = best_x[:]
                for i in range(dim):
                    jitter = (random.random() * 2.0 - 1.0) * 0.15 * span[i]
                    x0[i] += jitter
                clip(x0)

            f0 = eval_point(x0)
            if f0 < best:
                best, best_x = f0, x0

            # reset steps to a moderate size
            for i in range(dim):
                step[i] = max(0.10 * span[i], min_step[i] * 10.0)

    return best
