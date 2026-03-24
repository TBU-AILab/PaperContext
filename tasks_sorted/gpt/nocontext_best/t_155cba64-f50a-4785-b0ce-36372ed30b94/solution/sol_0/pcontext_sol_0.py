import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: Hybrid of
      - Latin-hypercube-like initialization (stratified per-dimension)
      - Local improvement via coordinate search (pattern search)
      - Occasional random-restart sampling
    Works without external libraries; 'func' is called with a Python list of floats.
    Returns best (minimum) fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + max_time

    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def evaluate(x):
        # Defensive: ensure inside bounds and numeric output
        xx = [clip(x[i], bounds[i][0], bounds[i][1]) for i in range(dim)]
        val = func(xx)
        return float(val), xx

    # --- Initialization: stratified sampling per dimension (LHS-like) ---
    # Choose n based on time/dimension; keep modest but useful.
    # If max_time is tiny, still do at least a few samples.
    n_init = max(8, min(60, 10 * dim))
    # Precompute strata positions for each dimension
    strata = []
    for j in range(dim):
        lo, hi = bounds[j]
        width = hi - lo
        # Random permutation of strata indices
        perm = list(range(n_init))
        random.shuffle(perm)
        strata.append((lo, width, perm))

    best = float("inf")
    best_x = None

    for i in range(n_init):
        if time.time() >= deadline:
            return best
        x = []
        for j in range(dim):
            lo, width, perm = strata[j]
            # pick a point uniformly in stratum perm[i]
            a = (perm[i] + random.random()) / n_init
            x.append(lo + a * width)
        val, xx = evaluate(x)
        if val < best:
            best, best_x = val, xx

    if best_x is None:
        # fallback if func failed somehow
        val, xx = evaluate(rand_in_bounds())
        best, best_x = val, xx

    # --- Local search: coordinate pattern search with adaptive step sizes ---
    # Initial step: fraction of range per dimension
    steps = []
    for j in range(dim):
        lo, hi = bounds[j]
        steps.append(0.25 * (hi - lo) if hi > lo else 1.0)

    # Parameters controlling exploration/exploitation
    shrink = 0.5
    grow = 1.2
    min_step_frac = 1e-9
    restart_prob = 0.05  # occasionally try a new random point

    # Precompute absolute minimum steps per dimension
    min_steps = []
    for j in range(dim):
        lo, hi = bounds[j]
        r = (hi - lo)
        min_steps.append(max(min_step_frac * (r if r > 0 else 1.0), 1e-12))

    x = best_x[:]
    no_improve_rounds = 0

    while time.time() < deadline:
        # occasional restart to escape local minima
        if random.random() < restart_prob and time.time() < deadline:
            cand = rand_in_bounds()
            val, cand = evaluate(cand)
            if val < best:
                best, best_x = val, cand
                x = cand[:]
                # reset steps a bit around new point
                for j in range(dim):
                    lo, hi = bounds[j]
                    steps[j] = max(0.25 * (hi - lo), min_steps[j])
            continue

        improved = False

        # Randomize coordinate order each round
        order = list(range(dim))
        random.shuffle(order)

        for j in order:
            if time.time() >= deadline:
                return best

            step = steps[j]
            if step < min_steps[j]:
                continue

            # Try positive and negative moves
            for direction in (1.0, -1.0):
                if time.time() >= deadline:
                    return best

                trial = x[:]
                trial[j] = clip(trial[j] + direction * step, bounds[j][0], bounds[j][1])
                val, trial = evaluate(trial)

                if val < best:
                    best, best_x = val, trial
                    x = trial[:]  # move to improved point
                    steps[j] *= grow
                    improved = True
                    break  # move to next coordinate
            else:
                # no break => no improvement in either direction
                steps[j] *= shrink

        if improved:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        # If we're stuck, reduce all steps; if very stuck, jump near best_x
        if no_improve_rounds >= 5:
            for j in range(dim):
                steps[j] = max(steps[j] * shrink, min_steps[j])
            no_improve_rounds = 0

            # small perturbation around current best
            if time.time() < deadline:
                pert = best_x[:]
                for j in range(dim):
                    lo, hi = bounds[j]
                    r = hi - lo
                    # perturb up to 1% of range (or min step)
                    amp = max(0.01 * r, min_steps[j])
                    pert[j] = clip(pert[j] + random.uniform(-amp, amp), lo, hi)
                val, pert = evaluate(pert)
                if val < best:
                    best, best_x = val, pert
                    x = pert[:]

    return best
