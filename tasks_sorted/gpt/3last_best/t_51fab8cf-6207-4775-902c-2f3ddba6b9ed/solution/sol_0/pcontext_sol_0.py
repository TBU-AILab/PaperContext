import random
import math
import time

def run(func, dim, bounds, max_time):
    # Time control
    start = time.time()
    deadline = start + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; keep it plain list (no numpy dependency).
        return float(func(x))

    # --- initialization: a few random samples ---
    best_x = None
    best = float("inf")

    # Budget a small initial batch
    init_n = max(8, 4 * dim)
    for _ in range(init_n):
        if time.time() >= deadline:
            return best
        x = rand_vec()
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    # If func is extremely fast, we can do more sophisticated search.
    # Strategy: adaptive coordinate search + occasional random restarts.
    #
    # Step sizes start as a fraction of the range and shrink on stagnation.
    step = [0.2 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Stagnation / restart controls
    no_improve = 0
    max_no_improve = 20 * max(1, dim)
    restart_prob = 0.05  # occasional diversification

    # Main loop
    while time.time() < deadline:
        # Random restart sometimes (keeps from getting stuck)
        if best_x is None or random.random() < restart_prob:
            x = rand_vec()
        else:
            x = best_x[:]  # start from incumbent

        improved = False

        # Explore coordinates in random order
        idxs = list(range(dim))
        random.shuffle(idxs)

        for i in idxs:
            if time.time() >= deadline:
                return best

            if step[i] <= min_step[i]:
                continue

            # Try positive and negative moves
            for direction in (1.0, -1.0):
                if time.time() >= deadline:
                    return best

                cand = x[:]
                cand[i] = clip(cand[i] + direction * step[i], lows[i], highs[i])

                # If clipping makes it identical, skip
                if cand[i] == x[i]:
                    continue

                f_cand = eval_f(cand)
                if f_cand < best:
                    best = f_cand
                    best_x = cand
                    x = cand
                    improved = True
                    break  # move to next coordinate
            # continue to next coordinate

        if improved:
            no_improve = 0
            # Mildly increase step sizes when we are making progress
            for i in range(dim):
                step[i] = min(step[i] * 1.05, spans[i])
        else:
            no_improve += 1
            # Shrink steps when stuck
            for i in range(dim):
                step[i] = max(step[i] * 0.5, min_step[i])

            # If heavily stagnated, do a stronger restart by resetting step
            if no_improve >= max_no_improve:
                no_improve = 0
                # Reset steps (but not all the way) and force restart behavior
                for i in range(dim):
                    step[i] = max(step[i], 0.1 * spans[i])
                # temporarily increase exploration
                restart_prob = min(0.25, restart_prob * 1.5)
        # Slowly anneal restart probability back down
        restart_prob = max(0.02, restart_prob * 0.995)

    return best
