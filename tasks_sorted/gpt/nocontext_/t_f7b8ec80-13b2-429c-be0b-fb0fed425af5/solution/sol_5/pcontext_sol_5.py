import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like initialization (stratified per-dimension)
      - Local coordinate search
      - Adaptive step-size (success -> enlarge, failure -> shrink)
      - Random restarts with occasional long jumps

    Returns:
      best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()

    # ---- helpers ----
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def eval_f(x):
        # func expects an array-like; keep it as a Python list for no external deps
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def lhs_batch(n):
        # Stratified sampling per dimension, then random permutation
        # (Not a full LHS correlation control, but good coverage and cheap.)
        pts = [[0.0] * dim for _ in range(n)]
        for j in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            for i in range(n):
                u = (perm[i] + random.random()) / n
                pts[i][j] = lows[j] + u * spans[j]
        return pts

    # ---- initialization ----
    # Budget some time for diversified initial sampling
    best_x = None
    best = float("inf")

    # choose initial sample size based on dimension (kept small, time-bounded)
    n0 = max(12, 8 * dim)
    init_points = lhs_batch(n0)

    for x in init_points:
        if time.time() - t0 >= max_time:
            return best
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x[:]

    if best_x is None:
        # fallback (shouldn't happen)
        best_x = rand_point()
        best = eval_f(best_x)

    # ---- main loop: adaptive coordinate + occasional random moves/restarts ----
    x = best_x[:]
    fx = best

    # initial step sizes: 10% of range each dimension (with minimum)
    steps = [0.1 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    min_steps = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    no_improve_iters = 0
    iter_count = 0

    while True:
        if time.time() - t0 >= max_time:
            return best

        iter_count += 1
        improved = False

        # coordinate exploration in random order
        order = list(range(dim))
        random.shuffle(order)

        for j in order:
            if time.time() - t0 >= max_time:
                return best

            step = steps[j]
            if step < min_steps[j]:
                continue

            # try both directions
            cand1 = x[:]
            cand1[j] = clip(cand1[j] + step, j)
            f1 = eval_f(cand1)

            if f1 < fx:
                x, fx = cand1, f1
                improved = True
            else:
                cand2 = x[:]
                cand2[j] = clip(cand2[j] - step, j)
                f2 = eval_f(cand2)
                if f2 < fx:
                    x, fx = cand2, f2
                    improved = True

            # track global best
            if fx < best:
                best = fx
                best_x = x[:]

        # adapt step sizes
        if improved:
            no_improve_iters = 0
            # gently increase steps where possible (promotes faster progress on slopes)
            for j in range(dim):
                steps[j] *= 1.05
                # cap step to full span
                if steps[j] > spans[j]:
                    steps[j] = spans[j]
        else:
            no_improve_iters += 1
            # shrink steps to refine
            for j in range(dim):
                steps[j] *= 0.6
                if steps[j] < min_steps[j]:
                    steps[j] = min_steps[j]

        # occasional perturbation / restart logic
        # - if stuck, either jump around best_x or restart globally
        if no_improve_iters >= 10:
            no_improve_iters = 0

            r = random.random()
            if r < 0.65:
                # local perturbation around best (scaled by current steps)
                x = best_x[:]
                for j in range(dim):
                    # symmetric noise in [-1, 1] scaled by steps
                    x[j] = clip(x[j] + (2.0 * random.random() - 1.0) * max(steps[j], 1e-9), j)
                fx = eval_f(x)
            else:
                # full random restart to escape local minima
                x = rand_point()
                fx = eval_f(x)

            if fx < best:
                best = fx
                best_x = x[:]

        # very rare long jump to diversify even without "stuck" signal
        if iter_count % 50 == 0 and (time.time() - t0) < max_time:
            if random.random() < 0.15:
                cand = rand_point()
                fc = eval_f(cand)
                if fc < best:
                    best = fc
                    best_x = cand[:]
                    x, fx = best_x[:], best

    # (unreachable) return best
    # return best
