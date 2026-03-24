import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like initial sampling
    - Local improvement (pattern search)
    - Occasional random/global restarts
    Requires: func(params)->float, dim:int, bounds:list[(low,high)], max_time:seconds
    Returns: best (float) fitness found
    """
    t0 = time.time()

    def time_left():
        return max_time - (time.time() - t0)

    # ---- utilities ----
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        y = []
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            v = x[i]
            if v < lo: v = lo
            if v > hi: v = hi
            y.append(v)
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # LHS-ish sampling without external libs:
    # For each dimension, we permute bins and sample within each bin.
    def lhs_points(n):
        # create n points
        pts = [[0.0] * dim for _ in range(n)]
        for d in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            for i in range(n):
                a = perm[i]
                # sample within bin [a/n, (a+1)/n)
                u = (a + random.random()) / n
                pts[i][d] = lows[d] + u * spans[d]
        return pts

    # Safe evaluation: if func errors or returns non-finite, treat as very bad.
    def evaluate(x):
        try:
            val = func(x)
            if val is None or isinstance(val, bool):
                return float("inf")
            val = float(val)
            if math.isnan(val) or math.isinf(val):
                return float("inf")
            return val
        except Exception:
            return float("inf")

    best = float("inf")
    best_x = None

    # ---- initial design ----
    # choose modest budget tied to dimension; keep small to respect time
    n0 = max(8, min(40, 8 * dim))
    for x in lhs_points(n0):
        if time_left() <= 0:
            return best
        f = evaluate(x)
        if f < best:
            best = f
            best_x = x

    # If nothing valid, fall back to random attempts
    if best_x is None:
        best_x = rand_point()
        best = evaluate(best_x)

    # ---- local search (pattern search with adaptive step) ----
    # initial step sizes: fraction of span
    step = [0.2 * s if s > 0 else 1.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    no_improve_iters = 0
    restart_after = 25 + 5 * dim

    x = best_x[:]
    fx = best

    while time_left() > 0:
        improved = False

        # explore coordinate directions
        order = list(range(dim))
        random.shuffle(order)
        for i in order:
            if time_left() <= 0:
                return best

            if step[i] <= min_step[i]:
                continue

            # try +step and -step
            for sgn in (1.0, -1.0):
                cand = x[:]
                cand[i] = cand[i] + sgn * step[i]
                cand = clamp(cand)
                fc = evaluate(cand)
                if fc < fx:
                    x, fx = cand, fc
                    improved = True
                    break
            if improved:
                break

        if improved:
            if fx < best:
                best = fx
                best_x = x[:]
            # slightly increase step on success (encourage progress)
            for i in range(dim):
                step[i] *= 1.05
            no_improve_iters = 0
        else:
            # reduce step if no progress
            for i in range(dim):
                step[i] *= 0.5
            no_improve_iters += 1

        # ---- occasional restart / global exploration ----
        # If stuck, jump to a new region (random or near-best mutation)
        if no_improve_iters >= restart_after:
            no_improve_iters = 0

            # 50/50: random restart vs. mutate best
            if random.random() < 0.5:
                x = rand_point()
            else:
                # gaussian-like mutation via sum of uniforms (no numpy)
                x = best_x[:]
                for i in range(dim):
                    # approx N(0,1): sum of 12 uniforms - 6
                    z = sum(random.random() for _ in range(12)) - 6.0
                    scale = 0.1 * spans[i]
                    x[i] = x[i] + z * scale
                x = clamp(x)

            fx = evaluate(x)
            if fx < best:
                best = fx
                best_x = x[:]
            # reset step sizes after restart
            step = [0.2 * s if s > 0 else 1.0 for s in spans]

    return best
