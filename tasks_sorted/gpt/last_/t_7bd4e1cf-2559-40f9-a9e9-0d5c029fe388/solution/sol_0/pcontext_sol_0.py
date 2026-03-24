import random
import math
import time

def run(func, dim, bounds, max_time):
    # Self-contained bounded optimizer: multi-start hill-climb + adaptive step + occasional restarts.
    # Uses only Python stdlib. Returns best (minimum) fitness found within max_time seconds.

    start = time.time()
    deadline = start + max_time

    # --- Helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def sample_uniform():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def reflect_into_bounds(vec):
        # Reflect to keep within [lo, hi] without sticking to edges too much.
        out = vec[:]
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            x = out[i]
            if lo == hi:
                out[i] = lo
                continue
            # Reflect repeatedly if needed (in case step is huge)
            while x < lo or x > hi:
                if x < lo:
                    x = lo + (lo - x)
                if x > hi:
                    x = hi - (x - hi)
            out[i] = clamp(x, lo, hi)
        return out

    def evaluate(vec):
        # func expects an array-like; we pass list (works for most implementations)
        return float(func(vec))

    # --- Initialization ---
    best = float("inf")
    best_x = None

    # Start with a few quick random probes to seed best
    # (kept small to avoid wasting time if max_time is tiny)
    init_tries = min(20, 2 * dim + 4)
    for _ in range(init_tries):
        if time.time() >= deadline:
            return best
        x = sample_uniform()
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # Should not happen, but keep safe
        best_x = sample_uniform()
        best = evaluate(best_x)

    # --- Main loop: multi-start local search with adaptive step sizes ---
    # Step sizes are relative to bounds; start moderate, shrink on stagnation, re-expand on restarts.
    base_sigma = [0.15 * s if s > 0 else 0.0 for s in spans]  # initial step scale per dimension
    min_sigma = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # Current state for local search
    x = best_x[:]
    fx = best
    sigma = base_sigma[:]

    # Budget/stagnation control
    no_improve = 0
    # Threshold before shrinking step
    shrink_after = 25 + 5 * dim
    # Threshold before restart
    restart_after = 80 + 10 * dim

    # Coordinate/gaussian move mixing
    prob_coord = 0.35  # coordinate-wise move probability

    while True:
        if time.time() >= deadline:
            return best

        # Propose a candidate
        cand = x[:]
        if random.random() < prob_coord:
            # Coordinate step (good for separable-ish problems)
            j = random.randrange(dim)
            step = random.gauss(0.0, sigma[j])
            cand[j] = cand[j] + step
        else:
            # Full gaussian step
            for i in range(dim):
                if sigma[i] > 0:
                    cand[i] = cand[i] + random.gauss(0.0, sigma[i])

        cand = reflect_into_bounds(cand)
        f_cand = evaluate(cand)

        if f_cand <= fx:
            # Accept improvement (or equal)
            x, fx = cand, f_cand
            no_improve = 0

            if f_cand < best:
                best, best_x = f_cand, cand
        else:
            no_improve += 1

        # Adaptation: shrink step sizes if stuck
        if no_improve == shrink_after:
            for i in range(dim):
                sigma[i] = max(min_sigma[i], sigma[i] * 0.5)

        # Restart if very stuck or step sizes too small
        if no_improve >= restart_after or all(sigma[i] <= 5.0 * min_sigma[i] for i in range(dim)):
            # With some probability, restart near global best; otherwise random restart
            if best_x is not None and random.random() < 0.65:
                x = best_x[:]
                # small perturbation around best
                for i in range(dim):
                    if spans[i] > 0:
                        x[i] = x[i] + random.gauss(0.0, 0.05 * spans[i])
                x = reflect_into_bounds(x)
                fx = evaluate(x)
            else:
                x = sample_uniform()
                fx = evaluate(x)

            # Reset step sizes
            sigma = base_sigma[:]
            no_improve = 0

            if fx < best:
                best, best_x = fx, x
