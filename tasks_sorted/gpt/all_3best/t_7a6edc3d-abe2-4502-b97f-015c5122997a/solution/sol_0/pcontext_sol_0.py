import random
import math
import time

def run(func, dim, bounds, max_time):
    # Seed for reproducibility across runs can be set here if desired:
    # random.seed(0)

    start = time.time()
    deadline = start + float(max_time)

    # ----- Helpers -----
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_point(x):
        # func expects an "array-like"; list is acceptable for most call sites.
        return float(func(x))

    # ----- Initialization: multi-start random sampling -----
    best_x = rand_point()
    best = eval_point(best_x)

    # How many random initial probes (kept small; time-limited anyway)
    init_trials = max(10, 20 * dim)
    for _ in range(init_trials - 1):
        if time.time() >= deadline:
            return best
        x = rand_point()
        fx = eval_point(x)
        if fx < best:
            best, best_x = fx, x

    # ----- Main algorithm: adaptive local search with occasional restarts -----
    # Strategy:
    # - Coordinate-wise random step around current point
    # - If improved: accept and slightly increase step (exploit)
    # - If not: shrink step (refine)
    # - Occasionally restart to escape local minima
    step = [0.25 * s if s > 0 else 1.0 for s in spans]  # initial relative step
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    current_x = list(best_x)
    current_f = best

    # Restart control
    no_improve = 0
    restart_after = 50 + 20 * dim

    while True:
        if time.time() >= deadline:
            return best

        # Restart if stuck
        if no_improve >= restart_after:
            # Random restart, but keep best overall
            current_x = rand_point()
            current_f = eval_point(current_x)
            # Reset step to encourage exploration after restart
            step = [0.25 * s if s > 0 else 1.0 for s in spans]
            no_improve = 0
            if current_f < best:
                best, best_x = current_f, list(current_x)
            continue

        # Propose a neighbor by perturbing a subset of coordinates
        candidate = list(current_x)

        # Choose how many coords to perturb (1..dim), biased toward small changes
        # so we do local refinements frequently.
        k = 1
        if dim > 1 and random.random() < 0.3:
            k = 1 + int(random.random() * dim)  # sometimes larger moves

        idxs = set()
        while len(idxs) < k:
            idxs.add(random.randrange(dim))

        for i in idxs:
            # symmetric perturbation; scale by step
            candidate[i] = clamp(candidate[i] + (2.0 * random.random() - 1.0) * step[i], i)

        cand_f = eval_point(candidate)

        if cand_f < current_f:
            # Accept improvement
            current_x, current_f = candidate, cand_f
            no_improve = 0

            # Slightly increase step on changed dimensions (momentum)
            for i in idxs:
                step[i] *= 1.15

            # Update global best
            if cand_f < best:
                best, best_x = cand_f, list(candidate)
        else:
            no_improve += 1
            # Shrink steps on changed dimensions to refine
            for i in idxs:
                step[i] *= 0.85
                if step[i] < min_step[i]:
                    step[i] = min_step[i]

            # Occasionally accept a worse move early on (tiny SA-like kick),
            # helps avoid shallow local traps without heavy tuning.
            if random.random() < 0.02:
                current_x, current_f = candidate, cand_f

    # Unreachable, but kept for clarity
    # return best
