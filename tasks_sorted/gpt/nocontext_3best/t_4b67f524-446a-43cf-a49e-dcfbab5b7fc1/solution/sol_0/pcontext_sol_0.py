import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: hybrid of
      - Latin-hypercube-like stratified initialization
      - adaptive local search with coordinate steps
      - occasional global restarts
      - lightweight annealing acceptance (to escape shallow local minima)

    Inputs:
      func(x): returns float fitness for list/tuple of length dim
      dim: int
      bounds: list of (low, high) for each dimension
      max_time: seconds (int/float)

    Returns:
      best: best (minimum) fitness found within time limit
    """
    t0 = time.time()

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

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an "array-like"; we pass list to avoid external deps.
        return float(func(x))

    # If any span is zero, treat as fixed coordinate
    fixed = [spans[i] == 0.0 for i in range(dim)]

    # --- initial stratified sampling (cheap exploration) ---
    # Choose a modest batch size that scales with dim but stays small.
    init_n = max(16, 8 * dim)
    # For each dimension, create shuffled strata indices
    strata = []
    for i in range(dim):
        order = list(range(init_n))
        random.shuffle(order)
        strata.append(order)

    best_x = None
    best = float("inf")

    # evaluate initial batch
    for k in range(init_n):
        if time.time() - t0 >= max_time:
            return best
        x = []
        for i in range(dim):
            if fixed[i]:
                x.append(lows[i])
            else:
                # sample within the k-th stratum for dim i
                a = strata[i][k]
                u = (a + random.random()) / init_n
                x.append(lows[i] + u * spans[i])
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # degenerate case (shouldn't happen), still safe
        best_x = rand_point()
        best = evaluate(best_x)

    # --- adaptive local search with restarts ---
    # Step sizes start as a fraction of bounds and shrink on stagnation.
    base_steps = [0.15 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
    steps = base_steps[:]

    # annealing temperature (relative); decreases over time
    T0 = 1.0

    # counters
    no_improve = 0
    iter_count = 0

    # restart policy
    restart_after = 40 + 10 * dim  # attempts without improvement before restart
    min_step_factor = 1e-6

    # current point
    x_cur = best_x[:]
    f_cur = best

    while True:
        if time.time() - t0 >= max_time:
            return best

        iter_count += 1

        # temperature schedule based on time fraction
        frac = (time.time() - t0) / max_time
        T = max(1e-9, T0 * (1.0 - frac) ** 2)

        # choose a move type: coordinate, multi-dim, or random jump
        r = random.random()
        x_new = x_cur[:]

        if r < 0.70:
            # coordinate step: pick one dim and move +/- step
            i = random.randrange(dim)
            if not fixed[i]:
                s = steps[i]
                if s > 0:
                    direction = -1.0 if random.random() < 0.5 else 1.0
                    x_new[i] = x_new[i] + direction * s
        elif r < 0.92:
            # multi-dim gaussian-like step scaled by steps
            for i in range(dim):
                if fixed[i]:
                    continue
                s = steps[i]
                if s > 0:
                    # approximate gaussian with sum of uniforms (Irwin–Hall)
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random() - 3.0) / 3.0
                    x_new[i] = x_new[i] + g * s
        else:
            # random jump (restart-ish)
            x_new = rand_point()

        x_new = clamp(x_new)
        f_new = evaluate(x_new)

        accept = False
        if f_new <= f_cur:
            accept = True
        else:
            # annealing acceptance
            # scale delta by (|f_cur|+1) to be somewhat objective-scale invariant
            denom = abs(f_cur) + 1.0
            delta = (f_new - f_cur) / denom
            # accept with probability exp(-delta/T)
            p = math.exp(-delta / max(T, 1e-12)) if delta > 0 else 1.0
            if random.random() < p:
                accept = True

        if accept:
            x_cur, f_cur = x_new, f_new

        # track global best
        if f_new < best:
            best = f_new
            best_x = x_new[:]
            no_improve = 0
        else:
            no_improve += 1

        # adapt steps: if stagnating, shrink; if improving, slightly expand
        if no_improve > 0 and no_improve % (10 + dim) == 0:
            # shrink steps
            for i in range(dim):
                steps[i] *= 0.5
            # if too small, trigger a restart around best
            if max(steps) < (max(spans) * min_step_factor if dim > 0 else 0.0):
                # reset steps and move current near best with jitter
                steps = base_steps[:]
                x_cur = best_x[:]
                for i in range(dim):
                    if fixed[i]:
                        continue
                    jitter = (random.random() * 2.0 - 1.0) * 0.05 * spans[i]
                    x_cur[i] += jitter
                x_cur = clamp(x_cur)
                f_cur = evaluate(x_cur)
                no_improve = 0

        # periodic restart to increase exploration
        if no_improve >= restart_after:
            steps = base_steps[:]
            # half the time jump to a fresh random point; half near current best
            if random.random() < 0.5:
                x_cur = rand_point()
            else:
                x_cur = best_x[:]
                for i in range(dim):
                    if fixed[i]:
                        continue
                    jitter = (random.random() * 2.0 - 1.0) * 0.10 * spans[i]
                    x_cur[i] += jitter
                x_cur = clamp(x_cur)
            f_cur = evaluate(x_cur)
            no_improve = 0
