import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid strategy:
      1) Random exploration
      2) Local coordinate perturbation with adaptive step sizes
      3) Occasional "restart" (re-sample) to escape local minima

    Arguments:
      func: callable(params_list)-> float
      dim: int, number of parameters
      bounds: list of (low, high) pairs, length == dim
      max_time: seconds (int/float)

    Returns:
      best: best (minimum) fitness found (float)
    """
    t0 = time.time()

    # --- Helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        lo, hi = lows[i], highs[i]
        if x < lo: return lo
        if x > hi: return hi
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; keep it list-only (no external libs)
        v = func(x)
        # Be robust to weird returns
        if v is None:
            return float("inf")
        try:
            v = float(v)
        except Exception:
            return float("inf")
        if math.isnan(v) or math.isinf(v):
            return float("inf")
        return v

    # --- Initialization ---
    x_best = rand_point()
    f_best = evaluate(x_best)

    # Per-dimension step sizes (start moderately large)
    step = [0.25 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Control parameters
    shrink = 0.7        # step *= shrink on failure
    grow = 1.05         # step *= grow on success (small)
    restart_prob = 0.05 # occasional restart even if improving
    no_improve = 0
    max_no_improve = 1000

    # --- Main loop ---
    while True:
        if time.time() - t0 >= max_time:
            return f_best

        # Restart logic: if stuck, or by small probability
        if no_improve >= max_no_improve or random.random() < restart_prob:
            x = rand_point()
            f = evaluate(x)
            if f < f_best:
                x_best, f_best = x, f
                # re-expand steps a bit after finding new best by restart
                step = [max(s, 0.25 * spans[i]) for i, s in enumerate(step)]
                no_improve = 0
            else:
                no_improve += 1
            continue

        # Local search around current best: coordinate-wise random perturbations
        x = x_best[:]  # start from incumbent
        improved = False

        # Randomize coordinate order
        idxs = list(range(dim))
        random.shuffle(idxs)

        for i in idxs:
            if time.time() - t0 >= max_time:
                return f_best

            if spans[i] <= 0:
                continue

            # Try a few candidates along this coordinate
            # symmetric move + a smaller move
            candidates = []
            delta = step[i]
            if delta <= 0:
                continue

            # Direction choice
            sign = -1.0 if random.random() < 0.5 else 1.0
            candidates.append(sign * delta)
            candidates.append(-sign * delta)
            candidates.append(sign * 0.5 * delta)
            candidates.append(-sign * 0.5 * delta)

            best_local = f_best
            best_x_local = None

            for d in candidates:
                xx = x_best[:]  # always perturb around best
                xx[i] = clamp(xx[i] + d, i)
                ff = evaluate(xx)
                if ff < best_local:
                    best_local = ff
                    best_x_local = xx

            if best_x_local is not None:
                # Accept improvement
                x_best = best_x_local
                f_best = best_local
                improved = True
                # Slightly grow step on success, but cap to span
                step[i] = min(step[i] * grow, spans[i])
            else:
                # Shrink step if no improvement on this coordinate
                step[i] = max(step[i] * shrink, min_step[i])

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        # If all steps became tiny, trigger a restart soon
        if all(step[i] <= 10.0 * min_step[i] for i in range(dim)):
            no_improve = max_no_improve

