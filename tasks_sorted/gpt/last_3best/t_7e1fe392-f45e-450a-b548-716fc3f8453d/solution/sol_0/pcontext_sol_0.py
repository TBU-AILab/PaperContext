import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - global exploration (uniform sampling)
      - local refinement (adaptive coordinate search / hill-climb)
      - occasional random restarts
    Returns: best (float) = best function value found within max_time.
    """
    t0 = time.time()

    # --------- helpers ----------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        return bounds[i][1] - bounds[i][0]

    # Robust evaluation wrapper: if func errors or returns non-finite -> treat as very bad
    def evaluate(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if not math.isfinite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # --------- initialization ----------
    best_x = rand_in_bounds()
    best = evaluate(best_x)

    # start with step sizes as a fraction of each dimension range
    base_steps = [0.2 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]
    steps = base_steps[:]  # adaptive

    # Simple schedule knobs (kept small & robust)
    shrink = 0.7          # shrink factor when stuck
    grow = 1.05           # mild growth on progress
    min_step_factor = 1e-12
    restart_prob = 0.03   # random restart probability per outer loop
    explore_prob = 0.25   # do pure random sample sometimes

    # --------- main loop ----------
    while True:
        if time.time() - t0 >= max_time:
            return best

        # occasional exploration
        if random.random() < explore_prob:
            x = rand_in_bounds()
            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x
                # when we find a new best globally, slightly increase steps
                for i in range(dim):
                    steps[i] = min(span(i), max(steps[i] * grow, base_steps[i] * 0.05))
            continue

        # occasional restart around best (or fully random)
        if random.random() < restart_prob:
            if random.random() < 0.5:
                best_x = rand_in_bounds()
            else:
                # jitter around current best
                x = []
                for i in range(dim):
                    s = max(steps[i], base_steps[i] * 0.01)
                    lo, hi = bounds[i]
                    x.append(clamp(best_x[i] + random.gauss(0.0, s), lo, hi))
                best_x = x
            best = min(best, evaluate(best_x))
            # reset steps moderately
            for i in range(dim):
                steps[i] = max(base_steps[i] * 0.1, steps[i])
            continue

        # local coordinate search around current best_x
        improved = False
        x0 = best_x
        f0 = best

        # randomize coordinate order
        order = list(range(dim))
        random.shuffle(order)

        for i in order:
            if time.time() - t0 >= max_time:
                return best

            lo, hi = bounds[i]
            si = steps[i]

            # try plus and minus moves; also a smaller move if needed
            candidates = []
            xi_plus = x0[i] + si
            xi_minus = x0[i] - si
            if xi_plus <= hi:
                candidates.append(xi_plus)
            if xi_minus >= lo:
                candidates.append(xi_minus)

            # if both clipped, try a smaller move
            if not candidates:
                si2 = si * 0.5
                if x0[i] + si2 <= hi:
                    candidates.append(x0[i] + si2)
                if x0[i] - si2 >= lo:
                    candidates.append(x0[i] - si2)

            best_local_f = f0
            best_local_xi = x0[i]

            for xi in candidates:
                x = list(x0)
                x[i] = clamp(xi, lo, hi)
                fx = evaluate(x)
                if fx < best_local_f:
                    best_local_f = fx
                    best_local_xi = x[i]

            if best_local_f < f0:
                # accept improvement immediately (first-improvement strategy)
                x0 = list(x0)
                x0[i] = best_local_xi
                f0 = best_local_f
                improved = True

        if improved:
            best_x, best = x0, f0
            # mild step growth on success (bounded by search space)
            for i in range(dim):
                steps[i] = min(span(i), steps[i] * grow)
        else:
            # shrink steps when no coordinate move helps
            for i in range(dim):
                steps[i] *= shrink
                # prevent underflow to zero
                min_step = max(span(i) * min_step_factor, 1e-15)
                if steps[i] < min_step:
                    steps[i] = min_step
