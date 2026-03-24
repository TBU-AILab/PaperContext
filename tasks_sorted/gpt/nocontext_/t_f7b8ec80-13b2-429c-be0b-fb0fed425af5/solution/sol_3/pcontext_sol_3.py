import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using a compact (1+1)-ES style search:
    - Keep best-so-far point x_best
    - Propose x_candidate = x_best + N(0, sigma) per dimension
    - If improves => accept and slightly increase sigma
      else        => reject and slightly decrease sigma
    - Occasional global restarts to escape local minima
    Returns: best (float) fitness found within max_time seconds.
    """
    t0 = time.time()

    # --- helpers ---
    def clip(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; we pass a Python list
        return float(func(x))

    # --- initialize ---
    x_best = rand_in_bounds()
    f_best = eval_f(x_best)

    # step sizes relative to range
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    base_sigma = [0.2 * r if r > 0 else 1.0 for r in ranges]  # initial scale
    sigma = base_sigma[:]

    # schedule / controls
    no_improve = 0
    iter_count = 0

    # caps to avoid degeneracy
    sigma_min = [1e-12 * (r if r > 0 else 1.0) for r in ranges]
    sigma_max = [0.5 * (r if r > 0 else 1.0) for r in ranges]

    # acceptance adaptation factors
    grow = 1.05
    shrink = 0.97

    # restart controls
    # If stuck, restart from a random point and reset sigmas.
    def do_restart():
        nonlocal x_best, f_best, sigma, no_improve
        x = rand_in_bounds()
        f = eval_f(x)
        if f < f_best:
            x_best, f_best = x, f
        sigma = base_sigma[:]
        no_improve = 0

    # ensure at least one evaluation already done; loop until time runs out
    while True:
        if time.time() - t0 >= max_time:
            return f_best

        iter_count += 1

        # occasional forced random samples (global exploration)
        # more frequent early, less later
        if iter_count % 50 == 0:
            x = rand_in_bounds()
            f = eval_f(x)
            if f < f_best:
                x_best, f_best = x, f
                no_improve = 0
            else:
                no_improve += 1
            continue

        # propose local mutation around current best
        x_cand = []
        for i in range(dim):
            lo, hi = bounds[i]
            step = random.gauss(0.0, sigma[i])
            xi = clip(x_best[i] + step, lo, hi)
            x_cand.append(xi)

        f_cand = eval_f(x_cand)

        if f_cand <= f_best:
            # accept improvement
            x_best, f_best = x_cand, f_cand
            no_improve = 0
            # slightly increase step size (encourage progress)
            for i in range(dim):
                sigma[i] = min(sigma_max[i], sigma[i] * grow)
        else:
            no_improve += 1
            # shrink step size (fine-tune locally)
            for i in range(dim):
                sigma[i] = max(sigma_min[i], sigma[i] * shrink)

        # restart if no improvement for a while (scaled by dimension)
        if no_improve >= max(200, 50 * dim):
            do_restart()
