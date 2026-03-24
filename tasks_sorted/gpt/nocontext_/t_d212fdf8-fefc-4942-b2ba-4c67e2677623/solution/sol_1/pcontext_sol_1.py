import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer (self-contained, no external libs).
    Strategy: adaptive (1+lambda) Evolution Strategy with:
      - log-normal step-size adaptation (success-based)
      - bound handling by reflection
      - occasional random restarts
    Returns: best (float) = minimum fitness found within max_time seconds.
    """

    # ---------- helpers ----------
    def clip_reflect(x, lo, hi):
        # Reflect at boundaries to keep values within [lo, hi]
        # Works even if step jumps far outside.
        if lo > hi:
            lo, hi = hi, lo
        width = hi - lo
        if width <= 0.0:
            return lo
        # Reflect using a "triangle wave" mapping
        y = (x - lo) % (2.0 * width)
        if y > width:
            y = 2.0 * width - y
        return lo + y

    def make_within(vec):
        return [clip_reflect(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def diag_scales():
        # characteristic scale per dimension
        s = []
        for lo, hi in bounds:
            w = float(hi - lo)
            if w <= 0.0:
                w = 1.0
            s.append(w)
        return s

    def approx_init_sigma(scales):
        # Start with ~10% of range (capped to sensible minimum)
        return 0.1 * sum(scales) / max(1, dim)

    # ---------- initialization ----------
    t0 = time.time()
    deadline = t0 + float(max_time)

    scales = diag_scales()
    sigma = approx_init_sigma(scales)

    # (1+lambda)-ES parameters
    lam = max(4, 4 * dim)          # offspring per iteration
    elite_keep = 1                  # only keep best parent
    p_target = 0.2                  # target success rate
    c_sigma = 0.25                  # adaptation speed
    min_sigma = 1e-12               # prevents collapse
    max_sigma = 1e6                 # prevents explosion

    # Random initial parent
    parent = rand_uniform_vec()
    best = func(parent)
    best_x = parent[:]
    parent_fit = best

    # Restart control
    no_improve_iters = 0
    restart_patience = 60 + 10 * dim  # iterations without improvement before restart
    hard_restart_fraction = 0.15      # restart chance when patience exceeded

    # ---------- main loop ----------
    it = 0
    while True:
        if time.time() >= deadline:
            return float(best)

        it += 1
        improved_this_iter = False
        successes = 0

        # Generate offspring and select best
        best_child = None
        best_child_fit = float("inf")

        for _ in range(lam):
            # Stop early if time is up
            if time.time() >= deadline:
                return float(best)

            # Mutate parent with Gaussian noise scaled per-dim
            child = []
            for i in range(dim):
                step = random.gauss(0.0, 1.0) * sigma * (scales[i] if scales[i] > 0 else 1.0)
                child.append(parent[i] + step)

            child = make_within(child)
            f = func(child)

            if f < best_child_fit:
                best_child_fit = f
                best_child = child

        # Selection: (1+lambda)
        if best_child_fit <= parent_fit:
            parent = best_child
            parent_fit = best_child_fit
            successes = 1
        else:
            successes = 0

        # Track global best
        if parent_fit < best:
            best = parent_fit
            best_x = parent[:]
            improved_this_iter = True

        # Step-size adaptation (success rule)
        # Increase sigma if success rate > target, decrease otherwise
        # For (1+lambda) where exactly 0/1 success per iter:
        p_succ = float(successes)
        sigma *= math.exp(c_sigma * (p_succ - p_target))
        if sigma < min_sigma:
            sigma = min_sigma
        elif sigma > max_sigma:
            sigma = max_sigma

        # Restart logic: if stuck, sometimes restart to explore new basins
        if improved_this_iter:
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        if no_improve_iters >= restart_patience:
            # Probabilistic hard restart
            if random.random() < hard_restart_fraction:
                parent = rand_uniform_vec()
                parent_fit = func(parent)
                if parent_fit < best:
                    best = parent_fit
                    best_x = parent[:]
                # reset sigma to initial scale
                sigma = approx_init_sigma(scales)
                no_improve_iters = 0
            else:
                # Soft restart: increase exploration
                sigma *= 2.0
                if sigma > max_sigma:
                    sigma = max_sigma
                no_improve_iters = int(restart_patience * 0.5)

    # return fitness of the best found solution
    return float(best)
