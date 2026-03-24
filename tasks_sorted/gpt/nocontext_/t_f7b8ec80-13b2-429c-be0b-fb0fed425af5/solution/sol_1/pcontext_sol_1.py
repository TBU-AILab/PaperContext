import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      1) Latin-hypercube-like random initialization
      2) (1+lambda) Evolution Strategy with self-adaptive step-size
      3) Occasional coordinate/local refinement
    Returns: best (float) = lowest fitness found.
    """

    # --------- helpers ----------
    def clamp(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def clip_vec(vec):
        return [clamp(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects "array-like"; keep it plain list to avoid external deps
        return float(func(x))

    def span(i):
        return bounds[i][1] - bounds[i][0]

    # --------- start ----------
    t0 = time.time()
    deadline = t0 + max_time
    if dim <= 0:
        return float("inf")

    # initial global scale (relative to bounds)
    base_sigma = [0.2 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]

    # (Quasi) Latin hypercube initialization: per-dimension stratified sampling
    # then shuffled to make initial points spread out.
    init_n = max(8, 4 * dim)
    strata = []
    for i in range(dim):
        lo, hi = bounds[i]
        w = (hi - lo) / init_n if init_n > 0 else 0.0
        vals = [lo + (k + random.random()) * w for k in range(init_n)]
        random.shuffle(vals)
        strata.append(vals)

    best_x = None
    best = float("inf")

    # Evaluate initialization set
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = [strata[i][k] for i in range(dim)]
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    if best_x is None:
        best_x = rand_in_bounds()
        best = eval_f(best_x)

    # Evolution Strategy parameters
    lam = max(8, 4 * dim)   # offspring count
    success_window = 20
    successes = 0
    trials = 0

    # self-adaptive sigma multiplier
    sigma_mult = 1.0
    min_sigma_mult = 1e-6
    max_sigma_mult = 1e2

    # local refinement settings
    local_every = max(25, 5 * dim)  # do local coordinate search occasionally
    it = 0

    # Main loop
    while time.time() < deadline:
        it += 1

        # Generate offspring around current best
        parent = best_x
        parent_f = best
        best_off_x = None
        best_off_f = float("inf")

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            child = []
            for i in range(dim):
                s = base_sigma[i] * sigma_mult
                # gaussian step; fallback if span is degenerate
                step = random.gauss(0.0, s if s > 0 else 1.0)
                child.append(parent[i] + step)

            child = clip_vec(child)
            f = eval_f(child)

            if f < best_off_f:
                best_off_f = f
                best_off_x = child

        # (1+lambda): accept best offspring if it improves
        trials += 1
        if best_off_f < parent_f:
            best_x, best = best_off_x, best_off_f
            successes += 1

        # 1/5 success rule adaptation every window
        if trials >= success_window:
            rate = successes / float(trials)
            # If success rate > 1/5 increase step size, else decrease
            if rate > 0.2:
                sigma_mult *= 1.5
            else:
                sigma_mult /= 1.5
            sigma_mult = max(min_sigma_mult, min(max_sigma_mult, sigma_mult))
            successes = 0
            trials = 0

        # Occasional coordinate/local refinement around best
        if it % local_every == 0 and time.time() < deadline:
            x = best_x[:]
            f0 = best
            improved = True
            # A short greedy coordinate descent with shrinking steps
            # Uses per-dimension step sizes derived from current sigma
            steps = [max(1e-12, base_sigma[i] * sigma_mult) for i in range(dim)]
            # limit iterations to keep time-bounded
            for _round in range(3):
                if time.time() >= deadline:
                    return best
                improved = False
                for i in range(dim):
                    if time.time() >= deadline:
                        return best
                    step = steps[i]
                    if step <= 0:
                        continue

                    # try +step and -step
                    for direction in (+1.0, -1.0):
                        cand = x[:]
                        cand[i] = clamp(cand[i] + direction * step, bounds[i][0], bounds[i][1])
                        fc = eval_f(cand)
                        if fc < f0:
                            x, f0 = cand, fc
                            improved = True

                    # if no improvement, reduce step in this coordinate
                    if not improved:
                        steps[i] *= 0.5

            if f0 < best:
                best, best_x = f0, x

        # Mild random restart if sigma becomes too small or stagnation likely
        # (helps escape local minima without heavy bookkeeping)
        if sigma_mult < 1e-5 and time.time() < deadline:
            # try a few random points quickly
            for _ in range(3):
                if time.time() >= deadline:
                    return best
                x = rand_in_bounds()
                f = eval_f(x)
                if f < best:
                    best, best_x = f, x
                    sigma_mult = 1.0
                    break
            # reset sigma to continue exploring
            sigma_mult = max(sigma_mult, 0.5)

    return best
