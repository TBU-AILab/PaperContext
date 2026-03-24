import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: hybrid of
      - Latin-hypercube-like stratified sampling (cheap global coverage)
      - (1+1)-ES style local search with step-size adaptation
      - Occasional random restarts
    Returns: best (float) = minimum function value found within max_time seconds.
    """

    # -------- helpers --------
    eps = 1e-12

    def clip(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi: 
            return hi
        return x

    def clip_vec(v):
        return [clip(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        return bounds[i][1] - bounds[i][0]

    def safe_eval(x):
        # Evaluate objective; if it errors or returns non-finite, treat as very bad.
        try:
            y = func(x)
            if y is None:
                return float("inf")
            y = float(y)
            if math.isnan(y) or math.isinf(y):
                return float("inf")
            return y
        except Exception:
            return float("inf")

    # -------- time control --------
    start = time.time()
    deadline = start + max(0.0, float(max_time))

    # If no time, still try a single evaluation at center (if possible)
    def time_left():
        return time.time() < deadline

    # -------- initialization --------
    # Start from center of bounds
    x_best = [(bounds[i][0] + bounds[i][1]) * 0.5 for i in range(dim)]
    x_best = clip_vec(x_best)
    best = safe_eval(x_best)

    # Also try a few random points quickly
    for _ in range(min(5, 2 * dim + 1)):
        if not time_left():
            return best
        x = rand_uniform_vec()
        fx = safe_eval(x)
        if fx < best:
            best = fx
            x_best = x

    # Global sampling budget per "round"
    # (kept small so local search gets most time)
    global_batch = max(8, 4 * dim)

    # Step-size (relative to search range)
    # Using a single global sigma scale, applied per dimension with its span.
    avg_span = sum(max(span(i), eps) for i in range(dim)) / max(1, dim)
    sigma = 0.2 * avg_span  # absolute scale in "units of x"
    sigma_min = 1e-6 * avg_span
    sigma_max = 0.5 * avg_span

    # Track a "current" point for local improvement (can restart)
    x_cur = list(x_best)
    f_cur = best

    # Success-based adaptation parameters
    # If improvement: slightly increase sigma; else decrease.
    inc = 1.08
    dec = 0.92

    # For occasional restart
    no_improve = 0
    restart_after = max(50, 15 * dim)

    # -------- main loop --------
    while time_left():

        # --- (A) global stratified sampling (Latin-hypercube-ish) ---
        # Build stratified candidates along each dimension for this batch.
        # Each dimension uses a random permutation of bins.
        if dim > 0:
            perms = []
            for i in range(dim):
                p = list(range(global_batch))
                random.shuffle(p)
                perms.append(p)

            for k in range(global_batch):
                if not time_left():
                    return best
                x = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    # pick within bin [b/batch, (b+1)/batch)
                    b = perms[i][k]
                    u = (b + random.random()) / float(global_batch)
                    x[i] = lo + u * (hi - lo)
                fx = safe_eval(x)
                if fx < best:
                    best = fx
                    x_best = x
                    x_cur = list(x)
                    f_cur = fx
                    no_improve = 0

        # --- (B) local search around current best (1+1)-ES style ---
        # Number of local iterations per round depends on dimension.
        local_steps = max(30, 20 * dim)

        for _ in range(local_steps):
            if not time_left():
                return best

            # Propose: Gaussian perturbation with per-dimension scaling by span
            x_new = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                # scale sigma relative to average span but clamp by actual span
                s_i = sigma * (span(i) / (avg_span + eps))
                # If a dimension has near-zero span, keep it fixed
                if span(i) <= eps:
                    x_new[i] = lo
                else:
                    x_new[i] = x_cur[i] + random.gauss(0.0, s_i)
                    x_new[i] = clip(x_new[i], lo, hi)

            f_new = safe_eval(x_new)

            if f_new < f_cur:
                x_cur, f_cur = x_new, f_new
                no_improve = 0
                sigma = min(sigma * inc, sigma_max)

                if f_new < best:
                    best = f_new
                    x_best = list(x_new)
            else:
                no_improve += 1
                sigma = max(sigma * dec, sigma_min)

            # --- (C) occasional random restart to escape stagnation ---
            if no_improve >= restart_after:
                if not time_left():
                    return best
                # restart near best with moderate sigma, or fully random with some probability
                if random.random() < 0.35:
                    x_cur = rand_uniform_vec()
                else:
                    x_cur = list(x_best)
                    # slight kick
                    for i in range(dim):
                        if span(i) > eps:
                            x_cur[i] = clip(
                                x_cur[i] + random.gauss(0.0, 0.05 * span(i)),
                                bounds[i][0], bounds[i][1]
                            )
                f_cur = safe_eval(x_cur)
                if f_cur < best:
                    best = f_cur
                    x_best = list(x_cur)
                # reset counters and sigma
                no_improve = 0
                sigma = max(0.1 * avg_span, sigma_min)

    return best
