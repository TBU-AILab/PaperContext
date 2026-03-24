import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - Latin-hypercube-like random initialization
      - Adaptive step-size local search (coordinate-wise + random directions)
      - Occasional restarts (best-biased) to escape local minima

    Returns:
      best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()

    # ---- helpers ----
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        return [min(highs[i], max(lows[i], x[i])) for i in range(dim)]

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func is assumed to accept an array-like of length dim
        return float(func(x))

    # ---- initialization: stratified sampling per dimension ----
    # Create a small set of diverse points quickly (similar to LHS).
    init_n = max(8, 4 * dim)
    # For each dimension, make init_n bins and permute
    bins = []
    for i in range(dim):
        perm = list(range(init_n))
        random.shuffle(perm)
        bins.append(perm)

    best_x = None
    best = float("inf")

    for k in range(init_n):
        if time.time() - t0 >= max_time:
            return best
        x = []
        for i in range(dim):
            # sample within bin
            a = (bins[i][k] + random.random()) / init_n
            x.append(lows[i] + a * spans[i])
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # fallback (should not happen)
        best_x = rand_point()
        best = eval_f(best_x)

    # ---- main loop: adaptive local search with restarts ----
    # initial step sizes: fraction of span
    step = [0.25 * s if s > 0 else 1.0 for s in spans]
    min_step = [1e-12 * (s if s > 0 else 1.0) for s in spans]

    # restart controls
    no_improve = 0
    restart_after = 30 + 10 * dim  # iterations without improvement before restart

    # internal iteration counter
    it = 0

    while time.time() - t0 < max_time:
        it += 1

        x = best_x[:]  # start from current best
        fx = best

        improved = False

        # --- coordinate-wise exploratory moves (pattern search style) ---
        order = list(range(dim))
        random.shuffle(order)

        for i in order:
            if time.time() - t0 >= max_time:
                return best

            si = step[i]
            if si <= min_step[i]:
                continue

            # try +step
            xp = x[:]
            xp[i] += si
            xp = clip(xp)
            fp = eval_f(xp)

            if fp < fx:
                x, fx = xp, fp
                improved = True
                continue

            # try -step
            xm = x[:]
            xm[i] -= si
            xm = clip(xm)
            fm = eval_f(xm)

            if fm < fx:
                x, fx = xm, fm
                improved = True
                continue

        # --- random direction refinement (helps when axes are not aligned) ---
        if time.time() - t0 >= max_time:
            return best

        # build random unit direction
        direction = [random.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(d * d for d in direction)) or 1.0
        direction = [d / norm for d in direction]

        # choose a scalar step based on typical step magnitude
        typical = sum(step) / float(dim)
        alpha = typical * (0.5 + random.random())  # in [0.5, 1.5] * typical

        xr = [x[i] + alpha * direction[i] for i in range(dim)]
        xr = clip(xr)
        fr = eval_f(xr)
        if fr < fx:
            x, fx = xr, fr
            improved = True

        # --- accept and adapt steps ---
        if fx < best:
            best, best_x = fx, x
            improved = True

        if improved:
            no_improve = 0
            # gently increase steps (but cap at span)
            for i in range(dim):
                step[i] = min(spans[i] if spans[i] > 0 else step[i], step[i] * 1.15)
        else:
            no_improve += 1
            # shrink steps
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.7)

        # --- restart logic ---
        if no_improve >= restart_after:
            no_improve = 0

            # best-biased restart: sample around best with decreasing radius
            # radius based on current step sizes; if too small, do a full random restart
            radius = [max(step[i], 0.05 * spans[i]) for i in range(dim)]
            if sum(step) <= sum(min_step) * 10:
                cand = rand_point()
            else:
                cand = [best_x[i] + random.uniform(-radius[i], radius[i]) for i in range(dim)]
                cand = clip(cand)

            fc = eval_f(cand)
            if fc < best:
                best, best_x = fc, cand

    return best
