import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - Latin-hypercube-like randomized initialization
      - Coordinate/local random search around best
      - Simulated annealing style occasional uphill acceptance
      - Step-size adaptation + periodic restarts

    Returns:
      best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # --- helpers (no external libs) ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    widths = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        y = list(x)
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_uniform_point():
        return [lows[i] + random.random() * widths[i] for i in range(dim)]

    def eval_f(x):
        # func is specified as accepting an array-like; pass a list
        return float(func(x))

    # --- initial sampling (stratified per-dimension) ---
    best = float("inf")
    best_x = None

    # number of initial points: scale with dimension but keep small
    init_n = max(8, 4 * dim)
    strata = []
    for i in range(dim):
        # create strata intervals and shuffle them
        idx = list(range(init_n))
        random.shuffle(idx)
        strata.append(idx)

    # Evaluate initial points
    for k in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # sample within a stratum
            a = strata[i][k] / init_n
            b = (strata[i][k] + 1) / init_n
            u = a + (b - a) * random.random()
            x.append(lows[i] + u * widths[i])
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    if best_x is None:
        # fallback
        best_x = rand_uniform_point()
        best = eval_f(best_x)

    # --- main loop: adaptive local/global search with annealing + restarts ---
    # step sizes start as a fraction of range, then adapt
    step = [0.2 * widths[i] if widths[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (abs(widths[i]) + 1.0) for i in range(dim)]

    # annealing temperature based on magnitude of best; keep > 0
    def initial_temp(f):
        return max(1e-9, 0.1 * (abs(f) + 1.0))

    T0 = initial_temp(best)
    T = T0

    # bookkeeping for adaptation
    iter_count = 0
    improve_streak = 0
    no_improve = 0

    # restart parameters
    restart_every = max(50, 20 * dim)  # iterations without improvement before restart
    shrink_every = max(30, 10 * dim)

    while True:
        if time.time() >= deadline:
            return best

        iter_count += 1

        # Cooling schedule (slow)
        # keep temperature from vanishing too quickly (time-bounded scenario)
        T = T0 / (1.0 + 0.01 * iter_count)

        # Decide whether to do local perturbation or a global jump (restart-like)
        do_global = (no_improve >= restart_every)

        if do_global:
            # global restart around best + random point mix
            alpha = 0.5 * random.random()
            xr = rand_uniform_point()
            x = [alpha * best_x[i] + (1.0 - alpha) * xr[i] for i in range(dim)]
            # also reset step sizes a bit
            step = [max(0.1 * widths[i], min_step[i]) for i in range(dim)]
            no_improve = 0
            improve_streak = 0
        else:
            # local move: perturb a subset of coordinates
            x = list(best_x)
            # number of perturbed dims
            k = 1 if dim == 1 else random.randint(1, min(dim, 1 + dim // 2))
            idxs = random.sample(range(dim), k)
            for i in idxs:
                if widths[i] <= 0:
                    continue
                # gaussian-like via sum of uniforms (CLT) => no external libs
                g = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random()) - 3.0
                x[i] += g * step[i]
            x = clip(x)

        fx = eval_f(x)

        # Accept if better, or sometimes accept worse (annealing)
        accept = False
        if fx <= best:
            accept = True
        else:
            # SA acceptance probability
            # guard against overflow/underflow
            delta = fx - best
            if T > 0:
                p = math.exp(-delta / (T + 1e-12))
                if random.random() < p:
                    accept = True

        if accept:
            if fx < best:
                best = fx
                best_x = x
                improve_streak += 1
                no_improve = 0
                # if we improve often, slightly expand steps to explore
                if improve_streak >= 5:
                    for i in range(dim):
                        step[i] = min(step[i] * 1.2, widths[i])
                    improve_streak = 0
            else:
                # accepted but not better: treat as exploration
                no_improve += 1
                improve_streak = 0
        else:
            no_improve += 1
            improve_streak = 0

        # Periodically shrink step sizes if not improving
        if no_improve > 0 and (no_improve % shrink_every == 0):
            for i in range(dim):
                step[i] = max(step[i] * 0.5, min_step[i])

        # If steps are tiny across the board, do a mild restart
        if all(step[i] <= 10.0 * min_step[i] for i in range(dim)):
            xr = rand_uniform_point()
            # blend with best to keep exploitation
            beta = 0.7 + 0.3 * random.random()
            best_x = clip([beta * best_x[i] + (1.0 - beta) * xr[i] for i in range(dim)])
            step = [max(0.15 * widths[i], min_step[i]) for i in range(dim)]
            # re-evaluate in case best_x drifted (optional)
            ftmp = eval_f(best_x)
            if ftmp < best:
                best = ftmp
            no_improve = 0
            improve_streak = 0
