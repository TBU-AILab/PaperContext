import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      1) Low-discrepancy-ish initialization (stratified per-dimension)
      2) Adaptive local search with decreasing step sizes
      3) Occasional heavy-tailed (Cauchy-like) jumps to escape local minima

    Returns:
      best (float): best (minimum) fitness value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- Helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        # func expects an "array-like"; list is fine per prompt example
        return float(func(x))

    # Safe initial point: random
    x_best = [lows[i] + random.random() * spans[i] for i in range(dim)]
    f_best = eval_f(x_best)

    # Track a "current" point for local search (start at best)
    x_cur = x_best[:]
    f_cur = f_best

    # Initial step sizes: fraction of range
    step = [0.2 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Budgeted initialization: stratified samples count based on dim and time
    # (kept small for high-dim)
    init_samples = max(10, min(200, 20 * dim))

    # --- Phase 1: stratified-ish sampling around space ---
    # For each sample, for each dimension pick a random stratum index.
    # This is a cheap alternative to true LHS without external libs.
    for s in range(init_samples):
        if time.time() >= deadline:
            return f_best

        x = [0.0] * dim
        # number of strata per dimension
        m = init_samples
        for i in range(dim):
            j = random.randrange(m)
            u = (j + random.random()) / m
            x[i] = lows[i] + u * spans[i]
        f = eval_f(x)
        if f < f_best:
            f_best = f
            x_best = x
            x_cur = x[:]
            f_cur = f

    # --- Phase 2: adaptive local search with occasional global jumps ---
    no_improve = 0
    iter_count = 0

    while time.time() < deadline:
        iter_count += 1

        # Occasionally restart from best with a random jump
        # Heavy-tailed step: tan(pi*(u-0.5)) ~ Cauchy-like
        do_jump = (no_improve > 50 and (iter_count % 5 == 0)) or (iter_count % 97 == 0)
        if do_jump:
            x_cur = x_best[:]
            for i in range(dim):
                u = random.random()
                # cauchy-ish scale relative to range
                jump = math.tan(math.pi * (u - 0.5)) * 0.05 * spans[i]
                # if spans is 0, keep as 0
                if spans[i] == 0:
                    jump = 0.0
                x_cur[i] += jump
            clip_vec(x_cur)
            f_cur = eval_f(x_cur)
            if f_cur < f_best:
                f_best = f_cur
                x_best = x_cur[:]
                no_improve = 0
            else:
                no_improve += 1
            continue

        # Propose a local move: coordinate-wise gaussian perturbation
        x_new = x_cur[:]
        for i in range(dim):
            if spans[i] == 0:
                continue
            # gaussian step with current step size, plus tiny uniform noise
            x_new[i] += random.gauss(0.0, step[i]) + (random.random() - 0.5) * 1e-6 * spans[i]
        clip_vec(x_new)

        f_new = eval_f(x_new)

        # Acceptance: greedy with mild simulated annealing early
        # temperature decays with time
        t = time.time()
        frac = (t - t0) / max(1e-9, max_time)
        temp = max(1e-12, (1.0 - frac))  # in [~0,1]
        accept = False
        if f_new <= f_cur:
            accept = True
        else:
            # probabilistic acceptance to escape shallow traps
            # scaled by magnitude to be robust
            scale = max(1.0, abs(f_cur))
            p = math.exp(-(f_new - f_cur) / (scale * 0.1 * temp + 1e-12))
            if random.random() < p:
                accept = True

        if accept:
            x_cur, f_cur = x_new, f_new
            # If improved global best, reduce no_improve and slightly tighten steps
            if f_new < f_best:
                f_best = f_new
                x_best = x_new[:]
                no_improve = 0
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.95)
            else:
                no_improve += 1
        else:
            no_improve += 1

        # If stuck, shrink steps; if very stuck, restart near best with larger steps
        if no_improve % 40 == 0:
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.7)

        if no_improve > 200:
            # restart around best with broadened steps
            x_cur = x_best[:]
            for i in range(dim):
                step[i] = min(0.2 * spans[i] if spans[i] > 0 else step[i], step[i] * 1.5)
                if spans[i] > 0:
                    x_cur[i] += random.gauss(0.0, step[i])
            clip_vec(x_cur)
            f_cur = eval_f(x_cur)
            no_improve = 0
            if f_cur < f_best:
                f_best = f_cur
                x_best = x_cur[:]

    # return fitness of the best found solution
    return f_best
