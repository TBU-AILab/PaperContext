import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization (per-dimension shuffles)
    - Local refinement with adaptive step-size hill-climbing (1/5 success rule)
    - Occasional random restarts to escape local minima

    Returns:
        best (float): best (minimum) fitness value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; list is acceptable per prompt (np.array mentioned but not required)
        return float(func(x))

    # --- initial sampling count (kept modest; time governs anyway) ---
    # scale with dimension, but cap so it doesn't waste time on large dims
    init_n = max(8, min(60, 10 * dim))

    best = float("inf")
    best_x = None

    # --- stratified (LHS-like) initial candidates ---
    # For each dimension, create a random permutation of strata indices 0..init_n-1
    perms = []
    for _ in range(dim):
        p = list(range(init_n))
        random.shuffle(p)
        perms.append(p)

    for j in range(init_n):
        if time.time() >= deadline:
            return best
        x = []
        for i in range(dim):
            # sample inside stratum perms[i][j]
            u = (perms[i][j] + random.random()) / init_n
            x.append(lows[i] + u * spans[i])
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    # If initialization didn't run, start from random
    if best_x is None:
        best_x = rand_vec()
        best = eval_f(best_x)

    # --- adaptive local search with restarts ---
    # step size per dimension
    base_sigma = [0.2 * s if s > 0 else 1.0 for s in spans]
    sigma = base_sigma[:]

    x = best_x[:]
    fx = best

    # Parameters chosen to be robust and simple
    stagnation_limit = 40  # iterations without improvement before restart
    no_improve = 0

    # For 1/5 success rule (adapt every window)
    window = 20
    succ = 0
    trials = 0

    # Probability of doing a "big jump" move (helps escape basins)
    big_jump_prob = 0.08

    while True:
        if time.time() >= deadline:
            return best

        # Propose a candidate
        y = x[:]

        if random.random() < big_jump_prob:
            # big jump: mix current point with a random point
            r = rand_vec()
            alpha = 0.5 + 0.5 * random.random()  # closer to random than to current
            for i in range(dim):
                y[i] = clamp((1.0 - alpha) * x[i] + alpha * r[i], i)
        else:
            # local gaussian perturbation (Box-Muller)
            for i in range(dim):
                # generate standard normal via Box-Muller
                u1 = random.random()
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(max(1e-12, u1))) * math.cos(2.0 * math.pi * u2)
                y[i] = clamp(x[i] + sigma[i] * z, i)

        fy = eval_f(y)

        improved = fy < fx
        trials += 1
        if improved:
            x, fx = y, fy
            succ += 1
            no_improve = 0
            if fy < best:
                best, best_x = fy, y
        else:
            no_improve += 1

        # Adapt step sizes using 1/5 success rule
        if trials >= window:
            rate = succ / float(trials)
            # If success rate > 0.2, increase sigma; else decrease
            if rate > 0.2:
                factor = 1.25
            else:
                factor = 0.8
            for i in range(dim):
                sigma[i] *= factor
                # keep sigma within reasonable bounds
                min_sig = 1e-12 * (spans[i] if spans[i] > 0 else 1.0)
                max_sig = spans[i] if spans[i] > 0 else 1.0
                if sigma[i] < min_sig:
                    sigma[i] = min_sig
                elif sigma[i] > max_sig:
                    sigma[i] = max_sig
            succ = 0
            trials = 0

        # Restart if stuck
        if no_improve >= stagnation_limit:
            # restart near best with moderate noise, or fully random
            if random.random() < 0.7 and best_x is not None:
                x = best_x[:]
                for i in range(dim):
                    # moderate reset
                    u1 = random.random()
                    u2 = random.random()
                    z = math.sqrt(-2.0 * math.log(max(1e-12, u1))) * math.cos(2.0 * math.pi * u2)
                    x[i] = clamp(x[i] + 0.3 * base_sigma[i] * z, i)
            else:
                x = rand_vec()
            fx = eval_f(x)
            if fx < best:
                best, best_x = fx, x
            sigma = base_sigma[:]  # reset step sizes
            no_improve = 0
