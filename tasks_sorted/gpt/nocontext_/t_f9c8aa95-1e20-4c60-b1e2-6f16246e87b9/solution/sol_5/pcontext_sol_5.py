import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using:
      - Latin-hypercube-like stratified initialization
      - Local coordinate search (pattern search)
      - Occasional random restarts
      - Step-size adaptation
    No external libraries required.
    """

    # ----------------- helpers -----------------
    def clip(x, lo, hi):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def clamp_vec(v):
        return [clip(v[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def vec_add(v, i, delta):
        u = v[:]  # copy
        u[i] += delta
        return u

    def safe_eval(x):
        # func is expected to accept an "array-like" of length dim
        try:
            y = func(x)
            # handle NaN / inf robustly
            if y is None or (isinstance(y, float) and (math.isnan(y) or math.isinf(y))):
                return float("inf")
            return float(y)
        except Exception:
            return float("inf")

    def lhs_batch(n):
        # Simple LHS-style stratified sampling per dimension:
        # For each dim, pick one point from each of n strata and permute.
        # Produces n points.
        strata = []
        for d in range(dim):
            lo, hi = bounds[d]
            width = hi - lo
            # pick one random position in each stratum
            vals = [lo + (k + random.random()) * (width / n) for k in range(n)]
            random.shuffle(vals)
            strata.append(vals)
        pts = []
        for k in range(n):
            pts.append([strata[d][k] for d in range(dim)])
        return pts

    # ----------------- initialization -----------------
    t0 = time.time()
    deadline = t0 + max_time

    # global best
    best_x = None
    best = float("inf")

    # initial step sizes ~ 10% of range per dimension
    step0 = []
    for lo, hi in bounds:
        r = hi - lo
        step0.append(0.1 * r if r > 0 else 1.0)

    # Evaluate an initial batch quickly to get a decent start
    # batch size scales mildly with dimension, but stays small for time-bounded runs
    init_n = max(8, min(40, 10 + 2 * dim))
    for x in lhs_batch(init_n):
        if time.time() >= deadline:
            return best
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # fallback
        best_x = rand_uniform_vec()
        best = safe_eval(best_x)

    # ----------------- main loop: local search + restarts -----------------
    x = best_x[:]
    fx = best
    step = step0[:]

    # parameters
    min_step_factor = 1e-8
    shrink = 0.5
    grow = 1.2
    restart_prob = 0.02  # occasional restart
    max_no_improve = 5 * dim + 10

    no_improve = 0

    while time.time() < deadline:
        # occasional random restart around global best or fully random
        if random.random() < restart_prob:
            if random.random() < 0.7 and best_x is not None:
                # restart near best
                x = best_x[:]
                # jitter
                for i in range(dim):
                    lo, hi = bounds[i]
                    r = (hi - lo)
                    jitter = (random.uniform(-1.0, 1.0) * 0.05 * r) if r > 0 else 0.0
                    x[i] = clip(x[i] + jitter, lo, hi)
            else:
                x = rand_uniform_vec()
            fx = safe_eval(x)
            if fx < best:
                best, best_x = fx, x[:]
            no_improve = 0

        improved_this_round = False

        # coordinate pattern search: try +/- step in each dimension
        # randomize order to avoid bias
        order = list(range(dim))
        random.shuffle(order)

        for i in order:
            if time.time() >= deadline:
                return best

            lo, hi = bounds[i]
            si = step[i]

            if si <= (hi - lo) * min_step_factor and (hi - lo) > 0:
                continue

            # try positive move
            xp = vec_add(x, i, si)
            xp = clamp_vec(xp)
            f_p = safe_eval(xp)

            # try negative move
            xm = vec_add(x, i, -si)
            xm = clamp_vec(xm)
            f_m = safe_eval(xm)

            # pick best among current and candidates
            if f_p < fx or f_m < fx:
                if f_p <= f_m:
                    x, fx = xp, f_p
                else:
                    x, fx = xm, f_m
                improved_this_round = True
                step[i] *= grow  # slightly increase step on success

                if fx < best:
                    best, best_x = fx, x[:]
            else:
                step[i] *= shrink  # shrink step on failure

        if improved_this_round:
            no_improve = 0
        else:
            no_improve += 1

        # if stuck, do a broader random probe around best
        if no_improve >= max_no_improve:
            no_improve = 0
            # reset steps
            step = step0[:]
            # probe a few random points biased to best
            probes = 5 + dim
            for _ in range(probes):
                if time.time() >= deadline:
                    return best
                if best_x is None:
                    cand = rand_uniform_vec()
                else:
                    cand = best_x[:]
                    for i in range(dim):
                        lo, hi = bounds[i]
                        r = hi - lo
                        # heavier-tailed perturbation (mixture)
                        scale = 0.1 * r if r > 0 else 1.0
                        if random.random() < 0.2:
                            delta = random.uniform(-1, 1) * 0.5 * r
                        else:
                            delta = random.uniform(-1, 1) * scale
                        cand[i] = clip(cand[i] + delta, lo, hi)
                f_c = safe_eval(cand)
                if f_c < best:
                    best, best_x = f_c, cand[:]
                    x, fx = best_x[:], best

    return best
