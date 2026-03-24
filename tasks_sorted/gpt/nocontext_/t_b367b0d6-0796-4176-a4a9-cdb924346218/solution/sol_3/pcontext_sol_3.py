import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization
    - (1+1)-ES style local search with adaptive step-size (success rule)
    - occasional coordinate search / random restart for robustness

    Returns: best (float) = best objective value found within max_time seconds.
    """

    # ---------- helpers ----------
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clamp(x):
        return [min(hi[i], max(lo[i], x[i])) for i in range(dim)]

    def rand_vec():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; we pass a plain list (no external libs)
        return float(func(x))

    def gaussian_step(x, sigma):
        # perturb each coordinate with N(0, sigma_i^2)
        y = [x[i] + random.gauss(0.0, sigma[i]) for i in range(dim)]
        return clamp(y)

    # Stratified sampling per-dimension (LHS-ish without numpy)
    def lhs_samples(n):
        # for each dimension, create n strata samples and shuffle
        per_dim = []
        for i in range(dim):
            vals = []
            for k in range(n):
                a = k / n
                b = (k + 1) / n
                u = a + (b - a) * random.random()
                vals.append(lo[i] + u * span[i])
            random.shuffle(vals)
            per_dim.append(vals)
        # combine
        samples = []
        for k in range(n):
            samples.append([per_dim[i][k] for i in range(dim)])
        return samples

    # ---------- time setup ----------
    t_end = time.time() + max_time
    if max_time <= 0:
        return float("inf")

    # ---------- initialization ----------
    best = float("inf")
    best_x = None

    # initial sample count scales mildly with dim, but stays small for speed
    n_init = max(8, min(40, 4 * dim))
    for x in lhs_samples(n_init):
        if time.time() >= t_end:
            return best
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        # fallback (shouldn't happen)
        best_x = rand_vec()
        best = evaluate(best_x)

    # ---------- main search state ----------
    # per-dimension step sizes
    sigma = [0.15 * s if s > 0 else 1.0 for s in span]
    sigma_min = [1e-12 * (s if s > 0 else 1.0) for s in span]
    sigma_max = [0.5 * (s if s > 0 else 1.0) for s in span]

    x = best_x[:]
    fx = best

    # parameters for adaptation
    # (roughly) 1/5th success rule with smoothing
    success_ema = 0.2
    ema_alpha = 0.15  # smoothing
    target = 0.2

    # control for restarts / diversification
    no_improve = 0
    restart_after = 200 + 40 * dim

    # coordinate search step multiplier
    coord_factor = 0.75

    # ---------- optimization loop ----------
    while time.time() < t_end:
        # Choose move type: mostly gaussian, sometimes coordinate, rarely restart
        r = random.random()

        improved = False

        if r < 0.80:
            # (1+1)-ES Gaussian mutation
            y = gaussian_step(x, sigma)
            fy = evaluate(y)
            if fy <= fx:
                x, fx = y, fy
                improved = True

        elif r < 0.97:
            # coordinate-wise probing (helps on separable / ridge problems)
            i = random.randrange(dim)
            step = coord_factor * sigma[i]
            # try both directions
            y1 = x[:]
            y1[i] = min(hi[i], x[i] + step)
            f1 = evaluate(y1)

            y2 = x[:]
            y2[i] = max(lo[i], x[i] - step)
            f2 = evaluate(y2)

            if f1 <= f2:
                y, fy = y1, f1
            else:
                y, fy = y2, f2

            if fy <= fx:
                x, fx = y, fy
                improved = True

        else:
            # random restart near current best or global
            if random.random() < 0.7 and best_x is not None:
                # restart around best_x
                y = [
                    best_x[i] + random.gauss(0.0, 0.25 * span[i]) for i in range(dim)
                ]
                y = clamp(y)
            else:
                y = rand_vec()
            fy = evaluate(y)
            if fy <= fx:
                x, fx = y, fy
                improved = True

        # update global best
        if fx < best:
            best = fx
            best_x = x[:]
            no_improve = 0
        else:
            no_improve += 1

        # update success rate EMA and adapt sigmas
        success = 1.0 if improved else 0.0
        success_ema = (1.0 - ema_alpha) * success_ema + ema_alpha * success

        # Increase steps if too successful, decrease if not successful enough
        # Multiplicative update keeps positivity and scale-invariance
        if success_ema > target:
            mult = 1.08
        else:
            mult = 0.92

        for i in range(dim):
            si = sigma[i] * mult
            if si < sigma_min[i]:
                si = sigma_min[i]
            elif si > sigma_max[i]:
                si = sigma_max[i]
            sigma[i] = si

        # forced diversification if stuck
        if no_improve >= restart_after:
            # jump to a new region, reset step sizes moderately
            x = rand_vec()
            fx = evaluate(x)
            sigma = [0.20 * s if s > 0 else 1.0 for s in span]
            no_improve = 0
            # keep best/best_x unchanged

    return best
