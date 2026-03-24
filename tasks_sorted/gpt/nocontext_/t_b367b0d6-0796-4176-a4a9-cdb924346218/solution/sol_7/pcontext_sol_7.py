import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded optimizer: hybrid of
      - Sobol-like (Halton) low-discrepancy sampling for global exploration
      - (1+1)-ES style local search with adaptive step-size (success-based)
      - occasional random restarts
    Returns: best (float) = best objective value found within max_time seconds
    """

    # ---------- utilities ----------
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def reflect_into_bounds(x, lo, hi):
        # Reflect x into [lo, hi] (works even if far outside by repeated reflection).
        if lo == hi:
            return lo
        span = hi - lo
        y = x
        # Bring into [lo, hi] using reflection on boundaries.
        # Use modulus on 2*span for efficiency.
        t = (y - lo) % (2.0 * span)
        if t <= span:
            return lo + t
        else:
            return hi - (t - span)

    def safe_eval(vec):
        # func is defined by caller; ensure we pass a plain list of floats
        try:
            val = func([float(v) for v in vec])
            if val is None:
                return float("inf")
            val = float(val)
            if math.isnan(val) or math.isinf(val):
                return float("inf")
            return val
        except Exception:
            return float("inf")

    # Halton sequence for quasi-random coverage (no external libs)
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def halton_index(i, base):
        # radical inverse in given base
        f = 1.0
        r = 0.0
        x = i
        while x > 0:
            f /= base
            r += f * (x % base)
            x //= base
        return r

    def halton_point(i, bases):
        return [halton_index(i, b) for b in bases]

    # ---------- initialization ----------
    t_end = time.time() + float(max_time)
    rng = random.Random()

    # precompute scaling
    lows = [float(bounds[j][0]) for j in range(dim)]
    highs = [float(bounds[j][1]) for j in range(dim)]
    spans = [highs[j] - lows[j] for j in range(dim)]
    # avoid zero spans
    spans = [s if s != 0.0 else 1.0 for s in spans]

    # global best
    best = float("inf")
    best_x = None

    # quasi-random bases (first dim primes)
    bases = first_primes(dim)

    # decide budgets by time (no fixed iteration count)
    # start with some global exploration; later mostly local
    halton_i = 1

    # local search state
    # start from random point
    def random_point():
        return [lows[j] + rng.random() * (highs[j] - lows[j]) for j in range(dim)]

    x = random_point()
    fx = safe_eval(x)
    best, best_x = fx, list(x)

    # step size per dimension (fraction of span)
    sigma = [0.2 * spans[j] for j in range(dim)]
    min_sigma = [1e-12 * spans[j] for j in range(dim)]
    max_sigma = [0.5 * spans[j] for j in range(dim)]

    # success-based adaptation
    success = 0
    trials = 0
    adapt_every = 25  # adjust step sizes every N local trials

    # restart controls
    no_improve = 0
    restart_after = 200  # local trials without improving global best triggers restart
    p_global_sample = 0.25  # probability to do a global sample vs local step

    # ---------- main loop ----------
    while time.time() < t_end:
        # Choose global exploration or local refinement
        if rng.random() < p_global_sample or best_x is None:
            # Halton low-discrepancy sample (global)
            u = halton_point(halton_i, bases)
            halton_i += 1
            cand = [lows[j] + u[j] * (highs[j] - lows[j]) for j in range(dim)]
            fc = safe_eval(cand)

            if fc < best:
                best, best_x = fc, list(cand)
                x, fx = list(cand), fc
                no_improve = 0
            else:
                no_improve += 1

        else:
            # Local step around current best_x with Gaussian-like perturbation
            # Use Box-Muller for normal sampling (no numpy).
            def randn():
                u1 = max(1e-12, rng.random())
                u2 = rng.random()
                return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

            base = best_x if best_x is not None else x
            cand = []
            for j in range(dim):
                step = sigma[j] * randn()
                v = base[j] + step
                v = reflect_into_bounds(v, lows[j], highs[j])
                cand.append(v)

            fc = safe_eval(cand)
            trials += 1

            if fc <= best:
                best, best_x = fc, list(cand)
                x, fx = list(cand), fc
                success += 1
                no_improve = 0
            else:
                no_improve += 1

            # adapt step sizes periodically
            if trials % adapt_every == 0:
                rate = success / float(adapt_every)
                success = 0

                # If too successful, increase; if not, decrease
                # target success around ~0.2 for (1+1)-ES style
                if rate > 0.25:
                    factor = 1.25
                elif rate < 0.15:
                    factor = 0.75
                else:
                    factor = 1.0

                if factor != 1.0:
                    for j in range(dim):
                        sigma[j] = clamp(sigma[j] * factor, min_sigma[j], max_sigma[j])

                # slowly reduce global sampling as time goes (favor exploitation)
                # but never eliminate it
                remaining = max(0.0, t_end - time.time())
                total = max(1e-9, float(max_time))
                frac_left = remaining / total
                p_global_sample = 0.05 + 0.30 * frac_left  # from ~0.35 down to ~0.05

        # random restart if stuck
        if no_improve >= restart_after and time.time() < t_end:
            # restart near best (small chance) or fully random (mostly)
            if best_x is not None and rng.random() < 0.2:
                # jitter best a bit
                cand = []
                for j in range(dim):
                    width = 0.05 * spans[j]
                    v = best_x[j] + (rng.random() * 2.0 - 1.0) * width
                    v = reflect_into_bounds(v, lows[j], highs[j])
                    cand.append(v)
            else:
                cand = random_point()

            fc = safe_eval(cand)
            if fc < best:
                best, best_x = fc, list(cand)
            x, fx = list(cand), fc

            # reset local state
            sigma = [0.2 * spans[j] for j in range(dim)]
            success = 0
            trials = 0
            no_improve = 0

    return best
