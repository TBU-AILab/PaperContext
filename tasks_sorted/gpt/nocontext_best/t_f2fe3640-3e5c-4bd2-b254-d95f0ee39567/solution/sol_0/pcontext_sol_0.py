import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a simple, robust hybrid:
    - quasi-random (Halton) exploration to cover the space quickly
    - local Gaussian perturbation around the incumbent (step-size adapted)
    - occasional random restarts to escape local minima

    Returns:
        best (float): best (minimum) fitness found within max_time seconds
    """

    # ---------------------- helpers (no external libs) ----------------------
    def clip(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi: 
            return hi
        return x

    def make_in_bounds(vec):
        return [clip(vec[i], bounds[i][0], bounds[i][1]) for i in range(dim)]

    # First primes for Halton; extend if dim is larger
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

    def van_der_corput(index, base):
        # index >= 1
        vdc, denom = 0.0, 1.0
        while index > 0:
            index, remainder = divmod(index, base)
            denom *= base
            vdc += remainder / denom
        return vdc

    def halton_point(k, bases):
        # k >= 1
        u = [van_der_corput(k, b) for b in bases]
        # scale to bounds
        pt = []
        for i in range(dim):
            lo, hi = bounds[i]
            pt.append(lo + u[i] * (hi - lo))
        return pt

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def eval_f(x):
        # func expects an array-like; a Python list is typically fine
        return float(func(x))

    # ---------------------- initialization ----------------------
    start = time.time()
    deadline = start + max_time

    bases = first_primes(dim)

    # initial incumbent from a few space-filling samples
    best = float("inf")
    best_x = None

    k = 1
    init_samples = max(10, 5 * dim)
    for _ in range(init_samples):
        if time.time() >= deadline:
            return best
        x = halton_point(k, bases)
        k += 1
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    # per-dimension step sizes (start with 10% of range)
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    sigma = [0.10 * r if r > 0 else 1.0 for r in ranges]
    min_sigma = [1e-12 * (r if r > 0 else 1.0) for r in ranges]

    # control parameters
    # probability of doing local step vs. global sample
    p_local = 0.75
    # probability of restart if stagnating
    restart_after = 50 + 10 * dim
    no_improve = 0

    # ---------------------- main loop ----------------------
    while True:
        if time.time() >= deadline:
            return best

        do_local = (best_x is not None) and (random.random() < p_local)

        if do_local:
            # local mutation around incumbent
            cand = []
            for i in range(dim):
                # Gaussian step with current sigma
                step = random.gauss(0.0, sigma[i])
                cand.append(best_x[i] + step)
            cand = make_in_bounds(cand)
        else:
            # global exploration: mix Halton and pure random
            if random.random() < 0.70:
                cand = halton_point(k, bases)
                k += 1
            else:
                cand = rand_point()

        f_cand = eval_f(cand)

        if f_cand < best:
            best, best_x = f_cand, cand
            no_improve = 0
            # slightly expand to move faster when improving
            for i in range(dim):
                sigma[i] *= 1.05
        else:
            no_improve += 1
            # contract when not improving to refine locally
            if do_local:
                for i in range(dim):
                    sigma[i] *= 0.985
                    if sigma[i] < min_sigma[i]:
                        sigma[i] = min_sigma[i]

        # random restart / diversification if stuck
        if no_improve >= restart_after:
            no_improve = 0

            # shrink local step (exploit) but jump globally (explore)
            for i in range(dim):
                sigma[i] = max(sigma[i] * 0.5, min_sigma[i])

            # evaluate a few global candidates and take the best as new center
            trials = 3 + dim // 2
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                if random.random() < 0.7:
                    x = halton_point(k, bases)
                    k += 1
                else:
                    x = rand_point()
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x

    # (unreachable) return best
