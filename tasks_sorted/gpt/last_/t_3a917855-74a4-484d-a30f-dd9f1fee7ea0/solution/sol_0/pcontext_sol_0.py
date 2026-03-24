import random
import math
import time

def run(func, dim, bounds, max_time):
    # Time management
    start = time.time()
    deadline = start + max_time

    # Helpers
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def ensure_bounds(x):
        for i in range(dim):
            x[i] = clamp(x[i], bounds[i][0], bounds[i][1])
        return x

    def eval_f(x):
        # func is specified to accept an array-like of parameters
        return float(func(x))

    # --- Hybrid optimizer: low-discrepancy init + adaptive local search (1+λ ES) + occasional restarts ---

    # Halton sequence (no external libs) for better-than-random coverage early on
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton(index, base):
        # index >= 1
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(dim)

    def halton_vec(k):
        x = []
        for i in range(dim):
            u = halton(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # Initial best
    best_x = rand_vec()
    best = eval_f(best_x)

    # Scale per dimension
    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # Strategy parameters
    lam = 8 if dim <= 10 else 12  # offspring per iteration
    sigma = 0.2                   # relative step size (w.r.t. span)
    sigma_min = 1e-12
    sigma_max = 0.5
    stall_limit = 60              # iterations without improvement before a restart
    stall = 0

    # Budget early: sample a small Halton batch for quick global coverage
    k = 1
    halton_budget = max(20, 10 * dim)

    # Main loop
    while True:
        if time.time() >= deadline:
            return best

        # 1) Global exploration phase (quasi-random)
        if k <= halton_budget:
            x = halton_vec(k)
            k += 1
            f = eval_f(x)
            if f < best:
                best = f
                best_x = x
                stall = 0
            else:
                stall += 1
            continue

        # 2) Local search around current best (1+λ evolution strategy)
        improved = False
        best_off_f = best
        best_off_x = None

        # Generate λ candidates
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            cand = best_x[:]  # copy
            # Gaussian perturbation with per-dim scaling
            for i in range(dim):
                step = random.gauss(0.0, sigma) * span[i]
                cand[i] += step
                # Reflect at boundaries (better than clamp for exploration near edges)
                lo, hi = bounds[i]
                if cand[i] < lo or cand[i] > hi:
                    # reflection
                    width = hi - lo
                    if width <= 0:
                        cand[i] = lo
                    else:
                        # map to [0, 2w) then reflect
                        t = (cand[i] - lo) % (2.0 * width)
                        cand[i] = (lo + t) if t <= width else (hi - (t - width))

            cand = ensure_bounds(cand)
            f = eval_f(cand)

            if f < best_off_f:
                best_off_f = f
                best_off_x = cand
                improved = True

        # Adaptation: 1/5 success rule style
        if improved and best_off_x is not None:
            best_x = best_off_x
            best = best_off_f
            stall = 0
            sigma = min(sigma_max, sigma * 1.15)
        else:
            stall += 1
            sigma = max(sigma_min, sigma * 0.85)

        # 3) Restart if stuck: jump to a new region, keep incumbent best
        if stall >= stall_limit:
            stall = 0
            sigma = 0.2
            # biased restart: mix uniform random with best (keeps some structure)
            x = rand_vec()
            mix = 0.3
            restarted = [mix * best_x[i] + (1.0 - mix) * x[i] for i in range(dim)]
            restarted = ensure_bounds(restarted)
            f = eval_f(restarted)
            if f < best:
                best = f
                best_x = restarted
