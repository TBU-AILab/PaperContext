import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a simple, robust hybrid:
    1) Low-discrepancy-ish stratified sampling (good initial coverage)
    2) Local improvement via coordinate perturbations + adaptive step sizes
    3) Occasional random restarts to escape local minima

    Returns:
        best (float): best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    eps_time = 1e-4

    # --------- helpers ----------
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # van der Corput for bases (for simple quasi-random sequence)
    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, r = divmod(n, base)
            denom *= base
            v += r / denom
        return v

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    bases = [primes[i % len(primes)] for i in range(dim)]

    def quasi_vec(k):
        # k starts from 1 to avoid all-zeros
        x = []
        for i in range(dim):
            u = vdc(k, bases[i])
            x.append(lows[i] + u * spans[i])
        return x

    def evaluate(x):
        # func expects an array-like; we pass a python list
        return float(func(x))

    # --------- main search ----------
    best = float("inf")
    best_x = None

    # Initial step size: fraction of range per dimension
    step = [0.15 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    min_step = [1e-12 * (spans[i] if spans[i] > 0 else 1.0) for i in range(dim)]

    # Budget heuristics derived from time (kept lightweight)
    # We'll adapt by checking time frequently.
    k = 1
    restarts = 0

    # Current point
    x = quasi_vec(k)
    k += 1
    fx = evaluate(x)
    best, best_x = fx, x[:]

    # Parameters controlling exploration/exploitation
    p_restart = 0.02          # chance to restart when stagnating
    p_global = 0.05           # chance to sample a global point
    shrink = 0.72             # step shrink factor on stagnation
    grow = 1.08               # mild growth when improving
    stagnation_limit = 40     # iterations without improvement before shrinking/restart

    stagnation = 0
    it = 0

    while True:
        if time.time() - t0 >= max_time - eps_time:
            return best

        it += 1

        # Occasionally do a global move (quasi-random then uniform)
        r = random.random()
        if r < p_global:
            if random.random() < 0.6:
                cand = quasi_vec(k)
                k += 1
            else:
                cand = rand_uniform_vec()
            fc = evaluate(cand)
            if fc < best:
                best, best_x = fc, cand[:]
                x, fx = cand, fc
                stagnation = 0
                # grow steps slightly on improvements
                for i in range(dim):
                    step[i] *= grow
            else:
                stagnation += 1
            continue

        # Local search: coordinate-wise perturbation around current/best
        # Mix current point with best to stabilize
        if best_x is not None and random.random() < 0.5:
            base = best_x
        else:
            base = x

        # Create candidate by perturbing a subset of coordinates
        cand = base[:]
        # number of perturbed dims: 1..min(8, dim) biased small
        m = 1 + int((min(8, dim) - 1) * (random.random() ** 2))
        idxs = random.sample(range(dim), m) if dim > 1 else [0]

        for j in idxs:
            if spans[j] <= 0:
                continue
            # symmetric perturbation with occasional larger jumps
            s = step[j]
            if random.random() < 0.15:
                delta = (random.random() * 2.0 - 1.0) * (3.0 * s)
            else:
                delta = (random.random() * 2.0 - 1.0) * s
            cand[j] += delta

        clamp(cand)
        fc = evaluate(cand)

        if fc < fx or (fc < best and random.random() < 0.9):
            # accept improvement w.r.t current, or strong improvement
            x, fx = cand, fc
        if fc < best:
            best, best_x = fc, cand[:]
            stagnation = 0
            for i in range(dim):
                step[i] *= grow
        else:
            stagnation += 1

        # If stagnating, shrink step sizes; sometimes restart
        if stagnation >= stagnation_limit:
            stagnation = 0
            for i in range(dim):
                step[i] = max(step[i] * shrink, min_step[i])

            # restart from a new global point with some probability
            if random.random() < p_restart:
                restarts += 1
                if random.random() < 0.7:
                    x = quasi_vec(k); k += 1
                else:
                    x = rand_uniform_vec()
                fx = evaluate(x)
                if fx < best:
                    best, best_x = fx, x[:]

            # also slightly increase global sampling chance after many restarts
            if restarts > 0 and restarts % 5 == 0:
                p_global = min(0.20, p_global * 1.15)
