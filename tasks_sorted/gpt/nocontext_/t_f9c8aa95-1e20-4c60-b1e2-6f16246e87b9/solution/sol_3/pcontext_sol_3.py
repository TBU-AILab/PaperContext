import time
import random
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid of:
      - quasi-random (Halton) initialization
      - local refinement via coordinate search (pattern search)
      - occasional random restarts

    Inputs:
      func   : callable(params_list_or_array) -> float
      dim    : int
      bounds : list of (low, high) for each dimension
      max_time : seconds (int/float)

    Returns:
      best fitness found (float)
    """

    # ----------------------------
    # Helpers (no external libs)
    # ----------------------------
    start = time.time()
    deadline = start + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x):
        # clamp vector to bounds
        y = list(x)
        for i in range(dim):
            if y[i] < lows[i]:
                y[i] = lows[i]
            elif y[i] > highs[i]:
                y[i] = highs[i]
        return y

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        # If user func throws or returns non-finite, treat as bad
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # Halton sequence for quasi-random coverage
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
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

    primes = first_primes(dim)

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

    def halton_point(k):
        # k starts at 1
        x = []
        for i in range(dim):
            u = halton(k, primes[i])
            x.append(lows[i] + u * spans[i])
        return x

    # ----------------------------
    # Algorithm parameters
    # ----------------------------
    # initial step sizes (fraction of range)
    step = [0.2 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
    min_step = 1e-12  # absolute min step guard
    shrink = 0.6      # step size shrink factor when stuck
    grow = 1.05       # mild growth if improving
    patience = 30     # iterations without improvement before restart
    max_halton = 200  # try a number of quasi-random seeds early

    # ----------------------------
    # Initialization: best of several seeds
    # ----------------------------
    best = float("inf")
    best_x = None

    k = 1
    while time.time() < deadline and k <= max_halton:
        x = halton_point(k)
        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x
        k += 1

    # Fallback if halton was cut short immediately
    if best_x is None:
        best_x = rand_point()
        best = safe_eval(best_x)

    # ----------------------------
    # Local search loop (pattern search with restarts)
    # ----------------------------
    no_improve = 0
    it = 0

    while time.time() < deadline:
        it += 1
        improved = False

        # coordinate-wise exploratory moves
        # randomize order to avoid bias
        order = list(range(dim))
        random.shuffle(order)

        x0 = best_x
        f0 = best

        for i in order:
            if time.time() >= deadline:
                return best

            if step[i] <= 0:
                continue

            # try +step and -step
            for direction in (1.0, -1.0):
                cand = list(x0)
                cand[i] += direction * step[i]
                cand = clip(cand)
                fc = safe_eval(cand)
                if fc < best:
                    best = fc
                    best_x = cand
                    improved = True
                    break  # accept first improvement for this coordinate
            if improved:
                # move base point for subsequent coordinates
                x0 = best_x
                f0 = best

        if improved:
            no_improve = 0
            # gently grow steps to speed along valleys
            for i in range(dim):
                step[i] *= grow
                # cap step to range
                if step[i] > spans[i]:
                    step[i] = spans[i]
        else:
            no_improve += 1
            # shrink steps when stuck
            for i in range(dim):
                step[i] *= shrink
                if step[i] < min_step:
                    step[i] = min_step

        # occasional random restart to escape local minima
        if no_improve >= patience and time.time() < deadline:
            # restart around current best with a mix of random and gaussian-like perturbation
            x = list(best_x)
            for i in range(dim):
                # triangular distribution (sum of uniforms) as a lightweight "normal-ish" perturbation
                u = (random.random() + random.random() + random.random()) / 3.0  # in [0,1]
                u = (u - 0.5) * 2.0  # in [-1,1]
                radius = 0.5 * spans[i]
                x[i] += u * radius
            x = clip(x)
            fx = safe_eval(x)
            if fx < best:
                best, best_x = fx, x

            # also sometimes jump to a fresh random point
            if random.random() < 0.5:
                x = rand_point()
                fx = safe_eval(x)
                if fx < best:
                    best, best_x = fx, x

            # reset steps to a moderate size after restart
            step = [0.2 * spans[i] if spans[i] > 0 else 0.0 for i in range(dim)]
            no_improve = 0

    return best
