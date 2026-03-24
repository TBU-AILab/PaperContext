import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-limited minimization using a self-contained hybrid:
    - low-discrepancy-like sampling (stratified via per-dimension random shifts),
    - local coordinate search around the current best,
    - adaptive step size with occasional restarts.

    Returns:
        best (float): best (minimum) fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # --- helpers ---
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Van der Corput sequence for base b (quasi-random 1D)
    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, r = divmod(n, base)
            denom *= base
            v += r / denom
        return v

    # Generate quasi-random point using different bases + random shift
    primes = []
    # small prime generator (enough for typical dims)
    cand = 2
    while len(primes) < max(1, dim):
        is_p = True
        r = int(math.sqrt(cand))
        for p in range(2, r + 1):
            if cand % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(cand)
        cand += 1

    # per-dimension random shifts (Cranley-Patterson rotation style)
    shifts = [random.random() for _ in range(dim)]

    def quasi_point(k):
        # k starts at 1 to avoid vdc(0)=0 for all dims
        x = []
        for i in range(dim):
            u = (vdc(k, primes[i]) + shifts[i]) % 1.0
            x.append(lows[i] + u * spans[i])
        return x

    # safe evaluation (in case func raises)
    def evaluate(x):
        try:
            return float(func(x))
        except Exception:
            return float("inf")

    # --- initialization ---
    best = float("inf")
    best_x = None

    # initial step sizes: fraction of span per dimension
    base_step = [0.25 * spans[i] if spans[i] > 0 else 1.0 for i in range(dim)]
    step = base_step[:]

    # control parameters
    shrink = 0.75         # step shrink factor when stuck
    expand = 1.15         # step expand factor on improvement (mild)
    min_step_frac = 1e-6  # stop shrinking below this fraction of span
    restart_prob = 0.05   # occasional random restart to escape local minima

    k = 1  # index for quasi points

    # Do a short burst of quasi-random exploration at the start
    init_budget = max(10, 5 * dim)
    for _ in range(init_budget):
        if time.time() >= deadline:
            return best
        x = quasi_point(k)
        k += 1
        f = evaluate(x)
        if f < best:
            best, best_x = f, x

    if best_x is None:
        # fallback (shouldn't happen unless func always errors)
        return best

    # --- main loop ---
    # Interleave: quasi-random exploration, then local coordinate refinements
    no_improve_iters = 0
    while time.time() < deadline:
        # occasional restart around a new quasi point
        if random.random() < restart_prob:
            x = quasi_point(k)
            k += 1
            f = evaluate(x)
            if f < best:
                best, best_x = f, x
                step = base_step[:]
            continue

        improved = False

        # Local coordinate search around best_x
        # Randomized dimension order
        dims = list(range(dim))
        random.shuffle(dims)

        for i in dims:
            if time.time() >= deadline:
                return best

            xi = best_x[i]
            si = step[i]

            # try both directions
            for direction in (-1.0, 1.0):
                cand = best_x[:]
                cand[i] = clip(xi + direction * si, i)
                f = evaluate(cand)
                if f < best:
                    best, best_x = f, cand
                    improved = True

                    # mild step expansion for this coordinate
                    step[i] = min(step[i] * expand, spans[i] if spans[i] > 0 else step[i] * expand)
                    break  # re-randomize dims after improvement
            if improved:
                break

        if improved:
            no_improve_iters = 0
            continue

        # If not improved locally, do one exploratory quasi-random sample
        if time.time() >= deadline:
            return best
        x = quasi_point(k)
        k += 1
        f = evaluate(x)
        if f < best:
            best, best_x = f, x
            step = base_step[:]
            no_improve_iters = 0
            continue

        no_improve_iters += 1

        # adaptively shrink steps if stuck
        if no_improve_iters >= max(5, dim):
            for i in range(dim):
                step[i] *= shrink
                # don't shrink below a tiny fraction of span (or absolute tiny)
                min_step = (spans[i] * min_step_frac) if spans[i] > 0 else 1e-12
                if step[i] < min_step:
                    step[i] = min_step
            no_improve_iters = 0

    return best
