import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a hybrid of:
    - Latin-hypercube-like diversified sampling (via stratified bins per dimension)
    - Local search with adaptive step size (coordinate + random directions)
    - Occasional random restarts
    No external libraries required.
    Returns: best (float) fitness found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---- helpers ----
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

    # Safe evaluation (in case func raises due to invalid params)
    def evaluate(x):
        try:
            v = func(list(x))
            # guard against NaN/inf
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return float("inf")
            return float(v)
        except Exception:
            return float("inf")

    def random_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Stratified sampling (simple LHS-style):
    # Choose a "bin index" per dimension, then sample within that bin.
    # Re-shuffle per restart.
    def stratified_point(bin_indices, bins):
        x = [0.0] * dim
        for i in range(dim):
            b = bin_indices[i]
            lo = lows[i] + (b / bins) * spans[i]
            hi = lows[i] + ((b + 1) / bins) * spans[i]
            x[i] = lo + random.random() * (hi - lo)
        return x

    # ---- initialization ----
    best_x = None
    best = float("inf")

    # Initial step sizes: fraction of range
    base_sigma = [0.15 * s if s > 0 else 1.0 for s in spans]
    sigma = base_sigma[:]

    # Determine bins for stratified sampling (small, time-friendly)
    bins = max(2, int(round(dim ** 0.5)) + 2)

    # Prebuild shuffled bin lists per dimension (cycled)
    bin_lists = [list(range(bins)) for _ in range(dim)]
    for i in range(dim):
        random.shuffle(bin_lists[i])
    bin_pos = [0] * dim

    # Seed with a few diversified samples
    init_samples = max(10, 5 * dim)
    for _ in range(init_samples):
        if time.time() >= deadline:
            return best
        # pick next bin index for each dimension
        inds = []
        for i in range(dim):
            inds.append(bin_lists[i][bin_pos[i]])
            bin_pos[i] = (bin_pos[i] + 1) % bins
            if bin_pos[i] == 0:
                random.shuffle(bin_lists[i])
        x = stratified_point(inds, bins)
        fx = evaluate(x)
        if fx < best:
            best = fx
            best_x = x

    # If everything failed, at least have a fallback
    if best_x is None:
        best_x = random_point()
        best = evaluate(best_x)

    # ---- main loop: adaptive local search with restarts ----
    # Parameters (kept simple & robust)
    no_improve = 0
    local_budget = 12 * dim  # attempts before considering restart/step shrink
    restart_threshold = 4 * local_budget

    while time.time() < deadline:
        x = best_x[:]  # start from current best

        # A "local phase"
        improved = False
        for _ in range(local_budget):
            if time.time() >= deadline:
                return best

            # Generate candidate: mix coordinate and random-direction moves
            cand = x[:]
            if dim == 1 or random.random() < 0.6:
                # coordinate move
                j = random.randrange(dim)
                step = random.gauss(0.0, sigma[j])
                cand[j] = cand[j] + step
            else:
                # random direction move
                for j in range(dim):
                    cand[j] = cand[j] + random.gauss(0.0, sigma[j])

            cand = clip(cand)
            fc = evaluate(cand)

            if fc < best:
                best = fc
                best_x = cand
                x = cand
                improved = True
                no_improve = 0

                # slightly expand steps on success (but not too much)
                for j in range(dim):
                    sigma[j] = min(base_sigma[j], sigma[j] * 1.05 + 1e-12)
            else:
                no_improve += 1

                # mild shrink on failures to focus search
                if no_improve % (2 * dim + 3) == 0:
                    for j in range(dim):
                        sigma[j] = max(1e-12, sigma[j] * 0.85)

        if not improved:
            # If stuck, either shrink more or restart
            if no_improve >= restart_threshold:
                # Restart with stratified/random mix, reset sigmas
                sigma = base_sigma[:]
                no_improve = 0

                if random.random() < 0.7:
                    # stratified restart
                    inds = []
                    for i in range(dim):
                        inds.append(bin_lists[i][bin_pos[i]])
                        bin_pos[i] = (bin_pos[i] + 1) % bins
                        if bin_pos[i] == 0:
                            random.shuffle(bin_lists[i])
                    rx = stratified_point(inds, bins)
                else:
                    rx = random_point()

                fr = evaluate(rx)
                if fr < best:
                    best = fr
                    best_x = rx
            else:
                # just shrink to intensify around best
                for j in range(dim):
                    sigma[j] = max(1e-12, sigma[j] * 0.7)

    return best
