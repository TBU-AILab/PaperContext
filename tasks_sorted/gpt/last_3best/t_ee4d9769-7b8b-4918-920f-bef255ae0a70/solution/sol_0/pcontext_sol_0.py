import random
import time
import math

def run(func, dim, bounds, max_time):
    # Self-contained derivative-free optimizer:
    # Hybrid of:
    # - Latin-hypercube-like stratified initialization (per-dimension shuffles)
    # - Local search with adaptive step sizes (coordinate + random directions)
    # - Occasional restarts to escape local minima
    #
    # Returns: best (float) = best fitness found within time budget.

    start = time.time()
    deadline = start + max_time

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

    def eval_f(x):
        # func expects an "array-like"; we pass a plain Python list.
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def lhs_points(n):
        # Simple LHS-style: stratify each dimension and randomly permute bins.
        # Produces n points in [0,1]^dim then scales to bounds.
        perms = []
        for j in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)

        pts = []
        for i in range(n):
            u = []
            for j in range(dim):
                # pick within stratum [k/n, (k+1)/n)
                k = perms[j][i]
                uj = (k + random.random()) / n
                u.append(lows[j] + uj * spans[j])
            pts.append(u)
        return pts

    # --- initial exploration budget ---
    # Keep small to preserve time for local search; scale with dimension.
    init_n = max(8, min(64, 8 * dim))

    best_x = None
    best = float("inf")

    # Evaluate initial stratified set
    for x in lhs_points(init_n):
        if time.time() >= deadline:
            return best
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = x

    # Fallback if something went wrong
    if best_x is None:
        best_x = rand_point()
        best = eval_f(best_x)

    # --- adaptive local search parameters ---
    # Start step: fraction of domain span; keep a minimum absolute epsilon
    step = [0.15 * s if s > 0 else 1.0 for s in spans]
    min_step = [max(1e-12, 1e-9 * (s if s > 0 else 1.0)) for s in spans]
    max_step = [0.5 * (s if s > 0 else 1.0) for s in spans]

    # Acceptance tracking for adapting step sizes
    accepted = 0
    tried = 0

    # Restart control
    last_improve_time = time.time()
    stall_seconds = max(0.15 * max_time, 0.25)  # restart if no improvement for a while

    # Main loop
    while time.time() < deadline:
        # If stalled, restart near best with larger perturbation, or global random
        if time.time() - last_improve_time > stall_seconds:
            if random.random() < 0.7:
                # local restart around best_x
                x = best_x[:]
                for i in range(dim):
                    # perturb with current step * random factor
                    sigma = step[i] * (0.5 + 1.5 * random.random())
                    x[i] = clip(x[i] + random.gauss(0.0, sigma), i)
            else:
                x = rand_point()

            fx = eval_f(x)
            if fx < best:
                best = fx
                best_x = x
                last_improve_time = time.time()

            # also mildly expand steps after restart
            for i in range(dim):
                step[i] = min(max_step[i], step[i] * 1.25)
            accepted = 0
            tried = 0
            continue

        # Choose move type: coordinate or random direction
        x0 = best_x
        cand = x0[:]

        if dim == 1 or random.random() < 0.65:
            # Coordinate move: pick one dimension and move +/- step
            j = random.randrange(dim)
            direction = -1.0 if random.random() < 0.5 else 1.0
            cand[j] = clip(cand[j] + direction * step[j] * (0.5 + random.random()), j)
        else:
            # Random direction move: normalized Gaussian direction
            v = [random.gauss(0.0, 1.0) for _ in range(dim)]
            norm = math.sqrt(sum(vi * vi for vi in v)) or 1.0
            # scale by mean step
            mean_step = sum(step) / dim
            scale = mean_step * (0.5 + random.random())
            for i in range(dim):
                cand[i] = clip(cand[i] + scale * (v[i] / norm), i)

        tried += 1
        fc = eval_f(cand)

        # Accept if improves; small chance to accept worse (very mild SA) early on
        if fc < best:
            best = fc
            best_x = cand
            accepted += 1
            last_improve_time = time.time()
        else:
            # mild probabilistic acceptance to escape shallow traps
            # probability decreases as time passes
            tfrac = (time.time() - start) / max(1e-12, max_time)
            temp = max(1e-12, 0.05 * (1.0 - tfrac))
            if temp > 1e-12:
                # Accept with exp(-(delta)/temp_scale) where temp_scale ~ (|best|+1)
                delta = fc - best
                scale = (abs(best) + 1.0)
                p = math.exp(-delta / (temp * scale))
                if random.random() < p:
                    best_x = cand  # move the state, but keep global best unchanged

        # Adapt steps every so often
        if tried >= (10 + 2 * dim):
            acc_rate = accepted / float(tried)

            # If too many accepts, increase step; if too few, decrease step
            if acc_rate > 0.25:
                factor = 1.15
            elif acc_rate < 0.10:
                factor = 0.70
            else:
                factor = 0.95

            for i in range(dim):
                step[i] *= factor
                if step[i] < min_step[i]:
                    step[i] = min_step[i]
                if step[i] > max_step[i]:
                    step[i] = max_step[i]

            accepted = 0
            tried = 0

    return best
