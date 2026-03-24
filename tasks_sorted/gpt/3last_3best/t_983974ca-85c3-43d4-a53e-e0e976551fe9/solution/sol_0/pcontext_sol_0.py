import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Evolution Strategy:
    - Start with random sampling to get a decent initial point
    - Then (1+lambda)-ES with:
        * per-dimension step sizes (sigma)
        * log-normal step-size adaptation
        * occasional global restarts if stagnating
    Returns: best (float) = best fitness found
    """
    t0 = time.time()
    deadline = t0 + max_time

    # Helpers
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_vec(x):
        # Clamp to bounds (fast, no numpy)
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def safe_eval(x):
        # Guard against exceptions/NaNs/Infs
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

    # --- Phase 1: quick random sampling to seed a good start
    best_x = None
    best = float("inf")

    # budget a small portion of time for seeding
    seed_until = min(deadline, t0 + 0.15 * max_time)
    while time.time() < seed_until:
        x = rand_vec()
        f = safe_eval(x)
        if f < best:
            best = f
            best_x = x

    if best_x is None:
        best_x = rand_vec()
        best = safe_eval(best_x)

    # --- Phase 2: (1+lambda)-ES with step-size adaptation + restarts
    # Initial sigma: 20% of range per coordinate (with floor)
    sigma = [max(1e-12, 0.2 * s) for s in spans]

    # ES parameters
    lam = max(4, 4 + int(2 * math.log(dim + 1)))  # offspring count
    tau = 1.0 / math.sqrt(2.0 * dim)              # global learning rate
    tau0 = 1.0 / math.sqrt(2.0 * math.sqrt(dim))  # coordinate learning rate
    min_sigma = 1e-15
    max_sigma = max(spans) if spans else 1.0

    # Stagnation / restart control
    no_improve = 0
    stagnation_limit = 40 + 10 * dim

    parent_x = list(best_x)
    parent_f = best

    while time.time() < deadline:
        # If stagnating, restart around best with refreshed sigmas
        if no_improve >= stagnation_limit:
            # soft restart: random point with bias towards best
            x = rand_vec()
            # blend with best_x (keeps some exploitation)
            mix = 0.5
            for i in range(dim):
                x[i] = mix * best_x[i] + (1.0 - mix) * x[i]
            parent_x = clip_vec(x)
            parent_f = safe_eval(parent_x)

            # reset sigmas
            sigma = [max(1e-12, 0.3 * s) for s in spans]
            no_improve = 0

            if parent_f < best:
                best = parent_f
                best_x = list(parent_x)

        # Generate offspring and pick the best among them (plus parent)
        best_off_x = parent_x
        best_off_f = parent_f
        best_off_sigma = sigma

        # Global noise for correlated sigma update
        global_n = random.gauss(0.0, 1.0)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # Mutate sigmas (log-normal)
            child_sigma = [0.0] * dim
            for i in range(dim):
                coord_n = random.gauss(0.0, 1.0)
                s = sigma[i] * math.exp(tau0 * global_n + tau * coord_n)
                # clamp sigma to reasonable range
                if s < min_sigma:
                    s = min_sigma
                elif s > max_sigma:
                    s = max_sigma
                child_sigma[i] = s

            # Mutate solution
            child_x = [0.0] * dim
            for i in range(dim):
                child_x[i] = parent_x[i] + random.gauss(0.0, child_sigma[i])
            clip_vec(child_x)

            child_f = safe_eval(child_x)

            if child_f < best_off_f:
                best_off_f = child_f
                best_off_x = child_x
                best_off_sigma = child_sigma

        # Selection (1+lambda): accept if offspring beats parent
        if best_off_f < parent_f:
            parent_x = list(best_off_x)
            parent_f = best_off_f
            sigma = list(best_off_sigma)
            no_improve = 0
        else:
            no_improve += 1
            # mild contraction to encourage local refinement
            for i in range(dim):
                sigma[i] = max(min_sigma, sigma[i] * 0.98)

        # Track global best
        if parent_f < best:
            best = parent_f
            best_x = list(parent_x)

    return best
