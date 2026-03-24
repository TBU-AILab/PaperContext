import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using:
      - Sobol-ish/LCG initial sampling (no external libs)
      - Adaptive (1+lambda) Evolution Strategy
      - Restarts with shrinking/expanding step sizes
      - Occasional coordinate/local refinements

    Returns: best (float) = best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ------------------------- helpers -------------------------
    def clip_to_bounds(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i]
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            y[i] = v
        return y

    def rand_uniform_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        lo, hi = bounds[i]
        return hi - lo

    # A tiny deterministic "quasi-random-ish" generator (LCG) to diversify starts
    lcg_state = 123456789
    def lcg():
        nonlocal lcg_state
        lcg_state = (1103515245 * lcg_state + 12345) & 0x7fffffff
        return lcg_state / 2147483647.0

    def lcg_vec():
        x = []
        for i in range(dim):
            u = lcg()
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # Robust evaluation wrapper
    def evaluate(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # ------------------------- initialization -------------------------
    best = float("inf")
    best_x = None

    # Initial probing budget: small but helpful
    init_tries = max(10, 5 * dim)
    for k in range(init_tries):
        if time.time() >= deadline:
            return best
        x = lcg_vec() if (k % 2 == 0) else rand_uniform_vec()
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        best_x = rand_uniform_vec()

    # ------------------------- main ES loop with restarts -------------------------
    # Global base sigma relative to domain size
    avg_span = sum(span(i) for i in range(dim)) / max(1, dim)
    base_sigma = 0.2 * avg_span if avg_span > 0 else 1.0

    # Restart parameters
    sigma = base_sigma
    no_improve = 0
    restart_after = 40 + 10 * dim  # evaluation-based stagnation threshold

    # (1+lambda)-ES settings
    lam = max(8, 4 * dim)          # offspring per generation
    coord_refine_prob = 0.15       # occasional local coordinate pokes

    while time.time() < deadline:
        # Parent (current center) is best_x, but allow mild jitter sometimes
        parent = best_x

        # Generate offspring and pick best
        gen_best_fx = float("inf")
        gen_best_x = None

        # Evaluate offspring
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            # Gaussian mutation (Box-Muller)
            child = parent[:]  # copy
            for i in range(dim):
                # N(0,1) via Box-Muller
                u1 = random.random()
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(max(1e-12, u1))) * math.cos(2.0 * math.pi * u2)
                child[i] += z * sigma

            # Occasionally do coordinate-only tweak (helps in narrow valleys)
            if random.random() < coord_refine_prob:
                j = random.randrange(dim)
                # small directional poke
                step = (random.random() * 2.0 - 1.0) * 0.25 * sigma
                child[j] += step

            child = clip_to_bounds(child)
            fchild = evaluate(child)

            if fchild < gen_best_fx:
                gen_best_fx = fchild
                gen_best_x = child

        # Selection + step-size adaptation (1/5th-ish success rule)
        if gen_best_fx < best:
            best = gen_best_fx
            best_x = gen_best_x
            no_improve = 0
            sigma *= 1.2  # expand on success
        else:
            no_improve += lam
            sigma *= 0.82  # contract on failure

        # Keep sigma within sensible limits
        sigma = max(1e-12, min(sigma, base_sigma * 2.0))

        # Restart if stagnating: jump to a new region, reset sigma
        if no_improve >= restart_after:
            # Mix: random + around-best perturbation
            if random.random() < 0.5:
                best_x = rand_uniform_vec()
            else:
                # jump around best with larger radius
                jump = []
                for i in range(dim):
                    lo, hi = bounds[i]
                    rad = 0.35 * (hi - lo)
                    jump.append(best_x[i] + (random.random() * 2.0 - 1.0) * rad)
                best_x = clip_to_bounds(jump)

            # Evaluate restart point and reset sigma
            f0 = evaluate(best_x)
            if f0 < best:
                best = f0
            sigma = base_sigma
            no_improve = 0

    return best
