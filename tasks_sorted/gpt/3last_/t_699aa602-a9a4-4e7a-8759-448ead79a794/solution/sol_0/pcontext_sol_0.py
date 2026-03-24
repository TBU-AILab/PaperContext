import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Derivative-free minimization under box constraints using a time-budgeted
    adaptive local search with random restarts (stochastic hill-climbing with
    coordinate/gaussian perturbations + step-size adaptation).

    Returns:
        best (float): fitness of the best found solution.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- helpers ----
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]

    def clamp(x):
        # Clamp to bounds
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    def evaluate(x):
        # func expects an array-like; list works for typical numpy code too
        return float(func(x))

    # ---- initialization ----
    best = float("inf")
    best_x = None

    # Set initial step sizes relative to range
    base_sigma = [0.2 * s if s > 0 else 1.0 for s in span]

    # A "restart" keeps exploring locally until stagnation; then restart elsewhere
    while time.time() < deadline:
        # Start either near current best or fresh random (mixed strategy)
        if best_x is not None and random.random() < 0.5:
            x = best_x[:]  # exploit
        else:
            x = rand_point()  # explore

        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

        # Local search state
        sigma = base_sigma[:]                 # per-dimension step scale
        no_improve = 0
        iters = 0

        # Budget local phase to avoid getting stuck too long
        # (adapts to remaining time)
        while time.time() < deadline:
            iters += 1

            # Choose a move type: coordinate tweak or multi-dim gaussian-like move
            if dim == 1 or random.random() < 0.7:
                # Coordinate move
                j = random.randrange(dim)
                x2 = x[:]
                # symmetric perturbation
                step = (2.0 * random.random() - 1.0) * sigma[j]
                x2[j] += step
            else:
                # Multi-dimensional move
                x2 = x[:]
                for j in range(dim):
                    # approximate gaussian by sum of uniforms (CLT), mean 0
                    g = (random.random() + random.random() + random.random() +
                         random.random() + random.random() + random.random()) - 3.0
                    x2[j] += g * (0.3 * sigma[j])

            clamp(x2)
            f2 = evaluate(x2)

            # Accept improvement; sometimes accept equal (helps plateaus)
            if f2 < fx or (f2 == fx and random.random() < 0.2):
                x, fx = x2, f2
                no_improve = 0

                # Successful step -> slightly increase step sizes (mild)
                for j in range(dim):
                    sigma[j] *= 1.05

                if fx < best:
                    best, best_x = fx, x[:]
            else:
                no_improve += 1

                # Unsuccessful -> shrink step sizes a bit
                for j in range(dim):
                    sigma[j] *= 0.98

            # If step sizes get too small, re-inflate a little (avoid collapse)
            for j in range(dim):
                min_sigma = 1e-12 * (span[j] if span[j] > 0 else 1.0)
                if sigma[j] < min_sigma:
                    sigma[j] = min_sigma

            # Stagnation / plateau detection: restart
            # threshold scales with dimension (more dims -> allow more attempts)
            if no_improve > 40 + 10 * dim:
                break

            # Extra guard: don't spend too long in one basin
            if iters > 2000 + 200 * dim:
                break

        # Slightly vary base_sigma for next restart (diversification)
        for j in range(dim):
            # Randomize within [0.05, 0.5] of span
            s = span[j] if span[j] > 0 else 1.0
            base_sigma[j] = (0.05 + 0.45 * random.random()) * s

    return best
