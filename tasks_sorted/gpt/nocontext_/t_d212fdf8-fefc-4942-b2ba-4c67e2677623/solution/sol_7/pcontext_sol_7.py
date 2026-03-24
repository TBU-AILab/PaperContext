import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained hybrid:
    - Latin-hypercube-like stratified initialization
    - Local Gaussian mutation hill-climbing
    - Simulated annealing-style occasional acceptance
    - Multi-restarts with shrinking step size on stagnation

    Returns:
        best (float): best (minimum) objective value found within max_time seconds.
    """
    t0 = time.time()

    # Defensive checks
    if dim <= 0:
        return float("inf")
    if bounds is None or len(bounds) != dim:
        return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if not (s > 0.0):
            return float("inf")

    def clip_vec(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def lhs_batch(n):
        # Simple stratified sampling per dimension; then shuffled.
        # Not a full LHS with correlation control, but improves coverage vs pure random.
        pts = [[0.0] * dim for _ in range(n)]
        for j in range(dim):
            perm = list(range(n))
            random.shuffle(perm)
            for i in range(n):
                # sample inside stratum
                u = (perm[i] + random.random()) / n
                pts[i][j] = lows[j] + u * spans[j]
        return pts

    def evaluate(x):
        # func expects an array-like; provide list (self-contained; no numpy).
        v = func(x)
        # Robustness: ensure float
        try:
            return float(v)
        except Exception:
            return float("inf")

    # --- Initialization ---
    best_val = float("inf")
    best_x = None

    # Small initial design; scale with dimension but keep cheap
    init_n = max(8, min(60, 10 * dim))
    for x in lhs_batch(init_n):
        if time.time() - t0 >= max_time:
            return best_val
        v = evaluate(x)
        if v < best_val:
            best_val, best_x = v, x[:]

    if best_x is None:
        # fallback
        best_x = rand_vec()
        best_val = evaluate(best_x)

    # --- Main loop parameters ---
    # Mutation scale starts moderately large then adapts
    step = [0.2 * spans[i] for i in range(dim)]
    min_step = [1e-12 * max(1.0, spans[i]) for i in range(dim)]

    # Annealing temperature (relative); used for occasional uphill moves
    T0 = 1.0
    # Stagnation control
    stagnation = 0
    stagnation_limit = 200 + 20 * dim

    # Current state for local search
    x = best_x[:]
    fx = best_val

    # Helper: mutate with per-dimension Gaussian noise
    def mutate(x0, step_scale=1.0):
        y = x0[:]
        for i in range(dim):
            sigma = step[i] * step_scale
            if sigma <= 0.0:
                continue
            y[i] += random.gauss(0.0, sigma)
        return clip_vec(y)

    # Main time-bounded optimization
    it = 0
    while True:
        if time.time() - t0 >= max_time:
            return best_val

        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max_time if max_time > 0 else 1.0

        # Temperature cools down with time
        T = max(1e-12, T0 * (1.0 - frac) ** 2)

        # Occasionally restart around global best or random point
        # More restarts early; fewer later
        if stagnation > stagnation_limit:
            stagnation = 0
            # shrink step sizes a bit upon stagnation
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.5)

            if random.random() < 0.7 and best_x is not None:
                x = best_x[:]
                fx = best_val
            else:
                x = rand_vec()
                fx = evaluate(x)

        # Adaptive mutation: smaller later in time; also occasional big jump
        if random.random() < 0.1:
            scale = 3.0  # exploration jump
        else:
            scale = max(0.05, 1.0 - frac)  # exploitation later

        y = mutate(x, step_scale=scale)
        fy = evaluate(y)

        # Accept if better or with SA probability
        accepted = False
        if fy <= fx:
            accepted = True
        else:
            # If fx is huge or equal, avoid division weirdness
            delta = fy - fx
            # Standard metropolis acceptance
            if random.random() < math.exp(-delta / max(1e-12, T)):
                accepted = True

        if accepted:
            x, fx = y, fy

        # Track global best
        if fy < best_val:
            best_val = fy
            best_x = y[:]
            stagnation = 0
            # Slightly expand step on improvements (helps move along ridges)
            for i in range(dim):
                step[i] = min(spans[i], step[i] * 1.05)
        else:
            stagnation += 1
            # Very mild decay of step sizes over time
            if (it % (50 + 5 * dim)) == 0:
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.95)
