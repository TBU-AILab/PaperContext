import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded global optimization using:
      - Latin-hypercube-like stratified initialization
      - Adaptive simulated annealing with occasional restarts
      - Coordinate/local steps mixed with global steps
    Returns:
      best (float): best (minimum) function value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    def clip(x, lo, hi):
        if x < lo: 
            return lo
        if x > hi: 
            return hi
        return x

    def rand_in_bounds():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # --- Stratified initial sampling (approx. LHS without external libs) ---
    # Choose number of strata per dimension based on time and dimension.
    # Keep it small but useful; ensures coverage early.
    # Total initial points ~= m (not m^dim) by drawing one stratum per dim per point.
    m = max(8, min(40, int(10 + 5 * math.log2(dim + 1))))
    strata = [list(range(m)) for _ in range(dim)]
    for d in range(dim):
        random.shuffle(strata[d])

    best_x = None
    best = float("inf")

    # Evaluate an initial batch quickly (stop if time exceeded)
    for k in range(m):
        if time.time() >= deadline:
            return best
        x = []
        for d in range(dim):
            lo, hi = bounds[d]
            # pick kth stratum for this dimension, jitter within stratum
            s = strata[d][k]
            u0 = s / m
            u1 = (s + 1) / m
            u = random.uniform(u0, u1)
            x.append(lo + u * (hi - lo))
        fx = func(x)
        if fx < best:
            best = fx
            best_x = x

    # --- Adaptive Simulated Annealing with mixed moves and restarts ---
    # Step sizes proportional to ranges; adapt with success rate.
    ranges = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    # initial step scale: moderate
    step_scale = [0.15 * r if r > 0 else 1.0 for r in ranges]

    # Temperature derived from observed variability; fallback to 1.0
    # We'll keep it simple and adaptive.
    T = 1.0
    # Restart schedule
    next_restart = time.time() + max_time / 6.0

    x = best_x[:] if best_x is not None else rand_in_bounds()
    fx = func(x)

    # Adaptation counters
    acc = 0
    tries = 0
    adapt_every = 40  # adjust step sizes periodically

    # Cooling parameters
    alpha = 0.985  # multiplicative cooling
    Tmin = 1e-12

    while True:
        now = time.time()
        if now >= deadline:
            return best

        # occasional restart (global jump) to escape local minima
        if now >= next_restart:
            next_restart = now + max_time / 6.0
            # restart near best with some probability, otherwise global
            if best_x is not None and random.random() < 0.7:
                # gaussian around best
                x = []
                for d in range(dim):
                    lo, hi = bounds[d]
                    sd = 0.25 * ranges[d] if ranges[d] > 0 else 1.0
                    val = random.gauss(best_x[d], sd)
                    x.append(clip(val, lo, hi))
            else:
                x = rand_in_bounds()
            fx = func(x)
            if fx < best:
                best = fx
                best_x = x[:]
            # bump temperature a bit after restart
            T = max(T, 1.0)
            continue

        # Choose move type: local coordinate / local full / global
        r = random.random()
        if r < 0.55:
            # coordinate move (good for high dim)
            y = x[:]
            d = random.randrange(dim)
            lo, hi = bounds[d]
            # heavy-tailed step via Cauchy-like (tan) for occasional big jumps
            u = random.random()
            cauchy = math.tan(math.pi * (u - 0.5))
            step = 0.35 * step_scale[d] * cauchy
            y[d] = clip(y[d] + step, lo, hi)
        elif r < 0.90:
            # full local move
            y = []
            for d in range(dim):
                lo, hi = bounds[d]
                # gaussian step
                step = random.gauss(0.0, step_scale[d])
                y.append(clip(x[d] + step, lo, hi))
        else:
            # global move
            y = rand_in_bounds()

        fy = func(y)
        tries += 1

        # Metropolis acceptance
        delta = fy - fx
        accept = False
        if delta <= 0:
            accept = True
        else:
            # avoid underflow/overflow issues
            if T > Tmin:
                p = math.exp(-delta / T)
                if random.random() < p:
                    accept = True

        if accept:
            x, fx = y, fy
            acc += 1
            if fy < best:
                best = fy
                best_x = y[:]

        # Cooling (slow, time-bounded)
        T = max(Tmin, T * alpha)

        # Adapt step sizes to target acceptance ~ 0.25-0.35
        if tries >= adapt_every:
            rate = acc / float(tries)
            # If accepting too often, increase step; if too rarely, decrease.
            if rate > 0.40:
                mult = 1.15
            elif rate < 0.20:
                mult = 0.70
            else:
                mult = 1.0

            if mult != 1.0:
                for d in range(dim):
                    # keep within reasonable limits
                    step_scale[d] = max(1e-15, min(0.5 * ranges[d] if ranges[d] > 0 else 1.0, step_scale[d] * mult))

            acc = 0
            tries = 0
