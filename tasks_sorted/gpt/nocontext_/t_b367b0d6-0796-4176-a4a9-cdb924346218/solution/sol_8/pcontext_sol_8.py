import random
import time
import math


def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained (no numpy) hybrid algorithm:
    - Latin-hypercube-like stratified initialization (covers space well)
    - Evolution Strategy style mutations with adaptive step-size (success rule)
    - Occasional global re-sampling to escape local minima
    Returns: best (float) fitness found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- helpers ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clamp(x, i):
        if x < lows[i]:
            return lows[i]
        if x > highs[i]:
            return highs[i]
        return x

    def rand_uniform_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def evaluate(x):
        # func expects "array-like"; Python list is acceptable in most evaluators
        return float(func(x))

    # Latin-hypercube-like: for each dimension, pick a random point from each bin
    def lhs_batch(n):
        # returns list of n points
        # create per-dimension permutations of bins
        perms = []
        for i in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)

        pts = []
        for k in range(n):
            x = []
            for i in range(dim):
                # sample uniformly inside chosen bin
                u = (perms[i][k] + random.random()) / n
                x.append(lows[i] + u * spans[i])
            pts.append(x)
        return pts

    # ---- initial sampling ----
    best = float("inf")
    best_x = None

    # Batch size scales mildly with dimension, but stays small for speed
    init_n = max(8, min(40, 10 + 2 * dim))

    # Ensure we do at least something even for extremely small max_time
    # Try a few random evaluations quickly
    while time.time() < deadline and best_x is None:
        x = rand_uniform_vec()
        f = evaluate(x)
        best_x, best = x, f

    # If time already expired, return what we have
    if time.time() >= deadline:
        return best

    # Do one LHS-like batch if time allows
    for x in lhs_batch(init_n):
        if time.time() >= deadline:
            return best
        f = evaluate(x)
        if f < best:
            best, best_x = f, x

    # ---- evolution strategy loop ----
    # Initial mutation scale: fraction of search space
    sigma = [0.2 * s if s > 0 else 1.0 for s in spans]
    # Adaptation parameters (1/5th success rule style)
    success_count = 0
    trial_count = 0
    adapt_every = 20  # adjust sigma after this many trials

    # A small elite set to restart locally from good points
    elites = [(best, best_x)]
    elite_max = 6

    def add_elite(f, x):
        nonlocal elites
        elites.append((f, x))
        elites.sort(key=lambda t: t[0])
        # keep unique-ish by fitness proximity
        elites = elites[:elite_max]

    add_elite(best, best_x)

    # probability of global restart grows when stuck
    no_improve = 0

    while time.time() < deadline:
        # Choose a parent: mostly best, sometimes another elite
        if len(elites) > 1 and random.random() < 0.35:
            parent = elites[random.randrange(len(elites))][1]
        else:
            parent = best_x

        # Occasionally do a pure global sample (helps escape local minima)
        # Increase frequency when no improvements occur for a while
        p_global = 0.02 + min(0.25, no_improve / 500.0)
        if random.random() < p_global:
            x = rand_uniform_vec()
        else:
            # Mutate parent with gaussian noise per dimension
            x = []
            for i in range(dim):
                # Box-Muller for standard normal
                u1 = max(1e-12, random.random())
                u2 = random.random()
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                xi = parent[i] + z * sigma[i]
                x.append(clamp(xi, i))

        f = evaluate(x)

        trial_count += 1
        improved = f < best
        if improved:
            best = f
            best_x = x
            add_elite(f, x)
            success_count += 1
            no_improve = 0
        else:
            no_improve += 1

        # Step-size adaptation (per coordinate) based on success rate
        if trial_count % adapt_every == 0:
            rate = success_count / float(adapt_every)
            # if too successful, increase step; if not, decrease
            if rate > 0.22:
                factor = 1.2
            elif rate < 0.18:
                factor = 0.82
            else:
                factor = 1.0

            for i in range(dim):
                sigma[i] *= factor
                # keep sigma within sensible limits
                lo = 1e-12 * (spans[i] if spans[i] > 0 else 1.0)
                hi = 0.5 * (spans[i] if spans[i] > 0 else 1.0)
                if sigma[i] < lo:
                    sigma[i] = lo
                elif sigma[i] > hi:
                    sigma[i] = hi

            success_count = 0

    return best
