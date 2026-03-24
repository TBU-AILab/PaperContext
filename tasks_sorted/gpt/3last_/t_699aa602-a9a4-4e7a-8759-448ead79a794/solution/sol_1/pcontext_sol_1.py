import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved derivative-free minimizer under box constraints.

    Hybrid strategy:
      1) Time-budgeted multi-start exploration (uniform + corners)
      2) Local optimization via (1+1)-ES with 1/5 success rule step adaptation
      3) Occasional coordinate probing + small random "kick" to escape plateaus
      4) Keeps best-so-far and restarts when stagnating

    Returns:
        best (float): best fitness found
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    span_safe = [s if s > 0 else 1.0 for s in span]

    def clamp_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]

    def rand_point():
        return [lo[i] + random.random() * span_safe[i] for i in range(dim)]

    def corner_point(mask):
        # mask bits choose hi (1) or lo (0)
        x = [0.0] * dim
        for i in range(dim):
            x[i] = hi[i] if ((mask >> i) & 1) else lo[i]
        return x

    def evaluate(x):
        return float(func(x))

    # --- Global best ---
    best = float("inf")
    best_x = None

    # --- Seed with a few diverse points early (helps on many benchmarks) ---
    # center
    if time.time() < deadline:
        x0 = [0.5 * (lo[i] + hi[i]) for i in range(dim)]
        f0 = evaluate(x0)
        best, best_x = f0, x0[:]

    # a few randoms + a few corners (limited by time)
    seed_count = 8 + 2 * dim
    corner_tries = min(8, 1 << min(dim, 3))  # up to 8 corners for dim>=3
    # corners
    for m in range(corner_tries):
        if time.time() >= deadline:
            return best
        x = corner_point(m)
        f = evaluate(x)
        if f < best:
            best, best_x = f, x[:]
    # randoms
    for _ in range(seed_count):
        if time.time() >= deadline:
            return best
        x = rand_point()
        f = evaluate(x)
        if f < best:
            best, best_x = f, x[:]

    # --- Main loop: repeated local ES runs with restarts ---
    # ES parameters
    # initial sigma relative to search box
    sigma0 = 0.25 * (sum(span_safe) / float(dim))  # scalar step size
    sigma_min = 1e-12 * (sum(span_safe) / float(dim))
    sigma_max = 0.5 * (sum(span_safe) / float(dim))

    # stagnation controls
    while time.time() < deadline:
        # restart either from best (exploit) or random (explore)
        if best_x is not None and random.random() < 0.65:
            x = best_x[:]
        else:
            x = rand_point()

        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

        # randomize sigma per restart
        sigma = sigma0 * (0.3 + 1.7 * random.random())
        sigma = max(sigma_min, min(sigma, sigma_max))

        # 1/5 success rule tracking
        succ = 0
        trials = 0
        adapt_window = 25 + 5 * dim

        no_improve = 0
        local_best = fx

        # precompute for normal-like noise using Box-Muller
        def randn():
            # standard normal via Box-Muller
            u1 = random.random()
            u2 = random.random()
            u1 = max(u1, 1e-300)
            return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

        while time.time() < deadline:
            # propose ES step
            y = x[:]
            # isotropic perturbation
            for i in range(dim):
                y[i] += sigma * span_safe[i] * 0.15 * randn()
            clamp_inplace(y)
            fy = evaluate(y)

            trials += 1
            improved = fy < fx
            if improved:
                x, fx = y, fy
                succ += 1
                no_improve = 0
                if fx < best:
                    best, best_x = fx, x[:]
                if fx < local_best:
                    local_best = fx
            else:
                no_improve += 1

            # occasional cheap coordinate probe (helps on separable / ridge functions)
            if time.time() < deadline and (no_improve % (8 + dim) == 0):
                j = random.randrange(dim)
                z = x[:]
                # try two directions
                step = (0.5 + random.random()) * sigma * span_safe[j] * 0.2
                z[j] = min(hi[j], max(lo[j], z[j] + step))
                fz = evaluate(z)
                if fz < fx:
                    x, fx = z, fz
                    succ += 1
                    no_improve = 0
                    if fx < best:
                        best, best_x = fx, x[:]
                else:
                    z[j] = min(hi[j], max(lo[j], x[j] - step))
                    fz = evaluate(z)
                    if fz < fx:
                        x, fx = z, fz
                        succ += 1
                        no_improve = 0
                        if fx < best:
                            best, best_x = fx, x[:]

            # adapt sigma using 1/5th success rule every window
            if trials >= adapt_window:
                rate = succ / float(trials)
                # if success rate > 1/5 -> increase; else decrease
                if rate > 0.2:
                    sigma *= 1.35
                else:
                    sigma *= 0.82
                sigma = max(sigma_min, min(sigma, sigma_max))
                succ = 0
                trials = 0

            # escape / restart criteria
            if no_improve > (35 + 8 * dim):
                # "kick" with moderate random move near the end of a basin
                if random.random() < 0.35 and best_x is not None:
                    x = best_x[:]
                    for i in range(dim):
                        x[i] += (2.0 * random.random() - 1.0) * 0.1 * span_safe[i]
                    clamp_inplace(x)
                    fx = evaluate(x)
                    if fx < best:
                        best, best_x = fx, x[:]
                break

    return best
