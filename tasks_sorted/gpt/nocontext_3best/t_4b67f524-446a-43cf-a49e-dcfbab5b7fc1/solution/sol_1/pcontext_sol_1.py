import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (no external libs):
      - Sobol-like (van der Corput) low-discrepancy exploration (fast, good coverage)
      - (1+1)-ES style local search with success-based step adaptation (1/5 rule)
      - Occasional "heavy-tail" jumps (Cauchy) to escape local minima
      - Multiple elites: keep a small set of best points; restart from them

    Returns:
      best: best (minimum) fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    fixed = [spans[i] == 0.0 for i in range(dim)]

    def clamp(x):
        y = list(x)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def evaluate(x):
        return float(func(x))

    # --- Low-discrepancy generator (van der Corput in different prime bases) ---
    def van_der_corput(n, base):
        vdc = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def primes_upto(k):
        # produce first k primes (small k: dim)
        primes = []
        candidate = 2
        while len(primes) < k:
            is_p = True
            r = int(candidate ** 0.5)
            for p in primes:
                if p > r:
                    break
                if candidate % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(candidate)
            candidate += 1
        return primes

    bases = primes_upto(max(1, dim))

    # Quasi-random point index -> point in bounds
    def qrand(idx):
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                u = van_der_corput(idx + 1, bases[i])  # avoid idx=0 being all zeros
                x[i] = lows[i] + u * spans[i]
        return x

    # --- heavy-tail step (Cauchy) ---
    def cauchy():
        # tan(pi*(u-0.5)) is standard Cauchy
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # --- initialization: mix quasi-random + random ---
    best = float("inf")
    best_x = None

    # Keep small elite set for restarts
    elite_k = max(3, min(10, 2 + dim // 3))
    elites = []  # list of (f, x)

    def push_elite(f, x):
        nonlocal elites
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_k:
            elites.pop()

    # Initial budget proportional to dim but capped
    init_n = max(40, min(400, 30 * dim + 40))
    # Ensure we don't overrun time if func is expensive
    for j in range(init_n):
        if time.time() >= deadline:
            return best
        if j % 5 == 0:
            x = [lows[i] + random.random() * spans[i] if not fixed[i] else lows[i] for i in range(dim)]
        else:
            x = qrand(j)
        f = evaluate(x)
        if f < best:
            best, best_x = f, x[:]
        push_elite(f, x)

    if best_x is None:
        best_x = qrand(0)
        best = evaluate(best_x)
        push_elite(best, best_x)

    # --- Local search (1+1 ES with success rule) ---
    # Start step as fraction of span; per-dim steps
    sigma0 = [0.20 * spans[i] if not fixed[i] else 0.0 for i in range(dim)]
    sigma = sigma0[:]

    # Track success rate over a window
    window = max(20, 10 + 2 * dim)
    successes = 0
    trials = 0

    x_cur = best_x[:]
    f_cur = best

    # For speed, precompute a minimal sigma floor
    span_max = max(spans) if dim > 0 else 0.0
    sigma_floor = 1e-12 + 1e-9 * span_max

    # Restart / jump control
    no_improve = 0
    restart_after = 80 + 15 * dim

    # Main loop
    idx = init_n
    while True:
        if time.time() >= deadline:
            return best

        trials += 1

        # Choose a parent: usually current, sometimes best elite to diversify
        if random.random() < 0.10 and elites:
            _, x_parent = random.choice(elites)
            x_base = x_parent[:]
            # Blend with current best a bit
            if best_x is not None and random.random() < 0.5:
                a = 0.5 + 0.5 * random.random()
                for i in range(dim):
                    x_base[i] = a * x_base[i] + (1.0 - a) * best_x[i]
        else:
            x_base = x_cur

        # Mutation type: mostly gaussian-ish, sometimes heavy-tail, sometimes global sample
        r = random.random()
        if r < 0.80:
            # approx Gaussian via sum of uniforms (fast)
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                g = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random() - 3.0) / 3.0
                x_new[i] += g * s
        elif r < 0.95:
            # Cauchy heavy-tail jump
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                s = max(sigma[i], sigma_floor)
                x_new[i] += cauchy() * 0.35 * s
        else:
            # global low-discrepancy sample
            x_new = qrand(idx)
            idx += 1

        x_new = clamp(x_new)
        f_new = evaluate(x_new)

        improved = False
        if f_new <= f_cur:
            x_cur, f_cur = x_new, f_new
            improved = True

        if f_new < best:
            best, best_x = f_new, x_new[:]
            push_elite(f_new, x_new)
            no_improve = 0
        else:
            no_improve += 1

        if improved:
            successes += 1
            push_elite(f_new, x_new)

        # Success-based step adaptation (1/5 rule)
        if trials >= window:
            rate = successes / float(trials)
            # If success rate too high -> increase step; too low -> decrease
            if rate > 0.22:
                factor = 1.35
            elif rate < 0.18:
                factor = 0.70
            else:
                factor = 1.0

            if factor != 1.0:
                for i in range(dim):
                    if fixed[i]:
                        continue
                    sigma[i] *= factor
                    # keep within reasonable bounds
                    if sigma[i] > spans[i] * 0.8:
                        sigma[i] = spans[i] * 0.8
                    if sigma[i] < sigma_floor:
                        sigma[i] = sigma_floor

            successes = 0
            trials = 0

        # If stagnating: restart near an elite with refreshed sigmas
        if no_improve >= restart_after:
            # pick best or a random elite, then jitter
            if elites and random.random() < 0.8:
                _, seed = elites[0] if random.random() < 0.6 else random.choice(elites)
                x_cur = seed[:]
            else:
                x_cur = qrand(idx)
                idx += 1

            # reset step sizes somewhat, smaller over time to refine
            frac = (time.time() - t0) / max_time if max_time > 0 else 1.0
            shrink = 0.5 + 0.5 * (1.0 - frac)  # larger early, smaller late
            sigma = [sigma0[i] * shrink for i in range(dim)]
            # jitter the restart point
            for i in range(dim):
                if fixed[i]:
                    continue
                x_cur[i] += (random.random() * 2.0 - 1.0) * 0.10 * spans[i]
            x_cur = clamp(x_cur)
            f_cur = evaluate(x_cur)
            if f_cur < best:
                best, best_x = f_cur, x_cur[:]
                push_elite(f_cur, x_cur)
            no_improve = 0
