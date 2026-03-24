import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Main ideas (kept lightweight, no external libs):
      1) Quasi-Latin-hypercube seeding to get a strong starting incumbent.
      2) (1+1)-ES style local search with *adaptive per-dimension step sizes*
         using the 1/5 success rule (robust, very fast).
      3) Occasional heavy-tailed (Cauchy-like) jumps to escape local minima.
      4) Periodic random restarts with shrinking radius around the best.
      5) Cache evaluations (rounded) to avoid wasting time on duplicates.

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    spans = [s if s > 0 else 1.0 for s in spans]

    # ---------- helpers ----------
    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def safe_eval(x):
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

    # Evaluation cache to avoid duplicates (common with clipping / tiny steps)
    # Key is rounded to a grid relative to span.
    cache = {}
    # Round resolution: ~1e-12 of span (not too aggressive, avoids huge keys)
    def key_of(x):
        k = []
        for i in range(dim):
            s = spans[i]
            # scale to unit then quantize
            u = (x[i] - lows[i]) / s
            k.append(int(u * 1e12 + 0.5))
        return tuple(k)

    def eval_cached(x):
        k = key_of(x)
        if k in cache:
            return cache[k]
        v = safe_eval(x)
        cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Heavy-tailed step: Cauchy via tan(pi*(u-0.5))
    def cauchy():
        u = random.random()
        # avoid exact 0/1
        if u <= 1e-12:
            u = 1e-12
        elif u >= 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------- quick exit ----------
    if max_time <= 0:
        return float("inf")

    # ---------- seeding (quasi Latin hypercube) ----------
    # Use independent permutations of strata per dimension.
    # Keep it modest to not burn budget on high dim / small time.
    seed_points = max(12, 6 * dim)
    # Reduce if time is extremely small
    if max_time < 0.05:
        seed_points = max(2, min(seed_points, 8))

    strata = seed_points
    perms = []
    for i in range(dim):
        p = list(range(strata))
        random.shuffle(p)
        perms.append(p)

    best = float("inf")
    best_x = None

    # ensure at least one eval
    x0 = rand_vec()
    best = eval_cached(x0)
    best_x = x0[:]

    for k in range(strata):
        if time.time() >= deadline:
            return best
        x = [0.0] * dim
        for i in range(dim):
            a = perms[i][k] / float(strata)
            b = (perms[i][k] + 1) / float(strata)
            u = a + (b - a) * random.random()
            x[i] = lows[i] + u * spans[i]
        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

    # ---------- main optimizer: adaptive (1+1)-ES + occasional heavy-tail jumps ----------
    x = best_x[:]
    fx = best

    # per-dimension step sizes
    # start at ~20% of span, but cap for stability
    sigma = [0.20 * spans[i] for i in range(dim)]
    sigma_min = [1e-14 * spans[i] + 1e-18 for i in range(dim)]
    sigma_max = [0.50 * spans[i] for i in range(dim)]

    # 1/5 success rule parameters
    window = 20                 # adaptation window
    target = 0.20               # target success rate
    adapt_rate = 0.9            # multiplicative factor base
    successes = 0
    trials = 0

    # restart / exploration controls
    heavy_jump_prob = 0.06      # do a heavy-tailed jump sometimes
    restart_period = 150        # attempts between restart checks
    last_restart_check = 0
    stagnant = 0
    best_at_last_check = best

    # For generating correlated "direction" without libs: simple normalized random sign vector
    def propose_local(x, sigma):
        y = x[:]
        # sparse perturbations help in high dimension
        # choose k dimensions to perturb (at least 1)
        k = 1 if dim <= 3 else max(1, dim // 4)
        # random subset
        idxs = random.sample(range(dim), k) if k < dim else list(range(dim))
        for i in idxs:
            # Gaussian-ish using sum of uniforms (CLT), fast
            g = (random.random() + random.random() + random.random() + random.random() - 2.0)  # ~N(0, ~0.33)
            y[i] += g * sigma[i]
        clip_inplace(y)
        return y

    def propose_heavy(x, sigma, focus_center):
        # jump around focus_center (usually best_x), heavy-tailed in a few dims
        y = focus_center[:]  # jump from best, not current, to diversify
        k = 1 if dim <= 5 else max(1, dim // 3)
        idxs = random.sample(range(dim), k) if k < dim else list(range(dim))
        for i in idxs:
            y[i] += cauchy() * (2.5 * sigma[i] + 1e-12)
        clip_inplace(y)
        return y

    attempts = 0
    while True:
        if time.time() >= deadline:
            return best

        attempts += 1

        # choose move type
        if random.random() < heavy_jump_prob:
            y = propose_heavy(x, sigma, best_x)
        else:
            y = propose_local(x, sigma)

        fy = eval_cached(y)
        trials += 1

        if fy < fx:
            x, fx = y, fy
            successes += 1
            stagnant = 0
            if fx < best:
                best, best_x = fx, x[:]
        else:
            stagnant += 1

        # adapt step sizes every 'window' trials (1/5 success rule)
        if trials >= window:
            rate = successes / float(trials)
            # if too successful -> enlarge; else shrink
            if rate > target:
                mult = 1.0 / adapt_rate
            else:
                mult = adapt_rate
            for i in range(dim):
                s = sigma[i] * mult
                if s < sigma_min[i]:
                    s = sigma_min[i]
                elif s > sigma_max[i]:
                    s = sigma_max[i]
                sigma[i] = s
            trials = 0
            successes = 0

        # periodic restart logic
        if attempts - last_restart_check >= restart_period:
            last_restart_check = attempts

            # if no real improvement recently, restart near best with a larger radius
            if best >= best_at_last_check - 1e-15:
                # radius based on spans, but shrink as time elapses
                elapsed = time.time() - t0
                frac = min(1.0, max(0.0, elapsed / max_time))
                # start broad, then narrow
                rad_scale = (0.35 * (1.0 - frac) + 0.05)

                x = best_x[:]
                for i in range(dim):
                    # uniform jitter
                    x[i] += (random.random() * 2.0 - 1.0) * (rad_scale * spans[i])
                clip_inplace(x)
                fx = eval_cached(x)

                # reset sigma moderately after restart
                for i in range(dim):
                    sigma[i] = max(sigma_min[i], min(sigma_max[i], 0.15 * spans[i]))
            best_at_last_check = best

            # also, if we're extremely stagnant, force a heavy jump probability spike
            if stagnant > 5 * dim:
                heavy_jump_prob = min(0.25, heavy_jump_prob * 1.4)
                stagnant = 0
            else:
                # relax back toward default
                heavy_jump_prob = max(0.04, heavy_jump_prob * 0.98)
