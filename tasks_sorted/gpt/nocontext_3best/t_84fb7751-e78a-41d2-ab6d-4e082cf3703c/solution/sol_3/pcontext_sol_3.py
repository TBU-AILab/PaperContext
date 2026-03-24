import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs the provided ones:
      - Better global coverage: scrambled Halton seeding + opposition points.
      - Stronger local search: small-population adaptive ES (mu+lambda) with
        per-dimension sigmas and a cheap "evolution path" direction.
      - Robust step adaptation: success-based + stagnation-triggered radius changes.
      - Escape strategy: heavy-tailed (Cauchy) kicks and periodic restarts.
      - Fast caching: time-adaptive quantization to avoid duplicate evals.

    Returns:
      best (float): best objective value found within max_time.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0:
        return float("inf")
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if not (spans[i] > 0.0):
            spans[i] = 1.0

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

    # time-adaptive cache quantization: coarser early, finer late
    cache = {}
    def quant_key(x, q):
        k = []
        for i in range(dim):
            u = (x[i] - lows[i]) / spans[i]
            if u < 0.0: u = 0.0
            if u > 1.0: u = 1.0
            k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0
        # q from 2^17 .. 2^27
        q = 1 << (17 + int(10 * frac))
        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # approx N(0,1) via CLT, cheap and adequate for search steps
    def randn():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0) * 0.7071067811865475

    def cauchy():
        u = random.random()
        if u <= 1e-12:
            u = 1e-12
        elif u >= 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------- low-discrepancy seeding (scrambled Halton) ----------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(x ** 0.5)
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(n, base, perm):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, d = divmod(n, base)
            d = perm[d]
            denom *= base
            v += d / denom
        return v

    primes = first_primes(dim)
    perms = []
    for b in primes:
        p = list(range(b))
        random.shuffle(p)
        perms.append(p)

    def halton_point(idx):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(idx, primes[i], perms[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---------- init ----------
    best = float("inf")
    best_x = None

    x = rand_vec()
    fx = eval_cached(x)
    best, best_x = fx, x[:]

    # seeding size: moderate, time-aware
    seed_n = max(24, 10 * dim)
    seed_n = min(seed_n, 400)
    if max_time < 0.05:
        seed_n = max(3, min(seed_n, 10))

    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best

        cand = halton_point(k) if k <= int(0.75 * seed_n) else rand_vec()
        f = eval_cached(cand)
        if f < best:
            best, best_x = f, cand[:]

        # opposition point (often helps on bounded problems)
        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - cand[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

    # ---------- main: small-pop (mu+lambda)-ES with diagonal sigmas ----------
    x = best_x[:]
    fx = best

    # population sizing (kept small for speed)
    lam = 8 if dim <= 8 else 12
    lam = min(lam, 18)
    mu = max(2, lam // 2)

    # weights (simple decreasing)
    weights = [mu - i for i in range(mu)]
    wsum = float(sum(weights))
    weights = [w / wsum for w in weights]

    sigma = [0.22 * spans[i] for i in range(dim)]
    sigma_min = [1e-15 * spans[i] + 1e-18 for i in range(dim)]
    sigma_max = [0.70 * spans[i] for i in range(dim)]

    # evolution-path-like direction (helps accelerate in valleys)
    path = [0.0] * dim
    path_decay = 0.85

    # adaptation controls
    win = 20
    succ = 0
    tri = 0
    target = 0.22
    shrink = 0.87

    no_improve = 0
    last_best_check = best
    restart_every = 35  # generations
    gen = 0

    while True:
        if time.time() >= deadline:
            return best

        gen += 1

        # time fraction
        frac = (time.time() - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # exploration schedule
        p_heavy = 0.10 * (1.0 - 0.7 * frac) + (0.02 if dim > 20 else 0.0)

        # generate lambda offspring
        pop = []
        for _ in range(lam):
            if time.time() >= deadline:
                return best

            y = x[:]

            r = random.random()
            if r < p_heavy:
                # heavy-tailed jump around best_x in a subset of dims
                center = best_x
                k = 1 if dim <= 5 else max(1, dim // 3)
                idxs = random.sample(range(dim), k) if k < dim else list(range(dim))
                for i in idxs:
                    y[i] = center[i] + cauchy() * (2.4 * sigma[i] + 1e-12)
            else:
                # diagonal gaussian step + a bit of path guidance
                # sparse in high dim for speed and better acceptance
                k = dim if dim <= 10 else max(3, dim // 4)
                idxs = random.sample(range(dim), k) if k < dim else list(range(dim))
                for i in idxs:
                    y[i] += 0.65 * path[i] + randn() * sigma[i]

            clip_inplace(y)
            fy = eval_cached(y)
            pop.append((fy, y))

        # select mu best
        pop.sort(key=lambda t: t[0])
        best_off_f, best_off_x = pop[0]

        # recombine (weighted mean of best mu)
        new_x = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            y = pop[j][1]
            for i in range(dim):
                new_x[i] += w * y[i]
        clip_inplace(new_x)
        new_fx = eval_cached(new_x)

        tri += 1
        improved = False

        # accept if improves current
        if new_fx < fx or best_off_f < fx:
            # choose better of recombination or best offspring
            if best_off_f <= new_fx:
                cand_x, cand_f = best_off_x, best_off_f
            else:
                cand_x, cand_f = new_x, new_fx

            # update path using accepted displacement
            for i in range(dim):
                step_i = cand_x[i] - x[i]
                path[i] = path_decay * path[i] + (1.0 - path_decay) * step_i

            x, fx = cand_x[:], cand_f
            succ += 1
            improved = True
            no_improve = 0

            if fx < best:
                best, best_x = fx, x[:]
        else:
            no_improve += 1

        # step-size adaptation (success rule)
        if tri >= win:
            rate = succ / float(tri)
            mult = (1.0 / shrink) if rate > target else shrink
            for i in range(dim):
                s = sigma[i] * mult
                if s < sigma_min[i]:
                    s = sigma_min[i]
                elif s > sigma_max[i]:
                    s = sigma_max[i]
                sigma[i] = s
            tri = 0
            succ = 0

        # stagnation: force exploration tweaks
        if no_improve > 6 * max(1, dim // 3):
            # inflate sigmas a bit and reset path
            for i in range(dim):
                sigma[i] = min(sigma_max[i], sigma[i] * 1.35 + 1e-18)
                path[i] *= 0.2
            no_improve = 0

        # periodic restart around best (radius tied to remaining time)
        if gen % restart_every == 0:
            if best >= last_best_check - 1e-15:
                rad = (0.45 * (1.0 - frac) + 0.04)
                x = best_x[:]
                for i in range(dim):
                    x[i] += (2.0 * random.random() - 1.0) * (rad * spans[i])
                clip_inplace(x)
                fx = eval_cached(x)
                path = [0.0] * dim
                for i in range(dim):
                    sigma[i] = max(sigma_min[i], min(sigma_max[i], (0.18 + 0.10 * (1.0 - frac)) * spans[i]))
            last_best_check = best

        # occasional cheap opposition probe of best
        if improved and random.random() < 0.03 and time.time() < deadline:
            opp = [lows[i] + highs[i] - best_x[i] for i in range(dim)]
            clip_inplace(opp)
            fo = eval_cached(opp)
            if fo < best:
                best, best_x = fo, opp[:]
                x, fx = best_x[:], best
                path = [0.0] * dim
