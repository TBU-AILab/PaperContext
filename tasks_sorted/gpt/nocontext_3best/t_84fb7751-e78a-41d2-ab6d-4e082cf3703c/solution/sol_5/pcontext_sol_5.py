import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements vs the provided algorithms:
      - Better global start: scrambled Halton + opposition + short coordinate probes.
      - Main optimizer: two-phase search
          (A) adaptive diagonal ES around incumbent (fast anytime improvement)
          (B) periodic trust-region local pattern search (good at squeezing last bits)
      - Smarter restarts: budget-aware, mixing global (Halton/random) and focused (best-centered).
      - Robust evaluation cache: time-adaptive quantization to cut duplicates without
        killing late-stage precision.
      - Cheap "surrogate direction": keeps a momentum-like path from successful steps.
    Returns:
      best (float): best objective value found within max_time seconds.
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

    # ---------------- helpers ----------------
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

    # time-adaptive cache quantization: coarse early, fine late
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
        # q from 2^15 .. 2^27
        q = 1 << (15 + int(12 * frac))
        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # approx N(0,1) via CLT
    def randn():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0) * 0.7071067811865475

    def cauchy():
        u = random.random()
        if u <= 1e-12: u = 1e-12
        if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------------- scrambled Halton ----------------
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

    # ---------------- init / seeding ----------------
    best = float("inf")
    best_x = rand_vec()
    best = eval_cached(best_x)

    # modest seeding; keep fast for tiny max_time
    seed_n = max(24, 10 * dim)
    seed_n = min(seed_n, 500)
    if max_time < 0.05:
        seed_n = max(3, min(seed_n, 12))

    # mixed Halton + random + opposition
    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best

        if k <= int(0.8 * seed_n):
            x = halton_point(k)
        else:
            x = rand_vec()

        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

        if time.time() >= deadline:
            return best

        opp = [lows[i] + highs[i] - x[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

    # quick coordinate probe around best (cheap and often beneficial)
    # try +/- small steps in a few dimensions
    probe_dims = list(range(dim))
    random.shuffle(probe_dims)
    probe_dims = probe_dims[:min(dim, 8)]
    for i in probe_dims:
        if time.time() >= deadline:
            return best
        for sgn in (-1.0, 1.0):
            x = best_x[:]
            x[i] += sgn * 0.10 * spans[i]
            clip_inplace(x)
            fx = eval_cached(x)
            if fx < best:
                best, best_x = fx, x[:]

    # ---------------- main search: ES + occasional local pattern steps ----------------
    x = best_x[:]
    fx = best

    # diagonal step sizes
    sigma = [0.18 * spans[i] for i in range(dim)]
    sigma_min = [1e-15 * spans[i] + 1e-18 for i in range(dim)]
    sigma_max = [0.70 * spans[i] for i in range(dim)]

    # momentum-like path
    path = [0.0] * dim
    path_decay = 0.85

    # (mu,lambda)-ES settings
    lam = 10 if dim <= 10 else (14 if dim <= 30 else 18)
    lam = min(max(6, lam), 24)
    mu = max(2, lam // 2)

    weights = [mu - i for i in range(mu)]
    wsum = float(sum(weights))
    weights = [w / wsum for w in weights]

    # success-based adaptation
    win = 18
    succ = 0
    tri = 0
    target = 0.22
    shrink = 0.88

    # pattern-search trust region (radius in [0,1] * span)
    tr = 0.12
    tr_min = 1e-14
    tr_max = 0.50
    pattern_every = 9  # generations

    no_improve = 0
    gen = 0
    last_best = best
    restart_every = 40

    while True:
        if time.time() >= deadline:
            return best

        gen += 1
        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # heavy tail probability decreases with time; extra in high dim
        p_heavy = 0.12 * (1.0 - 0.75 * frac) + (0.03 if dim > 25 else 0.0)

        # --- ES offspring ---
        pop = []
        # sparse perturbations in high dim
        k_sparse = dim if dim <= 10 else max(3, dim // 4)

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            y = x[:]
            if random.random() < p_heavy:
                # Cauchy around global best in subset of dims
                center = best_x
                kk = 1 if dim <= 5 else max(1, dim // 3)
                idxs = random.sample(range(dim), kk) if kk < dim else list(range(dim))
                for i in idxs:
                    y[i] = center[i] + cauchy() * (2.6 * sigma[i] + 1e-12)
            else:
                idxs = random.sample(range(dim), k_sparse) if k_sparse < dim else list(range(dim))
                for i in idxs:
                    y[i] += 0.65 * path[i] + randn() * sigma[i]

            clip_inplace(y)
            fy = eval_cached(y)
            pop.append((fy, y))

        pop.sort(key=lambda t: t[0])
        best_off_f, best_off_x = pop[0]

        # weighted recombination of top mu
        new_x = [0.0] * dim
        for j in range(mu):
            w = weights[j]
            y = pop[j][1]
            for i in range(dim):
                new_x[i] += w * y[i]
        clip_inplace(new_x)
        new_fx = eval_cached(new_x)

        tri += 1
        accepted = False

        # accept best among (recombined, best offspring) if it improves current
        cand_x, cand_f = (best_off_x, best_off_f) if best_off_f <= new_fx else (new_x, new_fx)
        if cand_f < fx:
            # update momentum path
            for i in range(dim):
                step_i = cand_x[i] - x[i]
                path[i] = path_decay * path[i] + (1.0 - path_decay) * step_i
            x, fx = cand_x[:], cand_f
            succ += 1
            accepted = True
            no_improve = 0
            if fx < best:
                best, best_x = fx, x[:]
        else:
            no_improve += 1

        # success rule on sigma
        if tri >= win:
            rate = succ / float(tri)
            mult = (1.0 / shrink) if rate > target else shrink
            for i in range(dim):
                s = sigma[i] * mult
                if s < sigma_min[i]: s = sigma_min[i]
                if s > sigma_max[i]: s = sigma_max[i]
                sigma[i] = s
            tri = 0
            succ = 0

        # --- occasional pattern search around current/best to refine ---
        if gen % pattern_every == 0 and time.time() < deadline:
            # choose center as best_x (more stable) late, else current
            center = best_x if frac > 0.45 else x
            base_f = best if center is best_x else fx

            # try a few coordinate moves (not all dims, for speed)
            dims = list(range(dim))
            random.shuffle(dims)
            nd = min(dim, 10 if dim <= 40 else 14)
            dims = dims[:nd]

            improved = False
            # step size shrinks with time
            step_scale = tr * (0.85 * (1.0 - frac) + 0.15)

            for i in dims:
                if time.time() >= deadline:
                    return best
                step = step_scale * spans[i]
                if step <= 0.0:
                    continue
                for sgn in (-1.0, 1.0):
                    y = center[:]
                    y[i] += sgn * step
                    clip_inplace(y)
                    fy = eval_cached(y)
                    if fy < best:
                        best, best_x = fy, y[:]
                        x, fx = best_x[:], best
                        improved = True
                        # also update path toward improvement
                        for k in range(dim):
                            path[k] = 0.7 * path[k] + 0.3 * (best_x[k] - center[k])
                        break

            if improved:
                tr = min(tr_max, tr * 1.12)
            else:
                tr = max(tr_min, tr * 0.82)

        # stagnation handling
        if no_improve > 6 * max(1, dim // 3):
            for i in range(dim):
                sigma[i] = min(sigma_max[i], sigma[i] * 1.4 + 1e-18)
                path[i] *= 0.2
            tr = min(tr_max, tr * 1.2)
            no_improve = 0

        # periodic restart (focused)
        if gen % restart_every == 0:
            if best >= last_best - 1e-15:
                # restart radius depends on remaining time
                rad = (0.50 * (1.0 - frac) + 0.05)
                x = best_x[:]
                for i in range(dim):
                    x[i] += (2.0 * random.random() - 1.0) * (rad * spans[i])
                clip_inplace(x)
                fx = eval_cached(x)
                path = [0.0] * dim
                for i in range(dim):
                    sigma[i] = max(sigma_min[i],
                                   min(sigma_max[i], (0.20 + 0.10 * (1.0 - frac)) * spans[i]))
                tr = max(tr_min, min(tr_max, 0.14 * (1.0 - frac) + 0.03))
            last_best = best

        # cheap opposition probe of best occasionally
        if accepted and random.random() < 0.02 and time.time() < deadline:
            opp = [lows[i] + highs[i] - best_x[i] for i in range(dim)]
            clip_inplace(opp)
            fo = eval_cached(opp)
            if fo < best:
                best, best_x = fo, opp[:]
                x, fx = best_x[:], best
                path = [0.0] * dim
