import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer.

    Key upgrades vs typical random/ES/pattern search:
      - Stronger seeding: scrambled Halton (low-discrepancy) + a few pure randoms.
      - Memetic core: adaptive diagonal ES (per-dim sigma) + occasional multi-dim CMA-like
        rank-1 direction (very lightweight).
      - Two-scale moves: (a) local Gaussian steps, (b) heavy-tailed Levy/Cauchy escapes.
      - "Opposition" / reflection candidates around best to quickly probe basins.
      - Evaluation cache with adaptive quantization to avoid repeated evaluations.
      - Automatic restarts with radius schedule tied to remaining time.

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    if max_time is None or max_time <= 0:
        return float("inf")
    deadline = t0 + float(max_time)

    # ---- bounds / spans ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0:
            spans[i] = 1.0

    # ---- helpers ----
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

    # adaptive cache quantization: coarse early, finer later
    cache = {}

    def quant_key(x, q):
        # q: integer quantization scale (higher -> finer)
        k = []
        for i in range(dim):
            u = (x[i] - lows[i]) / spans[i]  # [0,1]
            # clamp u defensively
            if u < 0.0: u = 0.0
            if u > 1.0: u = 1.0
            k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        # time-based quantization: start coarse, end finer
        now = time.time()
        frac = (now - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0
        # q ranges from 2^18 to 2^28 (keeps cache effective yet not exploding)
        q = 1 << (18 + int(10 * frac))
        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---- low-discrepancy seeding: scrambled Halton ----
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

    def van_der_corput(n, base, scramble):
        # radical inverse with digit permutation scramble
        v = 0.0
        denom = 1.0
        while n > 0:
            n, digit = divmod(n, base)
            digit = scramble[digit]
            denom *= base
            v += digit / denom
        return v

    primes = first_primes(dim)
    # per-dim digit permutation for scrambling (fixed per run, cheap)
    scrambles = []
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        scrambles.append(perm)

    def halton_point(index):
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(index, primes[i], scrambles[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---- heavy tail ----
    def cauchy():
        u = random.random()
        if u <= 1e-12:
            u = 1e-12
        elif u >= 1.0 - 1e-12:
            u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---- init / seeding ----
    best = float("inf")
    best_x = None

    # always at least one eval
    x = rand_vec()
    fx = eval_cached(x)
    best, best_x = fx, x[:]

    # seeding budget: scale with dim but capped; also respect short max_time
    seed_n = max(16, 8 * dim)
    if max_time < 0.05:
        seed_n = max(3, min(seed_n, 10))
    else:
        seed_n = min(seed_n, 256 + 8 * dim)

    # mix halton + random + "opposition" (1 - u)
    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best

        if k <= int(0.7 * seed_n):
            cand = halton_point(k)
        else:
            cand = rand_vec()

        f = eval_cached(cand)
        if f < best:
            best, best_x = f, cand[:]

        # opposition sample (reflect within bounds around mid)
        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - cand[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

    # ---- main loop: diagonal-ES + occasional rank-1 direction + restarts ----
    x = best_x[:]
    fx = best

    # diagonal sigmas
    sigma = [0.18 * spans[i] for i in range(dim)]
    sigma_min = [1e-15 * spans[i] + 1e-18 for i in range(dim)]
    sigma_max = [0.60 * spans[i] for i in range(dim)]

    # success-based adaptation
    win = 24
    succ = 0
    tri = 0
    target = 0.22
    shrink = 0.86  # < 1; shrink when success rate low

    # a persistent direction vector (very light "CMA-ish" rank-1 memory)
    direction = [0.0] * dim
    dir_decay = 0.90

    # control knobs
    restart_every = 220
    attempts = 0
    last_best = best
    no_improve = 0

    while True:
        if time.time() >= deadline:
            return best

        attempts += 1

        # time fraction
        frac = (time.time() - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        # choose move type
        # more exploration early, more exploitation late
        p_heavy = 0.10 * (1.0 - 0.6 * frac)
        p_rank1 = 0.18

        y = x[:]

        r = random.random()
        if r < p_heavy:
            # heavy-tailed jump around best_x (escape)
            center = best_x
            k = 1 if dim <= 4 else max(1, dim // 3)
            idxs = random.sample(range(dim), k) if k < dim else list(range(dim))
            for i in idxs:
                y[i] = center[i] + cauchy() * (2.8 * sigma[i] + 1e-12)
        elif r < p_heavy + p_rank1:
            # rank-1 style step along learned direction + some noise
            # build a noisy scalar step
            step_scale = 0.9 + 0.6 * (random.random() - 0.5)
            for i in range(dim):
                # gaussian-ish noise via sum of uniforms
                g = (random.random() + random.random() + random.random() + random.random() - 2.0)
                y[i] += step_scale * direction[i] + 0.35 * g * sigma[i]
        else:
            # diagonal ES step, sparse in high dimension
            k = dim if dim <= 6 else max(2, dim // 4)
            idxs = random.sample(range(dim), k) if k < dim else list(range(dim))
            for i in idxs:
                g = (random.random() + random.random() + random.random() + random.random() - 2.0)
                y[i] += g * sigma[i]

        clip_inplace(y)
        fy = eval_cached(y)

        tri += 1
        if fy < fx:
            # update direction with accepted move
            for i in range(dim):
                step_i = (y[i] - x[i])
                direction[i] = dir_decay * direction[i] + (1.0 - dir_decay) * step_i

            x, fx = y, fy
            succ += 1
            no_improve = 0
            if fx < best:
                best, best_x = fx, x[:]
        else:
            no_improve += 1

        # 1/5-ish adaptation
        if tri >= win:
            rate = succ / float(tri)
            if rate > target:
                mult = 1.0 / shrink
            else:
                mult = shrink
            for i in range(dim):
                s = sigma[i] * mult
                if s < sigma_min[i]:
                    s = sigma_min[i]
                elif s > sigma_max[i]:
                    s = sigma_max[i]
                sigma[i] = s
            succ = 0
            tri = 0

        # periodic restart / shake logic
        if attempts % restart_every == 0:
            # if not improved, restart around best with radius tied to remaining time
            if best >= last_best - 1e-15:
                # radius: broad early, narrow late
                rad = (0.40 * (1.0 - frac) + 0.03)
                x = best_x[:]
                for i in range(dim):
                    x[i] += (2.0 * random.random() - 1.0) * (rad * spans[i])
                clip_inplace(x)
                fx = eval_cached(x)

                # reset direction and moderately reset sigmas
                direction = [0.0] * dim
                for i in range(dim):
                    sigma[i] = max(sigma_min[i], min(sigma_max[i], (0.16 + 0.10 * (1.0 - frac)) * spans[i]))
            last_best = best

        # if very stuck, do an "opposition" jump of best_x (cheap diversification)
        if no_improve > 8 * dim and time.time() < deadline:
            no_improve = 0
            opp = [lows[i] + highs[i] - best_x[i] for i in range(dim)]
            clip_inplace(opp)
            fopp = eval_cached(opp)
            if fopp < best:
                best, best_x = fopp, opp[:]
                x, fx = best_x[:], best
                direction = [0.0] * dim
