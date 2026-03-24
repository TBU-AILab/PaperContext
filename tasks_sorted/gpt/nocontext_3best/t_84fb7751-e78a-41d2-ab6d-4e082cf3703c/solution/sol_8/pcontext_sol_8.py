import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over the provided algorithms:
      - Stronger global start: mixed Halton + LHS + random + opposition.
      - Main engine: "DE-best/2 + jitter" (very strong on bounded black-box) +
        an adaptive local trust-region pattern search around the current/best.
      - Heavy-tail escapes when stalled + periodic micro-restarts.
      - Time-adaptive evaluation cache (quantized) to avoid wasting evals.
      - Always returns best fitness found within max_time.

    Returns:
      best (float)
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

    # time-adaptive cache quantization (coarse early, finer later)
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

    def cauchy():
        u = random.random()
        if u <= 1e-12: u = 1e-12
        if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # ---------- low discrepancy (scrambled Halton) ----------
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

    # ---------- LHS sampler (one pass) ----------
    def lhs_points(n):
        # returns list of n points
        strata = max(1, int(n))
        per = []
        for i in range(dim):
            p = list(range(strata))
            random.shuffle(p)
            per.append(p)
        pts = []
        for k in range(strata):
            x = [0.0] * dim
            for i in range(dim):
                a = per[i][k] / float(strata)
                b = (per[i][k] + 1) / float(strata)
                u = a + (b - a) * random.random()
                x[i] = lows[i] + u * spans[i]
            pts.append(x)
        return pts

    # ---------- initialization / seeding ----------
    best_x = rand_vec()
    best = eval_cached(best_x)

    # keep a small population for DE (anytime friendly)
    NP = 10 if dim <= 10 else (14 if dim <= 30 else 18)
    NP = max(8, min(NP, 28))

    pop = []
    pop_f = []

    # seeding budget (time-aware)
    seed_n = min(600, max(30, 12 * dim))
    if max_time < 0.06:
        seed_n = min(seed_n, 14)

    # mix: halton + lhs + random
    lhs_n = min(seed_n // 3, 250)
    hal_n = min(seed_n // 2, 350)
    rnd_n = max(0, seed_n - lhs_n - hal_n)

    seed_pts = []
    for k in range(1, hal_n + 1):
        seed_pts.append(halton_point(k))
    seed_pts.extend(lhs_points(lhs_n))
    for _ in range(rnd_n):
        seed_pts.append(rand_vec())
    random.shuffle(seed_pts)

    for x in seed_pts:
        if time.time() >= deadline:
            return best

        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

        # opposition point
        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - x[i] for i in range(dim)]
        clip_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

        # fill DE population from best seeds
        if len(pop) < NP:
            pop.append(x[:])
            pop_f.append(fx)
        else:
            # replace worst if better
            wi = max(range(NP), key=lambda j: pop_f[j])
            if fx < pop_f[wi]:
                pop[wi] = x[:]
                pop_f[wi] = fx

    # ensure population exists
    while len(pop) < NP and time.time() < deadline:
        x = rand_vec()
        fx = eval_cached(x)
        pop.append(x)
        pop_f.append(fx)
        if fx < best:
            best, best_x = fx, x[:]

    # helper: best index in population
    def pop_best_index():
        bi = 0
        bf = pop_f[0]
        for i in range(1, NP):
            if pop_f[i] < bf:
                bf = pop_f[i]
                bi = i
        return bi

    # ---------- local pattern search around a center ----------
    def pattern_search(center, f_center, tr, max_dims):
        # try +/- steps in a subset of dims; first-improvement
        dims = list(range(dim))
        random.shuffle(dims)
        dims = dims[:max_dims]
        x0 = center
        f0 = f_center
        for i in dims:
            step = tr * spans[i]
            if step <= 0.0:
                continue
            for sgn in (-1.0, 1.0):
                if time.time() >= deadline:
                    return x0, f0, False
                y = x0[:]
                y[i] += sgn * step
                clip_inplace(y)
                fy = eval_cached(y)
                if fy < f0:
                    return y, fy, True
        return x0, f0, False

    # ---------- main loop: DE + local search + restarts ----------
    # DE parameters (adaptive-ish)
    F = 0.55
    CR = 0.90

    # trust region for pattern search
    tr = 0.12
    tr_min = 1e-16
    tr_max = 0.60

    stall = 0
    gen = 0
    last_best = best

    while True:
        if time.time() >= deadline:
            return best

        gen += 1
        frac = (time.time() - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        bi = pop_best_index()
        xb = pop[bi]
        fb = pop_f[bi]
        if fb < best:
            best, best_x = fb, xb[:]

        # --- one DE generation (NP trials) ---
        # Occasionally refresh F/CR (jDE-lite)
        if random.random() < 0.25:
            F = 0.35 + 0.55 * random.random()   # [0.35,0.90]
        if random.random() < 0.25:
            CR = 0.15 + 0.85 * random.random()  # [0.15,1.00]

        # heavy-tail injection probability rises with stagnation
        p_heavy = 0.03 + 0.02 * (1.0 - frac) + min(0.10, 0.01 * (stall // max(1, dim // 5)))

        for i in range(NP):
            if time.time() >= deadline:
                return best

            # choose distinct indices a,b,c,d != i
            idxs = list(range(NP))
            idxs.remove(i)
            a, b, c, d = random.sample(idxs, 4) if NP >= 5 else (0, 0, 0, 0)

            # DE-best/2
            v = [0.0] * dim
            for j in range(dim):
                v[j] = xb[j] + F * ((pop[a][j] - pop[b][j]) + (pop[c][j] - pop[d][j]))

            # binomial crossover
            u = pop[i][:]
            jrand = random.randrange(dim) if dim > 1 else 0
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    u[j] = v[j]

            # occasional heavy-tailed kick around global best
            if random.random() < p_heavy:
                k = 1 if dim <= 5 else max(1, dim // 3)
                sel = random.sample(range(dim), k) if k < dim else list(range(dim))
                for j in sel:
                    u[j] = best_x[j] + cauchy() * (0.25 * spans[j] * (1.0 - 0.6 * frac) + 1e-12)

            clip_inplace(u)
            fu = eval_cached(u)

            # selection
            if fu <= pop_f[i]:
                pop[i] = u
                pop_f[i] = fu
                if fu < best:
                    best, best_x = fu, u[:]

        # --- local improvement (pattern search) ---
        # Spend a bit more locally later, but keep cheap in high dim.
        if time.time() < deadline:
            center = best_x if frac > 0.35 else pop[bi]
            fcenter = best if frac > 0.35 else pop_f[bi]
            max_dims = min(dim, 10 if dim <= 40 else 14)
            x2, f2, ok = pattern_search(center, fcenter, tr * (0.9 + 0.6 * frac), max_dims)
            if ok:
                if f2 < best:
                    best, best_x = f2, x2[:]
                # inject into population by replacing worst
                wi = max(range(NP), key=lambda j: pop_f[j])
                if f2 < pop_f[wi]:
                    pop[wi] = x2[:]
                    pop_f[wi] = f2
                tr = min(tr_max, tr * 1.12)
            else:
                tr = max(tr_min, tr * 0.86)

        # --- stagnation & micro-restarts ---
        if best < last_best - 1e-15:
            stall = 0
            last_best = best
        else:
            stall += 1

        # If stalling, re-seed a few individuals around best (shrinking radius with time)
        if stall > 10 + dim // 2:
            rad = (0.45 * (1.0 - frac) + 0.06)
            n_replace = min(max(2, NP // 4), 6)
            # replace worst n_replace
            order = sorted(range(NP), key=lambda j: pop_f[j], reverse=True)
            for t in range(n_replace):
                if time.time() >= deadline:
                    return best
                j = order[t]
                x = best_x[:]
                for k in range(dim):
                    x[k] += (2.0 * random.random() - 1.0) * rad * spans[k]
                clip_inplace(x)
                fx = eval_cached(x)
                pop[j] = x
                pop_f[j] = fx
                if fx < best:
                    best, best_x = fx, x[:]
            stall = 0

        # occasional opposition probe of best
        if random.random() < 0.02 and time.time() < deadline:
            opp = [lows[i] + highs[i] - best_x[i] for i in range(dim)]
            clip_inplace(opp)
            fo = eval_cached(opp)
            if fo < best:
                best, best_x = fo, opp[:]
                wi = max(range(NP), key=lambda j: pop_f[j])
                pop[wi] = opp[:]
                pop_f[wi] = fo
