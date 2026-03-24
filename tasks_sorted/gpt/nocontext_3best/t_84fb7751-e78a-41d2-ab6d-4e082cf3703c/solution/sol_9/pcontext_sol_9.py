import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    What is improved vs the provided algorithms:
      - Better “anytime” behavior by splitting time into phases:
          (A) broad quasi-random seeding (scrambled Halton + LHS + opposition),
          (B) restartable Diagonal-CMA-ES core (strong local+global tradeoff),
          (C) late trust-region coordinate/pattern refinement near current best.
      - Robust bound handling via reflection (better than clipping for search dynamics).
      - Budget-aware evaluation caching with adaptive quantization.
      - Automatic restarts on stagnation with increasing population (IPOP-lite),
        plus occasional heavy-tailed injections to escape local minima.

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

    # --------------------- helpers ---------------------
    def reflect_inplace(x):
        # Reflect at bounds (better exploration than hard clip).
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            if x[i] < lo or x[i] > hi:
                if hi <= lo:
                    x[i] = lo
                    continue
                w = hi - lo
                y = (x[i] - lo) % (2.0 * w)
                if y < 0.0:
                    y += 2.0 * w
                x[i] = lo + (y if y <= w else (2.0 * w - y))
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

    # adaptive cache quantization to reduce duplicates
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
        # q from 2^16 .. 2^27
        q = 1 << (16 + int(11 * frac))
        k = quant_key(x, q)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # cheap N(0,1) using CLT
    def randn():
        return (random.random() + random.random() + random.random() +
                random.random() + random.random() + random.random() - 3.0) * 0.7071067811865475

    def cauchy():
        u = random.random()
        if u <= 1e-12: u = 1e-12
        if u >= 1.0 - 1e-12: u = 1.0 - 1e-12
        return math.tan(math.pi * (u - 0.5))

    # --------------------- low-discrepancy (scrambled Halton) ---------------------
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

    # --------------------- init / seeding ---------------------
    best_x = rand_vec()
    best = eval_cached(best_x)

    # seeding budget: keep moderate, but ensure diversity
    seed_n = max(30, 12 * dim)
    seed_n = min(seed_n, 700)
    if max_time < 0.06:
        seed_n = max(4, min(seed_n, 14))

    # LHS permutations for seeding
    strata = seed_n
    lhs_perms = []
    for i in range(dim):
        p = list(range(strata))
        random.shuffle(p)
        lhs_perms.append(p)

    # mix Halton / LHS / random + opposition
    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            return best

        r = random.random()
        if r < 0.55:
            x = halton_point(k)
        elif r < 0.88:
            kk = (k - 1) % strata
            x = [0.0] * dim
            for i in range(dim):
                a = lhs_perms[i][kk] / float(strata)
                b = (lhs_perms[i][kk] + 1) / float(strata)
                u = a + (b - a) * random.random()
                x[i] = lows[i] + u * spans[i]
        else:
            x = rand_vec()

        reflect_inplace(x)
        fx = eval_cached(x)
        if fx < best:
            best, best_x = fx, x[:]

        if time.time() >= deadline:
            return best
        opp = [lows[i] + highs[i] - x[i] for i in range(dim)]
        reflect_inplace(opp)
        fo = eval_cached(opp)
        if fo < best:
            best, best_x = fo, opp[:]

    # --------------------- Diagonal CMA-ES with restarts ---------------------
    def setup_cma(lam):
        mu = lam // 2
        ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(ws)
        w = [wi / wsum for wi in ws]
        mueff = 1.0 / sum(wi * wi for wi in w)

        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)

        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        return mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu

    def mean_span():
        return sum(spans) / float(dim) if dim > 0 else 1.0

    lam0 = 10 if dim <= 10 else (14 if dim <= 30 else 18)
    lam0 = min(max(lam0, 8), 32)

    # late-stage coordinate refinement trust radius
    tr = 0.12
    tr_min = 1e-18
    tr_max = 0.6

    # global state
    last_global_improve = time.time()
    restart = 0

    # main outer loop: restarts
    while True:
        if time.time() >= deadline:
            return best

        lam = min(72, int(lam0 * (2 ** min(restart, 4))))  # IPOPlite capped
        mu, weights, mueff, c_sigma, d_sigma, c_c, c1, c_mu = setup_cma(lam)

        # restart around best
        xmean = best_x[:]
        ps = [0.0] * dim
        pc = [0.0] * dim

        # diag std factors and global sigma
        frac = (time.time() - t0) / max_time
        if frac < 0.0: frac = 0.0
        if frac > 1.0: frac = 1.0

        sigma = (0.40 * (1.0 - frac) + 0.05) * mean_span()
        if sigma <= 0.0:
            sigma = 1.0

        D = [max(1e-12, 0.30 * spans[i]) for i in range(dim)]
        for i in range(dim):
            D[i] = min(D[i], 0.80 * spans[i] + 1e-12)

        best_local = best
        stall_gens = 0

        # inner CMA loop
        while True:
            if time.time() >= deadline:
                return best

            pop = []
            heavy_prob = 0.02 + (0.05 if dim > 25 else 0.0) + (0.02 if stall_gens > 10 else 0.0)

            for _ in range(lam):
                if time.time() >= deadline:
                    return best

                z = [randn() for _ in range(dim)]

                # occasional heavy-tail injection
                if random.random() < heavy_prob:
                    kk = 1 if dim <= 6 else max(1, dim // 3)
                    idxs = random.sample(range(dim), kk) if kk < dim else range(dim)
                    for j in idxs:
                        z[j] += 0.30 * cauchy()

                x = [xmean[i] + sigma * (D[i] * z[i]) for i in range(dim)]
                reflect_inplace(x)
                fx = eval_cached(x)
                pop.append((fx, x, z))

            pop.sort(key=lambda t: t[0])

            if pop[0][0] < best:
                best = pop[0][0]
                best_x = pop[0][1][:]
                last_global_improve = time.time()

            if pop[0][0] < best_local - 1e-15:
                best_local = pop[0][0]
                stall_gens = 0
            else:
                stall_gens += 1

            old_mean = xmean[:]
            xmean = [0.0] * dim
            zmean = [0.0] * dim
            for j in range(mu):
                w = weights[j]
                xj, zj = pop[j][1], pop[j][2]
                for i in range(dim):
                    xmean[i] += w * xj[i]
                    zmean[i] += w * zj[i]

            # update ps
            cs = c_sigma
            a = math.sqrt(cs * (2.0 - cs) * mueff)
            for i in range(dim):
                ps[i] = (1.0 - cs) * ps[i] + a * zmean[i]

            # sigma update
            norm_ps = math.sqrt(sum(p * p for p in ps))
            chi_n = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim)) if dim > 0 else 1.0
            sigma *= math.exp((cs / d_sigma) * (norm_ps / max(1e-12, chi_n) - 1.0))

            # update pc
            cc = c_c
            # simple hsig test (cheap, stable)
            hsig = 1.0 if norm_ps < (1.6 + 2.0/(dim+1.0)) * chi_n else 0.0
            b = hsig * math.sqrt(cc * (2.0 - cc) * mueff)
            inv_sigma = 1.0 / max(1e-18, sigma)
            for i in range(dim):
                pc[i] = (1.0 - cc) * pc[i] + b * (xmean[i] - old_mean[i]) * inv_sigma

            # diagonal covariance update via z^2 moments
            dz2 = [0.0] * dim
            for j in range(mu):
                w = weights[j]
                zj = pop[j][2]
                for i in range(dim):
                    dz2[i] += w * (zj[i] * zj[i])

            for i in range(dim):
                # keep positivity and stability
                Di2 = D[i] * D[i]
                v = (1.0 - c1 - c_mu) * Di2 + c1 * Di2 * (pc[i] * pc[i]) + c_mu * Di2 * dz2[i]
                if v < 1e-30:
                    v = 1e-30
                D[i] = math.sqrt(v)

                Dmin = 1e-14 * spans[i] + 1e-12
                Dmax = 2.5 * spans[i] + 1e-12
                if D[i] < Dmin: D[i] = Dmin
                if D[i] > Dmax: D[i] = Dmax

            # ---------- late-stage coordinate refinement (cheap local polish) ----------
            now = time.time()
            frac = (now - t0) / max_time
            if frac > 0.70 and (stall_gens % 6 == 0):
                center = best_x
                dims = list(range(dim))
                random.shuffle(dims)
                nd = min(dim, 12 if dim <= 40 else 16)
                dims = dims[:nd]

                step_scale = (0.65 * (1.0 - frac) + 0.12)
                improved = False
                for i in dims:
                    if time.time() >= deadline:
                        return best
                    step = tr * step_scale * spans[i]
                    if step <= 0.0:
                        continue
                    for sgn in (-1.0, 1.0):
                        y = center[:]
                        y[i] += sgn * step
                        reflect_inplace(y)
                        fy = eval_cached(y)
                        if fy < best:
                            best, best_x = fy, y[:]
                            xmean = best_x[:]  # pull mean to improvement
                            improved = True
                            break
                if improved:
                    tr = min(tr_max, tr * 1.08)
                else:
                    tr = max(tr_min, tr * 0.85)

            # ---------- restart conditions ----------
            if stall_gens > 22 + dim // 2:
                break
            if sigma < 1e-14 * mean_span() + 1e-18:
                break
            if time.time() - last_global_improve > 0.40 * max_time and stall_gens > 10:
                break

            # occasional opposition probe of mean
            if random.random() < 0.015 and time.time() < deadline:
                opp = [lows[i] + highs[i] - xmean[i] for i in range(dim)]
                reflect_inplace(opp)
                fo = eval_cached(opp)
                if fo < best:
                    best, best_x = fo, opp[:]
                    xmean = best_x[:]
                    ps = [0.0] * dim
                    pc = [0.0] * dim
                    stall_gens = 0

        restart += 1
        if time.time() + 0.02 >= deadline:
            return best
