import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Stronger time-bounded black-box minimizer (self-contained; no numpy).

    Main improvements vs your current best (~43.09):
      - Proper *diagonal CMA-ES* ask/tell loop (rank-1 + rank-mu update on diagonal only),
        including evolution paths and step-size control.
      - Bound handling via *reflection* (billiard), not clamping (better geometry near edges).
      - Two complementary generators:
          (1) diag-CMA samples around mean (main driver)
          (2) DE/current-to-best for global jumps
        plus a tiny coordinate pattern search for cheap final polishing.
      - Restart logic: stagnation, sigma collapse, and periodic IPOP-like population growth.
      - Evaluation frugality: generate a batch, evaluate all (since func is the real cost),
        but keep population small and updates O(pop*dim).

    Returns:
      best (float): best objective value found
    """

    # -------------------- helpers --------------------
    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    def reflect(v, lo, hi):
        if lo == hi:
            return lo
        span = hi - lo
        y = (v - lo) % (2.0 * span)
        if y > span:
            y = 2.0 * span - y
        return lo + y

    # Box-Muller
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # scrambled Halton (cheap LDS seeding)
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(math.isqrt(k))
        p = 3
        while p <= r:
            if k % p == 0:
                return False
            p += 2
        return True

    def next_prime(n):
        x = max(2, n)
        while not is_prime(x):
            x += 1
        return x

    def halton_value(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, shift):
        x = [0.0] * dim
        for j in range(dim):
            base = _PRIMES[j] if j < len(_PRIMES) else next_prime(127 + 2 * j)
            u = (halton_value(k, base) + shift[j]) % 1.0
            lo, hi = bounds[j]
            x[j] = lo + u * (hi - lo)
        return x

    def opposite_point(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    def pick3(n, exclude):
        while True:
            b = random.randrange(n)
            if b != exclude:
                break
        while True:
            c = random.randrange(n)
            if c != exclude and c != b:
                break
        while True:
            d = random.randrange(n)
            if d != exclude and d != b and d != c:
                break
        return b, c, d

    # small greedy coordinate search (first improvement)
    def coord_pattern(x, fx, step_d, tries):
        for _ in range(tries):
            j = random.randrange(dim)
            lo, hi = bounds[j]
            step = step_d[j]
            if step <= 0.0:
                continue
            xp = list(x)
            xp[j] = reflect(xp[j] + step, lo, hi)
            fp = eval_f(xp)
            if fp < fx:
                return xp, fp
            xm = list(x)
            xm[j] = reflect(xm[j] - step, lo, hi)
            fm = eval_f(xm)
            if fm < fx:
                return xm, fm
        return x, fx

    # -------------------- time --------------------
    start = time.time()
    deadline = start + max(0.0, float(max_time) if max_time is not None else 0.0)
    if dim <= 0:
        return float("inf")

    span = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    span = [s if s > 0 else 1.0 for s in span]

    # -------------------- diag-CMA-ES parameters --------------------
    # initial population size (lambda)
    lam0 = max(12, min(48, 4 + int(3.0 * math.sqrt(dim)) + 2 * dim))
    lam = lam0
    ipop_mult = 2  # increase lambda on restart
    mu = lam // 2

    # log weights
    w_raw = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
    w_sum = sum(w_raw) if sum(w_raw) > 0 else float(mu)
    w = [wi / w_sum for wi in w_raw]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # learning rates (CMA-ES defaults)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # state
    mean = [(bounds[j][0] + bounds[j][1]) * 0.5 for j in range(dim)]
    # diagonal std (sqrt(diag(C)))
    diagC = [1.0] * dim
    # start step sizes relative to span
    sigma = 0.25
    # evolution paths
    ps = [0.0] * dim
    pc = [0.0] * dim

    # clamp diagC bounds (avoid numeric issues)
    min_diag = [(1e-30) for _ in range(dim)]
    max_diag = [(1e6) for _ in range(dim)]

    # -------------------- DE backup operator params --------------------
    F_base = 0.60
    CR_base = 0.85

    # -------------------- initialization: build archive/pop --------------------
    best = float("inf")
    best_x = None
    last_improve = start

    # We maintain a small "population" of evaluated points too (for DE donors and restarts).
    pop_cap = max(16, min(80, 6 * dim + 20))
    pop = []  # list of (f, x)

    shift = [random.random() for _ in range(dim)]
    k = 1
    init_budget = max(lam, 10 * dim)

    i = 0
    while i < init_budget and time.time() < deadline:
        if (i % 4) == 0:
            x = rand_vec()
        else:
            x = halton_point(k, shift)
            k += 1
        fx = eval_f(x)
        pop.append((fx, x))
        if fx < best:
            best, best_x = fx, list(x)
            last_improve = time.time()

        if time.time() >= deadline:
            break
        if random.random() < 0.55:
            xo = opposite_point(x)
            fxo = eval_f(xo)
            pop.append((fxo, xo))
            if fxo < best:
                best, best_x = fxo, list(xo)
                last_improve = time.time()
        i += 1

    if not pop:
        return eval_f(rand_vec())

    pop.sort(key=lambda t: t[0])
    pop = pop[:pop_cap]
    best, best_x = pop[0][0], list(pop[0][1])
    mean = list(best_x)

    # restart controls
    base_stag = max(0.35, 0.10 * float(max_time))
    stag = base_stag
    restarts = 0

    def sigma_collapse():
        # sigma * sqrt(Cjj) very small for most dims
        tiny = 0
        for j in range(dim):
            if sigma * math.sqrt(max(min_diag[j], diagC[j])) < 5e-14 * span[j]:
                tiny += 1
        return tiny >= max(1, int(0.8 * dim))

    # -------------------- main loop (diag-CMA ask/tell + DE mixing) --------------------
    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0 if deadline <= start else (now - start) / (deadline - start)
        if tfrac < 0.0:
            tfrac = 0.0
        elif tfrac > 1.0:
            tfrac = 1.0

        # restart if stagnation/collapse
        if (now - last_improve) > stag or sigma_collapse():
            restarts += 1
            lam = min(160, lam * ipop_mult)  # IPOP-ish
            mu = lam // 2
            w_raw = [max(0.0, math.log(mu + 0.5) - math.log(i + 1.0)) for i in range(mu)]
            w_sum = sum(w_raw) if sum(w_raw) > 0 else float(mu)
            w = [wi / w_sum for wi in w_raw]
            mueff = 1.0 / sum(wi * wi for wi in w)

            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

            ps = [0.0] * dim
            pc = [0.0] * dim
            diagC = [1.0] * dim
            sigma = 0.35

            # re-seed mean near best but with spread
            if best_x is None:
                mean = rand_vec()
            else:
                mean = []
                for j in range(dim):
                    lo, hi = bounds[j]
                    mean.append(reflect(best_x[j] + gauss() * (0.15 + 0.25 * random.random()) * span[j], lo, hi))

            # diversify pop
            inject = min(pop_cap, max(12, 3 * dim))
            newp = []
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if best_x is not None and random.random() < 0.7:
                    x = []
                    for j in range(dim):
                        lo, hi = bounds[j]
                        x.append(reflect(best_x[j] + gauss() * 0.35 * span[j], lo, hi))
                else:
                    x = rand_vec()
                fx = eval_f(x)
                newp.append((fx, x))
                if fx < best:
                    best, best_x = fx, list(x)
                    last_improve = time.time()
            pop = (pop[:max(1, pop_cap // 3)] + newp)
            pop.sort(key=lambda t: t[0])
            pop = pop[:pop_cap]

            last_improve = time.time()
            stag = base_stag * (1.0 + 0.15 * restarts)
            continue

        # -------- ask: build lambda candidates (mostly CMA, sometimes DE) --------
        cand = []
        # precompute sqrt(diagC)
        sqrtC = [math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]

        # DE donor pool from pop
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_cap]

        for _ in range(lam):
            if time.time() >= deadline:
                break

            use_de = (len(pop) >= 6) and (random.random() < (0.20 * (1.0 - tfrac) + 0.05))
            x = [0.0] * dim

            if use_de and best_x is not None:
                # DE/current-to-best/1 with reflection
                idx = random.randrange(len(pop))
                fx_i, xi = pop[idx]
                b, c, d = pick3(len(pop), idx)
                xb = pop[b][1]
                xc = pop[c][1]
                F = F_base + 0.30 * (random.random() - 0.5)
                if F < 0.25: F = 0.25
                if F > 0.95: F = 0.95
                CR = CR_base + 0.25 * (random.random() - 0.5)
                if CR < 0.10: CR = 0.10
                if CR > 0.98: CR = 0.98
                jrand = random.randrange(dim)
                for j in range(dim):
                    lo, hi = bounds[j]
                    if random.random() < CR or j == jrand:
                        vj = xi[j] + F * (best_x[j] - xi[j]) + F * (xb[j] - xc[j])
                    else:
                        vj = xi[j]
                    x[j] = reflect(vj, lo, hi)
            else:
                # diag-CMA sample: x = mean + sigma * sqrtC * z
                for j in range(dim):
                    lo, hi = bounds[j]
                    z = gauss()
                    x[j] = reflect(mean[j] + sigma * sqrtC[j] * z, lo, hi)

            fx = eval_f(x)
            cand.append((fx, x))
            if fx < best:
                best, best_x = fx, list(x)
                last_improve = time.time()

        if not cand:
            break

        cand.sort(key=lambda t: t[0])

        # keep pop updated for DE/restarts
        pop.extend(cand[:max(2, lam // 6)])
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_cap]

        # -------- tell: update mean, paths, diagC, sigma --------
        old_mean = list(mean)

        # new mean from top mu
        mean = [0.0] * dim
        for i in range(min(mu, len(cand))):
            wi = w[i] if i < len(w) else 0.0
            xi = cand[i][1]
            for j in range(dim):
                mean[j] += wi * xi[j]

        # y = (mean - old_mean) / (sigma * sqrtC)
        y = [0.0] * dim
        inv_sigma = 1.0 / max(1e-300, sigma)
        for j in range(dim):
            denom = max(1e-300, sqrtC[j])
            y[j] = (mean[j] - old_mean[j]) * inv_sigma / denom

        # ps update
        c_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        ps_norm2 = 0.0
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + c_ps * y[j]
            ps_norm2 += ps[j] * ps[j]
        ps_norm = math.sqrt(ps_norm2)

        # sigma update
        sigma *= math.exp((cs / damps) * ((ps_norm / max(1e-300, chi_n)) - 1.0))
        if sigma < 0.03:
            sigma = 0.03
        if sigma > 6.0:
            sigma = 6.0

        # hsig and pc update
        # simplified hsig criterion
        hsig = 1.0 if ps_norm < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
        c_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for j in range(dim):
            pc[j] = (1.0 - cc) * pc[j] + hsig * c_pc * y[j]

        # rank-mu update for diagonal covariance: use top mu steps around old_mean
        z2_bar = [0.0] * dim
        for i in range(min(mu, len(cand))):
            wi = w[i] if i < len(w) else 0.0
            xi = cand[i][1]
            for j in range(dim):
                denom = max(1e-300, sqrtC[j])
                zj = (xi[j] - old_mean[j]) * inv_sigma / denom
                z2_bar[j] += wi * (zj * zj)

        # diagC update
        # Cjj <- (1 - c1 - cmu) Cjj + c1 pc^2 + cmu * z2_bar
        for j in range(dim):
            Cjj = diagC[j]
            Cjj = (1.0 - c1 - cmu) * Cjj + c1 * (pc[j] * pc[j]) + cmu * z2_bar[j]
            if Cjj < min_diag[j]:
                Cjj = min_diag[j]
            if Cjj > max_diag[j]:
                Cjj = max_diag[j]
            diagC[j] = Cjj

        # -------- small late-stage coordinate polish around best --------
        if tfrac > 0.70 and best_x is not None and random.random() < 0.25 and time.time() < deadline:
            # step per dim: sigma*sqrtC scaled down
            step_d = [(0.35 - 0.20 * tfrac) * sigma * math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]
            xb, fb = coord_pattern(best_x, best, step_d, tries=max(1, min(6, dim)))
            if fb < best:
                best, best_x = fb, list(xb)
                last_improve = time.time()
                pop.append((fb, xb))
                pop.sort(key=lambda t: t[0])
                pop = pop[:pop_cap]

    return best
