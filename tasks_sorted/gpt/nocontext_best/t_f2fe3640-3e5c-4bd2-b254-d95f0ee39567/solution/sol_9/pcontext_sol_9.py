import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only).

    Improvements vs. provided CMA-like code:
      - Uses a fast, robust "ask-tell" optimizer based on:
          * global seeding (Halton + random + best-centered)
          * multi-start adaptive (diagonal) evolution strategy (sep-CMA style)
          * occasional heavy-tail mutations to escape local basins
          * lightweight local coordinate/pattern refinement
      - Avoids expensive O(n^3) eigendecompositions entirely (diagonal covariance only),
        which is typically much faster under tight time limits.
      - Uses reflection in normalized [0,1] domain for bounds (less bias than clipping).
      - Includes evaluation caching (quantized) to reduce duplicate work.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)
    n = int(dim)
    eps = 1e-15

    lo = [float(bounds[i][0]) for i in range(n)]
    hi = [float(bounds[i][1]) for i in range(n)]
    span = [hi[i] - lo[i] for i in range(n)]
    for i in range(n):
        if not (span[i] > 0.0):
            span[i] = 1.0

    def now():
        return time.time()

    # -------- bounds handling in normalized space --------
    def reflect01(u):
        # Reflect any real number into [0,1]
        if 0.0 <= u <= 1.0:
            return u
        u = u % 2.0
        if u > 1.0:
            u = 2.0 - u
        return u

    def x_from_u(u):
        return [lo[i] + u[i] * span[i] for i in range(n)]

    # -------- caching (quantized in u-space) --------
    cache = {}
    q = 1e-10  # quantization; keep small but not too small

    def key_u(u):
        return tuple(int(reflect01(u[i]) / q) for i in range(n))

    def eval_u(u):
        ur = [reflect01(u[i]) for i in range(n)]
        k = key_u(ur)
        v = cache.get(k)
        if v is not None:
            return v, ur
        fx = float(func(x_from_u(ur)))
        cache[k] = fx
        return fx, ur

    # -------- Halton sequence for seeding --------
    def first_primes(m):
        primes = []
        x = 2
        while len(primes) < m:
            r = int(math.isqrt(x))
            ok = True
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(x)
            x += 1
        return primes

    def vdc(k, base):
        out = 0.0
        denom = 1.0
        while k:
            k, r = divmod(k, base)
            denom *= base
            out += r / denom
        return out

    bases = first_primes(n)
    hal_k = 1

    def halton_u(k):
        return [vdc(k, bases[i]) for i in range(n)]

    # -------- helpers --------
    def norm2(v):
        s = 0.0
        for x in v:
            s += x * x
        return s

    def cauchy():
        # heavy tail
        return math.tan(math.pi * (random.random() - 0.5))

    # -------- local refinement (coordinate + small random) --------
    def local_refine(u0, f0, step):
        u = u0[:]
        f = f0
        idx = list(range(n))
        random.shuffle(idx)
        improved = False

        # coordinate tries
        for i in idx:
            si = step[i]
            if si <= 1e-18:
                continue
            base = u[i]
            for sgn in (1.0, -1.0):
                uu = u[:]
                uu[i] = reflect01(base + sgn * si)
                ff, uur = eval_u(uu)
                if ff < f:
                    u, f = uur, ff
                    improved = True
            if now() >= deadline:
                break

        # tiny random tweak around best
        if now() < deadline:
            uu = u[:]
            for i in range(n):
                uu[i] = reflect01(uu[i] + random.gauss(0.0, 0.25 * step[i]))
            ff, uur = eval_u(uu)
            if ff < f:
                u, f = uur, ff
                improved = True

        return u, f, improved

    # ============================================================
    # Phase 1: global seeding
    # ============================================================
    best = float("inf")
    best_u = None

    seed_budget = max(80, 30 * n)
    for _ in range(seed_budget):
        if now() >= deadline:
            return best

        r = random.random()
        if r < 0.65:
            u = halton_u(hal_k); hal_k += 1
        elif r < 0.90:
            u = [random.random() for _ in range(n)]
        else:
            if best_u is None:
                u = [random.random() for _ in range(n)]
            else:
                u = [reflect01(best_u[i] + random.gauss(0.0, 0.20)) for i in range(n)]

        fx, ur = eval_u(u)
        if fx < best:
            best, best_u = fx, ur

    if best_u is None:
        return best

    # ============================================================
    # Phase 2: Multi-start diagonal evolution strategy (sep-CMA style)
    #   - mean m in [0,1]
    #   - per-dimension std devs s (diagonal covariance)
    #   - path-like step-size adaptation on overall sigma
    # ============================================================
    # population sizing (kept moderate for time-bounded runs)
    lam_base = max(12, 6 + int(4 * math.log(n + 1.0)))
    lam = min(64 + 2 * n, max(lam_base, 10 + n))
    mu = lam // 2

    # log weights
    w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(w)
    w = [wi / wsum for wi in w]
    mueff = 1.0 / sum(wi * wi for wi in w)

    # parameters for diag-cov update
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0)
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # state
    m = best_u[:]
    sigma = 0.30
    s = [1.0 for _ in range(n)]         # diagonal "shape" (std multipliers)
    pc = [0.0 for _ in range(n)]
    ps = [0.0 for _ in range(n)]

    # local step schedule (for local refine)
    lstep = [0.10 for _ in range(n)]

    no_imp = 0
    stagnate = 20 + 6 * n

    # restart settings
    restarts = 0
    max_restarts = 12  # bounded by time anyway

    # heavy-tail mutation probability
    ht_p = 0.08

    while True:
        if now() >= deadline:
            return best

        # Ask
        pop = []  # (fx, u, z)
        for _ in range(lam):
            if now() >= deadline:
                return best

            # occasional injection to maintain global exploration
            inj = random.random()
            if inj < 0.06:
                u = halton_u(hal_k); hal_k += 1
                fx, ur = eval_u(u)
                pop.append((fx, ur, None))
                continue
            if inj < 0.10:
                u = [random.random() for _ in range(n)]
                fx, ur = eval_u(u)
                pop.append((fx, ur, None))
                continue

            z = [random.gauss(0.0, 1.0) for _ in range(n)]
            if random.random() < ht_p:
                # heavy-tail kick
                kick = 0.35 * cauchy()
                for i in range(n):
                    z[i] += kick * random.gauss(0.0, 1.0)

            u = [0.0] * n
            for i in range(n):
                u[i] = reflect01(m[i] + sigma * s[i] * z[i])

            fx, ur = eval_u(u)
            pop.append((fx, ur, z))

        pop.sort(key=lambda t: t[0])

        if pop[0][0] < best:
            best, best_u = pop[0][0], pop[0][1]
            no_imp = 0
        else:
            no_imp += 1

        # Tell (recombine)
        m_old = m[:]
        m = [0.0] * n
        # weighted mean in u-space (since we reflect, this is "good enough" and fast)
        for k in range(mu):
            uk = pop[k][1]
            wk = w[k]
            for i in range(n):
                m[i] += wk * uk[i]
        m = [reflect01(m[i]) for i in range(n)]

        # Compute y = (m - m_old) / (sigma*s) in diag-normalized coordinates
        y = [0.0] * n
        for i in range(n):
            denom = sigma * (s[i] + eps)
            y[i] = (m[i] - m_old[i]) / denom

        # ps update (diag whitening => y already "whitened")
        c_sig = math.sqrt(cs * (2.0 - cs) * mueff)
        for i in range(n):
            ps[i] = (1.0 - cs) * ps[i] + c_sig * y[i]

        # sigma update
        psn = math.sqrt(norm2(ps))
        sigma *= math.exp((cs / ds) * (psn / (chi_n + eps) - 1.0))
        sigma = max(1e-12, min(0.80, sigma))

        # hsig and pc update
        hsig_denom = math.sqrt(1.0 - (1.0 - cs) ** (2.0 * (restarts + 1)) + eps)
        hsig = 1.0 if (psn / hsig_denom) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
        c_cum = math.sqrt(cc * (2.0 - cc) * mueff)
        for i in range(n):
            pc[i] = (1.0 - cc) * pc[i] + hsig * c_cum * (m[i] - m_old[i]) / (sigma + eps)

        # diag covariance/shape update using top mu steps relative to m_old
        # Use u-differences; robustify with tiny floor.
        # Update s^2 multiplicatively from weighted second moments.
        # Equivalent-ish to sep-CMA with diagonal C.
        # Compute weighted variance in normalized coordinates:
        v = [0.0] * n
        for k in range(mu):
            uk = pop[k][1]
            wk = w[k]
            for i in range(n):
                d = (uk[i] - m_old[i]) / (sigma + eps)
                v[i] += wk * d * d

        # apply rank-one (pc) and rank-mu (v) updates on diagonal
        # s represents stddev shape => s^2 corresponds to diag(C)
        for i in range(n):
            ci = s[i] * s[i]
            ci *= max(0.0, 1.0 - c1 - cmu)
            ci += c1 * (pc[i] * pc[i])
            ci += cmu * v[i]
            # keep sane
            if ci < 1e-18:
                ci = 1e-18
            if ci > 1e6:
                ci = 1e6
            s[i] = math.sqrt(ci)

        # local refine periodically around best
        if best_u is not None and (no_imp % 5 == 0) and now() < deadline:
            u2, f2, imp = local_refine(best_u, best, lstep)
            if f2 < best:
                best, best_u = f2, u2
                no_imp = 0
                # pull mean towards improved point
                m = best_u[:]
                # shrink local steps on success
                for i in range(n):
                    lstep[i] = max(1e-18, lstep[i] * 0.75)
            else:
                # slow decay
                for i in range(n):
                    lstep[i] = max(1e-18, lstep[i] * 0.93)

        # stagnation restart
        if no_imp >= stagnate and restarts < max_restarts:
            restarts += 1
            no_imp = 0

            # restart near best with larger sigma and reset diagonal shape
            if best_u is not None:
                m = best_u[:]
            else:
                m = [random.random() for _ in range(n)]

            sigma = min(0.60, max(0.20, sigma * 1.8))
            s = [1.0 for _ in range(n)]
            pc = [0.0 for _ in range(n)]
            ps = [0.0 for _ in range(n)]

            # vary population size a bit (helps on some landscapes)
            lam = min(220 + 6 * n, max(lam_base, int(lam * 1.25)))
            mu = lam // 2
            w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            wsum = sum(w)
            w = [wi / wsum for wi in w]
            mueff = 1.0 / sum(wi * wi for wi in w)

            cs = (mueff + 2.0) / (n + mueff + 5.0)
            ds = 1.0 + cs + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # unreachable
    # return best
