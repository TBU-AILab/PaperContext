import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only).

    Improvements vs. previous:
      - Uses a small CMA-ES-like evolution strategy with *full covariance* (rank-1 + rank-mu)
        in pure Python (Cholesky-based sampling). This is typically much stronger than
        diagonal-only sigma adaptation on rotated/ill-conditioned problems.
      - Still keeps low-discrepancy (Halton) global injections + occasional heavy-tail steps.
      - Adds a short deterministic coordinate/pattern refinement on the incumbent.
      - Has robust restarts (IPOP-ish) when stagnating.

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps = 1e-12

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    for i in range(dim):
        if span[i] <= 0:
            span[i] = 1.0

    def now():
        return time.time()

    def clip(v, a, b):
        if v < a: return a
        if v > b: return b
        return v

    def project(x):
        return [clip(x[i], lo[i], hi[i]) for i in range(dim)]

    # --- caching with quantization (normalized space) ---
    cache = {}
    q = 1e-9  # coarse to improve hit-rate without exploding dict

    def key_of(x):
        return tuple(int(((x[i] - lo[i]) / span[i]) / q) for i in range(dim))

    def eval_f(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(x))
        cache[k] = fx
        return fx

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(dim)]

    # -------------------- Halton sequence --------------------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
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

    bases = first_primes(dim)
    hal_k = 1

    def halton_point(k):
        x = []
        for i in range(dim):
            u = vdc(k, bases[i])
            x.append(lo[i] + u * span[i])
        return x

    # -------------------- linear algebra (stdlib) --------------------
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def mat_vec(L, z):  # L lower-triangular
        n = len(z)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            row = L[i]
            for j in range(i + 1):
                s += row[j] * z[j]
            out[i] = s
        return out

    def cholesky(A):
        # A symmetric positive definite
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 0.0:
                        s = eps
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / (L[j][j] + eps)
        return L

    # -------------------- small local refinement --------------------
    def pattern_refine(x0, f0, steps):
        x = x0[:]
        f = f0
        idxs = list(range(dim))
        random.shuffle(idxs)
        improved = False

        for i in idxs:
            si = steps[i]
            if si <= 1e-18 * span[i]:
                continue

            xp = x[:]
            xp[i] = clip(xp[i] + si, lo[i], hi[i])
            fp = eval_f(xp)
            if fp < f:
                x, f = xp, fp
                improved = True
                continue

            xm = x[:]
            xm[i] = clip(xm[i] - si, lo[i], hi[i])
            fm = eval_f(xm)
            if fm < f:
                x, f = xm, fm
                improved = True

            if now() >= deadline:
                break

        return x, f, improved

    # -------------------- initialization (strong global coverage) --------------------
    best = float("inf")
    best_x = None

    init_n = max(120, 40 * dim)
    for t in range(init_n):
        if now() >= deadline:
            return best
        r = random.random()
        if r < 0.70:
            x = halton_point(hal_k); hal_k += 1
        elif r < 0.90:
            x = rand_point()
        else:
            # jitter around current best if any
            if best_x is None:
                x = rand_point()
            else:
                x = [clip(best_x[i] + random.gauss(0.0, 0.15) * span[i], lo[i], hi[i]) for i in range(dim)]
        fx = eval_f(x)
        if fx < best:
            best, best_x = fx, x

    if best_x is None:
        return best

    # -------------------- CMA-ES-ish state --------------------
    n = dim

    # population sizing (small but robust under time constraints)
    lam0 = max(10, 4 + int(3 * math.log(n + 1.0)))
    lam = max(lam0, 4 * n // 3 + 8)
    mu = lam // 2

    # recombination weights (log)
    ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
    wsum = sum(ws)
    ws = [w / wsum for w in ws]
    mueff = 1.0 / sum(w*w for w in ws)

    # strategy params (standard defaults)
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # mean in normalized coordinates
    m = [(best_x[i] - lo[i]) / span[i] for i in range(n)]
    m = [clip(m[i], 0.0, 1.0) for i in range(n)]

    # initial global step size in normalized space
    sigma = 0.25

    # covariance matrix in normalized space
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = 1.0

    pc = [0.0] * n
    ps = [0.0] * n

    # restart / stagnation
    no_imp = 0
    stagnate = 25 + 8 * n
    heavy_tail_p = 0.06

    # local pattern step sizes in original units
    pstep = [0.08 * span[i] for i in range(n)]

    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    # main loop
    gen = 0
    while True:
        if now() >= deadline:
            return best

        gen += 1

        # Build sampling matrix (Cholesky) occasionally (cov updates happen each gen)
        # If numerical issues occur, add jitter to diagonal.
        try:
            L = cholesky(C)
        except Exception:
            for i in range(n):
                C[i][i] += 1e-10
            L = cholesky(C)

        # Sample population
        pop = []  # (fx, y, x_orig)
        for k in range(lam):
            if now() >= deadline:
                return best

            # occasional global injection to avoid premature convergence
            if random.random() < 0.10:
                x = halton_point(hal_k); hal_k += 1
                fx = eval_f(x)
                # For update, treat as if y came from current distribution (weakly)
                y = [((x[i] - lo[i]) / span[i]) - m[i] for i in range(n)]
                pop.append((fx, y, x))
                continue

            # sample in normalized space: x = m + sigma * L * z
            z = [random.gauss(0.0, 1.0) for _ in range(n)]
            if random.random() < heavy_tail_p:
                # occasional heavy-tail component
                ht = 0.35 * cauchy()
                z = [z[i] + ht * random.gauss(0.0, 1.0) for i in range(n)]

            y = mat_vec(L, z)
            xn = [m[i] + sigma * y[i] for i in range(n)]

            # project to [0,1] then to original bounds
            xn = [clip(xn[i], 0.0, 1.0) for i in range(n)]
            x = [lo[i] + xn[i] * span[i] for i in range(n)]

            fx = eval_f(x)
            pop.append((fx, y, x))

        pop.sort(key=lambda t: t[0])

        # update global best
        if pop[0][0] < best:
            best = pop[0][0]
            best_x = pop[0][2]
            no_imp = 0
        else:
            no_imp += 1

        # recombine to new mean in normalized space
        m_old = m[:]
        y_w = [0.0] * n
        for i in range(mu):
            _, y_i, x_i = pop[i]
            for j in range(n):
                y_w[j] += ws[i] * y_i[j]

        m = [m_old[j] + sigma * y_w[j] for j in range(n)]
        m = [clip(m[j], 0.0, 1.0) for j in range(n)]

        # Update evolution paths (approx; using C^{-1/2} via L^{-1} is expensive).
        # We use a practical surrogate: assume L roughly C^{1/2}; normalize by stds.
        # This still works well with full C updates below.
        diag = [max(eps, C[i][i]) for i in range(n)]
        inv_sqrt_diag = [1.0 / math.sqrt(diag[i]) for i in range(n)]

        # ps
        for i in range(n):
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * (y_w[i] * inv_sqrt_diag[i])

        # sigma control
        ps_norm = math.sqrt(dot(ps, ps))
        sigma *= math.exp((cs / damps) * (ps_norm / (chi_n + eps) - 1.0))
        sigma = max(1e-12, min(0.6, sigma))

        # hsig
        hsig = 1.0 if ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

        # pc
        for i in range(n):
            pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

        # Covariance update: rank-1 + rank-mu
        # C = (1 - c1 - cmu) C + c1 pc pc^T + cmu sum w_i y_i y_i^T
        a = 1.0 - c1 - cmu
        if a < 0.0:
            a = 0.0

        # scale old
        for i in range(n):
            Ci = C[i]
            for j in range(i + 1):
                Ci[j] *= a

        # rank-1
        c1_fac = c1
        if hsig == 0.0:
            # slight damping if path is not reliable
            c1_fac *= 0.5

        for i in range(n):
            for j in range(i + 1):
                C[i][j] += c1_fac * pc[i] * pc[j]

        # rank-mu
        for k in range(mu):
            _, yk, _ = pop[k]
            wk = cmu * ws[k]
            for i in range(n):
                for j in range(i + 1):
                    C[i][j] += wk * yk[i] * yk[j]

        # symmetrize upper triangle implicitly by mirroring when needed in cholesky:
        # (our cholesky reads A[i][j] for j<=i only)

        # numeric stability: keep diagonal positive, add tiny jitter if needed
        for i in range(n):
            if C[i][i] <= 1e-18:
                C[i][i] = 1e-18

        # occasional local refinement on best_x
        if gen % 5 == 0 and now() < deadline:
            x2, f2, imp = pattern_refine(best_x, best, pstep)
            if f2 < best:
                best, best_x = f2, x2
                no_imp = 0
                # tighten local step
                for i in range(n):
                    pstep[i] = max(1e-18 * span[i], pstep[i] * 0.75)
                # also pull mean to this improved point
                m = [clip((best_x[i] - lo[i]) / span[i], 0.0, 1.0) for i in range(n)]
            else:
                # slow decay
                for i in range(n):
                    pstep[i] = max(1e-18 * span[i], pstep[i] * 0.92)

        # restart on stagnation (IPOP-ish increase of lambda, reset covariance)
        if no_imp >= stagnate:
            no_imp = 0

            # move mean to current best
            m = [clip((best_x[i] - lo[i]) / span[i], 0.0, 1.0) for i in range(n)]

            # reset paths/cov, enlarge pop a bit, re-expand sigma
            pc = [0.0] * n
            ps = [0.0] * n
            C = [[0.0] * n for _ in range(n)]
            for i in range(n):
                C[i][i] = 1.0

            sigma = min(0.45, max(0.18, sigma * 1.8))
            lam = min(max(12, lam * 2), 200 + 10 * n)  # cap to avoid too-slow gens
            mu = lam // 2

            ws = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
            wsum = sum(ws)
            ws = [w / wsum for w in ws]
            mueff = 1.0 / sum(w*w for w in ws)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

    # unreachable
