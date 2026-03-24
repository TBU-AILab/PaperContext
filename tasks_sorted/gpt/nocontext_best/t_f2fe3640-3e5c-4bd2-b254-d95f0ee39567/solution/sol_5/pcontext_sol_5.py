import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (stdlib-only).

    Improved vs. provided code:
      1) Proper CMA-ES sampling and step-size control:
         - Uses C^{-1/2} * (m-m_old)/sigma via solving with the Cholesky factor
           (instead of the diagonal surrogate), which materially improves sigma control.
      2) Mirrors boundary handling (instead of hard clipping) to reduce boundary bias.
      3) Occasional elitist local search around incumbent (stochastic coordinate search).
      4) Restarts with increasing population (IPOP-lite) and sigma reset on stagnation.
      5) Caching with quantization remains (helps when func is expensive/noisy-ish).

    Returns:
      best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps = 1e-12

    n = int(dim)
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(n)]
    for i in range(n):
        if not (span[i] > 0.0):
            span[i] = 1.0

    def now():
        return time.time()

    # -------- robust boundary handling: mirror into [lo,hi] --------
    def mirror_into(v, a, b):
        if a == b:
            return a
        w = b - a
        x = (v - a) % (2.0 * w)
        if x > w:
            x = 2.0 * w - x
        return a + x

    def project_mirror(x):
        return [mirror_into(x[i], lo[i], hi[i]) for i in range(n)]

    # -------- caching (quantized normalized coordinates) ----------
    cache = {}
    q = 1e-9

    def key_of(x):
        # quantize normalized coords; stable keys
        k = []
        for i in range(n):
            u = (x[i] - lo[i]) / span[i]
            k.append(int(u / q))
        return tuple(k)

    def eval_f(x):
        xx = project_mirror(x)
        k = key_of(xx)
        v = cache.get(k)
        if v is not None:
            return v
        fx = float(func(xx))
        cache[k] = fx
        return fx

    # ------------------- Halton for global init -------------------
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

    def halton_point(k):
        x = []
        for i in range(n):
            u = vdc(k, bases[i])
            x.append(lo[i] + u * span[i])
        return x

    def rand_point():
        return [lo[i] + random.random() * span[i] for i in range(n)]

    # --------------------- linear algebra -------------------------
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(dot(a, a))

    def cholesky_lower(A):
        # A stored as full n x n, but we only rely on symmetry.
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

    def mat_vec_lower(L, z):
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Li = L[i]
            for j in range(i + 1):
                s += Li[j] * z[j]
            out[i] = s
        return out

    def solve_lower(L, b):
        # L x = b
        x = [0.0] * n
        for i in range(n):
            s = b[i]
            Li = L[i]
            for j in range(i):
                s -= Li[j] * x[j]
            x[i] = s / (Li[i] + eps)
        return x

    def solve_upper_from_lower(L, b):
        # solve L^T x = b
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= L[j][i] * x[j]
            x[i] = s / (L[i][i] + eps)
        return x

    # ------------------ local search (cheap) ----------------------
    def local_search(best_x, best_f, budget, base_scale):
        # stochastic coordinate search with decreasing step
        x = best_x[:]
        f = best_f
        step = [base_scale * span[i] for i in range(n)]
        for _ in range(budget):
            if now() >= deadline:
                break
            i = random.randrange(n)
            s = step[i]
            if s <= 1e-18 * span[i]:
                continue
            # try +/- step with random factor
            r = (0.5 + random.random())  # [0.5,1.5)
            delta = s * r
            cand1 = x[:]
            cand1[i] = cand1[i] + delta
            f1 = eval_f(cand1)
            if f1 < f:
                x, f = project_mirror(cand1), f1
                step[i] *= 1.1
                continue
            cand2 = x[:]
            cand2[i] = cand2[i] - delta
            f2 = eval_f(cand2)
            if f2 < f:
                x, f = project_mirror(cand2), f2
                step[i] *= 1.1
            else:
                step[i] *= 0.85
        return x, f

    # -------------------- initialization --------------------------
    best = float("inf")
    best_x = None

    init_n = max(160, 50 * n)
    for _ in range(init_n):
        if now() >= deadline:
            return best
        r = random.random()
        if r < 0.75:
            x = halton_point(hal_k); hal_k += 1
        elif r < 0.95:
            x = rand_point()
        else:
            if best_x is None:
                x = rand_point()
            else:
                x = [best_x[i] + random.gauss(0.0, 0.2) * span[i] for i in range(n)]
        fx = eval_f(x)
        if fx < best:
            best = fx
            best_x = project_mirror(x)

    if best_x is None:
        return best

    # ------------------- CMA-ES parameters ------------------------
    # initial pop size
    lam_base = max(12, 4 + int(3 * math.log(n + 1.0)))
    lam = max(lam_base, 4 * n + 8)
    mu = lam // 2

    def recomb_weights(mu_):
        ws_ = [math.log(mu_ + 0.5) - math.log(i + 1.0) for i in range(mu_)]
        s_ = sum(ws_)
        ws_ = [w / s_ for w in ws_]
        mueff_ = 1.0 / sum(w * w for w in ws_)
        return ws_, mueff_

    ws, mueff = recomb_weights(mu)

    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # mean in normalized [0,1]
    m = [(best_x[i] - lo[i]) / span[i] for i in range(n)]
    # start sigma moderate
    sigma = 0.30

    # covariance matrix (full), start identity
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = 1.0

    pc = [0.0] * n
    ps = [0.0] * n

    # restart / stagnation
    no_imp = 0
    stagnate = 20 + 10 * n
    restart_count = 0

    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    gen = 0
    while True:
        if now() >= deadline:
            return best

        gen += 1

        # Cholesky of C
        # Add small diagonal jitter if needed
        for i in range(n):
            if C[i][i] <= 1e-18:
                C[i][i] = 1e-18
        try:
            L = cholesky_lower(C)
        except Exception:
            for i in range(n):
                C[i][i] += 1e-10
            L = cholesky_lower(C)

        # sample population
        pop = []  # (f, z, y, xn)
        heavy_p = 0.05
        inject_p = 0.08

        for _ in range(lam):
            if now() >= deadline:
                return best

            if random.random() < inject_p:
                x = halton_point(hal_k); hal_k += 1
                fx = eval_f(x)
                # fabricate z/y for update as zero-mean perturb around m (weakly)
                xn = [(project_mirror(x)[i] - lo[i]) / span[i] for i in range(n)]
                y = [(xn[i] - m[i]) / max(eps, sigma) for i in range(n)]
                z = y[:]  # not exact, but keeps update from breaking
                pop.append((fx, z, y, xn))
                continue

            z = [random.gauss(0.0, 1.0) for _ in range(n)]
            if random.random() < heavy_p:
                ht = 0.30 * cauchy()
                z = [z[i] + ht * random.gauss(0.0, 1.0) for i in range(n)]

            y = mat_vec_lower(L, z)              # y ~ N(0, C)
            xn = [m[i] + sigma * y[i] for i in range(n)]
            # map to real space with mirror (avoid bias of clipping in normalized)
            x = [lo[i] + xn[i] * span[i] for i in range(n)]
            fx = eval_f(x)
            # store xn as raw (can be outside), but for recombination use y/z
            pop.append((fx, z, y, xn))

        pop.sort(key=lambda t: t[0])

        if pop[0][0] < best:
            best = pop[0][0]
            # convert best xn to original bounded point (mirrored)
            bx = [lo[i] + pop[0][3][i] * span[i] for i in range(n)]
            best_x = project_mirror(bx)
            no_imp = 0
        else:
            no_imp += 1

        # recombination in y-space
        y_w = [0.0] * n
        z_w = [0.0] * n
        for i in range(mu):
            _, z_i, y_i, _ = pop[i]
            wi = ws[i]
            for j in range(n):
                y_w[j] += wi * y_i[j]
                z_w[j] += wi * z_i[j]

        m_old = m[:]
        for j in range(n):
            m[j] = m_old[j] + sigma * y_w[j]

        # ---- evolution path update using proper C^{-1/2} ----
        # ps <- (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} * (m-m_old)/sigma
        # where (m-m_old)/sigma = y_w
        # C^{-1/2} y_w = (L^{-T}) (L^{-1} y_w)
        tmp = solve_lower(L, y_w)
        cinv_sqrt_yw = solve_upper_from_lower(L, tmp)

        cfac = math.sqrt(cs * (2.0 - cs) * mueff)
        for i in range(n):
            ps[i] = (1.0 - cs) * ps[i] + cfac * cinv_sqrt_yw[i]

        ps_norm = norm(ps)

        # step-size control
        sigma *= math.exp((cs / damps) * (ps_norm / (chi_n + eps) - 1.0))
        # keep sigma within reasonable range
        if sigma < 1e-12:
            sigma = 1e-12
        if sigma > 0.9:
            sigma = 0.9

        # hsig
        denom = math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) + eps
        hsig = 1.0 if (ps_norm / denom) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

        # pc
        ccfac = math.sqrt(cc * (2.0 - cc) * mueff)
        for i in range(n):
            pc[i] = (1.0 - cc) * pc[i] + hsig * ccfac * y_w[i]

        # covariance update (lower triangle only, but store full for simplicity)
        a = 1.0 - c1 - cmu
        if a < 0.0:
            a = 0.0

        # scale old
        for i in range(n):
            for j in range(i + 1):
                C[i][j] *= a
                if i != j:
                    C[j][i] = C[i][j]

        # rank-1
        c1_eff = c1 * (0.5 if hsig == 0.0 else 1.0)
        for i in range(n):
            for j in range(i + 1):
                C[i][j] += c1_eff * pc[i] * pc[j]
                if i != j:
                    C[j][i] = C[i][j]

        # rank-mu
        for k in range(mu):
            _, _, yk, _ = pop[k]
            wk = cmu * ws[k]
            for i in range(n):
                yi = yk[i]
                for j in range(i + 1):
                    C[i][j] += wk * yi * yk[j]
                    if i != j:
                        C[j][i] = C[i][j]

        # keep diagonal sane
        for i in range(n):
            if C[i][i] < 1e-18:
                C[i][i] = 1e-18

        # occasional local search around incumbent
        if gen % 6 == 0 and now() < deadline and best_x is not None:
            bx, bf = local_search(best_x, best, budget=8 + n, base_scale=0.06)
            if bf < best:
                best, best_x = bf, bx
                # pull mean to improved incumbent
                m = [(best_x[i] - lo[i]) / span[i] for i in range(n)]
                no_imp = 0

        # restart on stagnation
        if no_imp >= stagnate:
            no_imp = 0
            restart_count += 1

            # reset around best
            m = [(best_x[i] - lo[i]) / span[i] for i in range(n)]
            pc = [0.0] * n
            ps = [0.0] * n
            C = [[0.0] * n for _ in range(n)]
            for i in range(n):
                C[i][i] = 1.0

            # IPOP-ish: increase lambda
            lam = min(int(lam * 1.6) + 1, 250 + 12 * n)
            mu = lam // 2
            ws, mueff = recomb_weights(mu)

            # refresh strategy params
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

            # reset sigma (slightly larger after more restarts)
            sigma = min(0.55, 0.25 * (1.15 ** min(10, restart_count)))
