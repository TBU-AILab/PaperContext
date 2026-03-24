import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Improvements over the provided codes:
      - Uses a "CMA-ES style" full covariance adaptation with Cholesky factor (rank-μ update)
        (not just diagonal / rank-1). This is a major upgrade on ill-conditioned problems.
      - Uses mirrored sampling to reduce noise / improve step-direction signal at same eval budget.
      - Robust boundary handling: sample in normalized space, map to real, reflect, then map back.
      - Two-stage search each generation:
          (1) CMA generation update (global+adaptive)
          (2) very cheap opportunistic local coordinate/pattern steps on best (few evals)
      - Restart schedule with IPOP-like population growth + sigma reset on stagnation.

    Returns:
        best (float): best function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    if dim <= 0:
        return float("inf")
    if bounds is None or len(bounds) != dim:
        raise ValueError("bounds must be a list of (low, high) pairs, one per dimension")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if not (s > 0.0):
            raise ValueError("Each bound must satisfy high > low")

    inv_spans = [1.0 / (spans[i] + 1e-300) for i in range(dim)]

    # ---------------- utilities ----------------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    # CLT approx N(0,1)
    def gauss01():
        return sum(random.random() for _ in range(12)) - 6.0

    def to_unit(x):
        return [(x[i] - lows[i]) * inv_spans[i] for i in range(dim)]

    def from_unit(y):
        return [lows[i] + y[i] * spans[i] for i in range(dim)]

    def reflect01_inplace(y):
        # reflect into [0,1] (can do multiple reflections if far out)
        for i in range(dim):
            v = y[i]
            if v < 0.0 or v > 1.0:
                # reflect with period 2
                v = v % 2.0
                if v > 1.0:
                    v = 2.0 - v
                y[i] = v
        return y

    def reflect_box(x):
        # reflect in real space once, then clip (safe)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            v = x[i]
            if v < lo:
                v = lo + (lo - v)
                if v > hi:
                    v = lo
            elif v > hi:
                v = hi - (v - hi)
                if v < lo:
                    v = hi
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            x[i] = v
        return x

    def random_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # ---------------- Halton (for robust init / injections) ----------------
    def first_primes(n):
        ps = []
        c = 2
        while len(ps) < n:
            ok = True
            r = int(c ** 0.5)
            for p in ps:
                if p > r:
                    break
                if c % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(c)
            c += 1
        return ps

    primes = first_primes(dim)
    halton_shift = [random.random() for _ in range(dim)]
    halton_index = 1

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n > 0:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton_point():
        nonlocal halton_index
        idx = halton_index
        halton_index += 1
        x = [0.0] * dim
        for i in range(dim):
            u = vdc(idx, primes[i]) + halton_shift[i]
            u -= int(u)
            x[i] = lows[i] + u * spans[i]
        return x

    def opposite(x):
        return [lows[i] + highs[i] - x[i] for i in range(dim)]

    # ---------------- linear algebra (no numpy): Cholesky + solves ----------------
    def eye(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def matvec(L, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Li = L[i]
            for j in range(n):
                s += Li[j] * v[j]
            out[i] = s
        return out

    def chol_decomp(A):
        # A must be symmetric PD
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                if i == j:
                    if s <= 1e-24:
                        s = 1e-24
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / (L[j][j] + 1e-300)
        return L

    def forward_solve(L, b):
        n = len(b)
        y = [0.0] * n
        for i in range(n):
            s = b[i]
            Li = L[i]
            for j in range(i):
                s -= Li[j] * y[j]
            y[i] = s / (Li[i] + 1e-300)
        return y

    def backward_solve_transpose(L, b):
        # solve L^T x = b where L is lower triangular
        n = len(b)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= L[j][i] * x[j]
            x[i] = s / (L[i][i] + 1e-300)
        return x

    # ---------------- cheap local refinement ----------------
    def local_refine(best_x, best_f, time_limit, eval_budget):
        x = best_x[:]
        fx = best_f
        steps = [0.05 * spans[i] for i in range(dim)]
        shrink = 0.55
        grow = 1.25
        evals = 0

        # random coordinate order
        order = list(range(dim))
        for k in range(dim - 1, 0, -1):
            j = int(random.random() * (k + 1))
            order[k], order[j] = order[j], order[k]

        for i in order:
            if time.time() >= time_limit or evals >= eval_budget:
                break
            si = steps[i]
            if si <= 1e-15 * (spans[i] + 1e-300):
                continue

            xi = x[i]
            xp = x[:]
            xp[i] = xi + si
            reflect_box(xp)
            fp = safe_eval(xp)
            evals += 1

            if time.time() >= time_limit or evals >= eval_budget:
                if fp < fx:
                    return xp, fp
                break

            xm = x[:]
            xm[i] = xi - si
            reflect_box(xm)
            fm = safe_eval(xm)
            evals += 1

            if fp < fx or fm < fx:
                if fp <= fm:
                    x, fx = xp, fp
                else:
                    x, fx = xm, fm
                steps[i] = min(0.25 * spans[i], steps[i] * grow)
            else:
                steps[i] *= shrink

        return x, fx

    # ---------------- initialization ----------------
    best = float("inf")
    best_x = None

    # A larger but still cheap init: mix Halton/random/opposition + corners
    init_n = max(200, min(1600, 250 + 40 * dim))
    for k in range(init_n):
        if time.time() >= deadline:
            return best

        r = random.random()
        if r < 0.15:
            x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
            for _ in range(max(1, dim // 3)):
                j = random.randrange(dim)
                x[j] = lows[j] + random.random() * spans[j]
        elif r < 0.85:
            x = halton_point()
        else:
            x = random_point()

        fx = safe_eval(x)
        if fx < best:
            best, best_x = fx, x[:]

        if time.time() >= deadline:
            return best
        xo = opposite(x)
        fo = safe_eval(xo)
        if fo < best:
            best, best_x = fo, xo[:]

    if best_x is None:
        return best

    # ---------------- CMA-ES (full covariance) with restarts ----------------
    # Work in unit space for stability
    m = to_unit(best_x)

    # initial sigma in unit space
    sigma = 0.25

    # covariance matrix in unit space, start identity
    C = eye(dim)
    L = eye(dim)  # Cholesky(C)

    # recombination weights
    def make_weights(mu):
        w = [0.0] * mu
        s = 0.0
        for i in range(mu):
            w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
            s += w[i]
        inv = 1.0 / (s + 1e-300)
        for i in range(mu):
            w[i] *= inv
        return w

    # dynamic population (IPOP-ish on restarts)
    base_lam = max(18, min(70, 14 + int(10.0 * math.log(dim + 2.0))))
    lam = base_lam
    mu = max(4, lam // 2)
    w = make_weights(mu)

    # effective mu
    mueff = 0.0
    for wi in w:
        mueff += wi * wi
    mueff = 1.0 / (mueff + 1e-300)

    # learning rates (standard-ish)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1e-300)) - 1.0) + cs

    # evolution paths
    pc = [0.0] * dim
    ps = [0.0] * dim

    # expectation of ||N(0,I)||
    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    # stagnation/restart control
    last_improve_t = time.time()
    last_best = best
    no_improve_gens = 0
    gen = 0

    # budgets for local refine
    refine_eval_budget = max(6, 2 * dim)
    refine_time_slice = max(0.02, 0.05 * float(max_time))

    while time.time() < deadline:
        gen += 1

        # rebuild Cholesky periodically or if needed
        if gen == 1 or gen % 6 == 0:
            # ensure symmetry + tiny jitter
            for i in range(dim):
                C[i][i] += 1e-18
                for j in range(i + 1, dim):
                    s = 0.5 * (C[i][j] + C[j][i])
                    C[i][j] = s
                    C[j][i] = s
            L = chol_decomp(C)

        # mirrored sampling: generate lam/2 z, then -z
        half = lam // 2
        zs = []
        off = []

        base_best = best
        successes = 0

        def eval_from_z(z):
            # y = m + sigma * L z
            step = matvec(L, z)
            y = [m[i] + sigma * step[i] for i in range(dim)]
            reflect01_inplace(y)
            x = from_unit(y)
            # extra safety in real space
            reflect_box(x)
            f = safe_eval(x)
            return f, x, y

        for _ in range(half):
            if time.time() >= deadline:
                return best
            z = [gauss01() for _ in range(dim)]
            zs.append(z)

            f1, x1, y1 = eval_from_z(z)
            off.append((f1, x1, y1, z))
            if f1 < best:
                best, best_x = f1, x1[:]
                last_improve_t = time.time()
            if f1 < base_best:
                successes += 1

            if time.time() >= deadline:
                return best
            zn = [-zi for zi in z]
            f2, x2, y2 = eval_from_z(zn)
            off.append((f2, x2, y2, zn))
            if f2 < best:
                best, best_x = f2, x2[:]
                last_improve_t = time.time()
            if f2 < base_best:
                successes += 1

        # if odd lam, add one extra
        if len(off) < lam and time.time() < deadline:
            z = [gauss01() for _ in range(dim)]
            f1, x1, y1 = eval_from_z(z)
            off.append((f1, x1, y1, z))
            if f1 < best:
                best, best_x = f1, x1[:]
                last_improve_t = time.time()
            if f1 < base_best:
                successes += 1

        off.sort(key=lambda t: t[0])
        elites = off[:mu]

        # recombination in unit space
        m_old = m[:]
        m = [0.0] * dim
        for j in range(mu):
            yj = elites[j][2]
            wj = w[j]
            for i in range(dim):
                m[i] += wj * yj[i]
        reflect01_inplace(m)

        # update paths:
        # compute zmean = (m - m_old) / sigma in coordinate system of C^{-1/2}
        dm = [(m[i] - m_old[i]) for i in range(dim)]
        # solve L * u = dm  => u = L^{-1} dm ; then zmean = u / sigma
        u = forward_solve(L, dm)
        zmean = [ui / (sigma + 1e-300) for ui in u]

        for i in range(dim):
            ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * zmean[i]

        # compute norm(ps)
        norm_ps = math.sqrt(sum(pi * pi for pi in ps))
        # hsig
        hsig = 1.0 if (norm_ps / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen + 1.0)) / (chiN + 1e-300)) < (1.4 + 2.0 / (dim + 1.0)) else 0.0

        # pc update in original coords: pc = (1-cc)pc + hsig*sqrt(cc*(2-cc)*mueff) * (m-m_old)/sigma
        y_step = [dmi / (sigma + 1e-300) for dmi in dm]
        for i in range(dim):
            pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_step[i]

        # covariance update: C = (1-c1-cmu)C + c1*(pc pc^T + (1-hsig)cc(2-cc)C) + cmu*sum w_i * y_i y_i^T
        # where y_i = (y_elite - m_old)/sigma
        # implement in-place with explicit loops (O(n^2))
        a = (1.0 - c1 - cmu)
        if a < 0.0:
            a = 0.0

        # scale C
        for i in range(dim):
            Ci = C[i]
            for j in range(dim):
                Ci[j] *= a

        # rank-1
        rank1 = [pc[i] for i in range(dim)]
        kfac = c1
        if hsig < 0.5:
            # add the "correction" term proportional to C
            corr = c1 * cc * (2.0 - cc)
            for i in range(dim):
                Ci = C[i]
                for j in range(dim):
                    Ci[j] += corr * C[i][j]  # note: uses already scaled C; acceptable as a stabilizer

        for i in range(dim):
            for j in range(i + 1):
                C[i][j] += kfac * rank1[i] * rank1[j]
                if i != j:
                    C[j][i] = C[i][j]

        # rank-mu
        for j in range(mu):
            wj = w[j]
            yj = elites[j][2]
            yy = [(yj[i] - m_old[i]) / (sigma + 1e-300) for i in range(dim)]
            for a_i in range(dim):
                for b_i in range(a_i + 1):
                    C[a_i][b_i] += cmu * wj * yy[a_i] * yy[b_i]
                    if a_i != b_i:
                        C[b_i][a_i] = C[a_i][b_i]

        # sigma update
        sigma *= math.exp((cs / damps) * (norm_ps / (chiN + 1e-300) - 1.0))
        sigma = max(1e-12, min(0.8, sigma))

        # opportunistic local refine
        if time.time() < deadline and (gen % 2 == 0):
            tl = min(deadline, time.time() + refine_time_slice)
            xr, fr = local_refine(best_x, best, tl, refine_eval_budget)
            if fr < best:
                best, best_x = fr, xr[:]
                last_improve_t = time.time()
                m = to_unit(best_x)

        # occasional injections
        if time.time() < deadline and (gen % 5 == 0):
            for _ in range(2 if dim <= 12 else 3):
                if time.time() >= deadline:
                    return best
                xg = halton_point() if random.random() < 0.7 else random_point()
                fg = safe_eval(xg)
                if fg < best:
                    best, best_x = fg, xg[:]
                    last_improve_t = time.time()
                    m = to_unit(best_x)
                if time.time() >= deadline:
                    return best
                xo = opposite(xg)
                fo = safe_eval(xo)
                if fo < best:
                    best, best_x = fo, xo[:]
                    last_improve_t = time.time()
                    m = to_unit(best_x)

        # restart logic
        if best < last_best - 1e-12:
            last_best = best
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        too_long = (time.time() - last_improve_t) > max(0.30 * float(max_time), 1.0)
        if too_long or no_improve_gens > 18:
            # IPOP-style: increase population, reset covariance, bump sigma, rescramble Halton
            halton_shift = [random.random() for _ in range(dim)]
            lam = min(200, int(lam * 2))
            mu = max(4, lam // 2)
            w = make_weights(mu)
            mueff = 0.0
            for wi in w:
                mueff += wi * wi
            mueff = 1.0 / (mueff + 1e-300)

            cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1e-300)) - 1.0) + cs

            C = eye(dim)
            L = eye(dim)
            pc = [0.0] * dim
            ps = [0.0] * dim

            sigma = min(0.6, max(0.18, sigma * 1.8))
            # re-center at best and inject a few new points
            m = to_unit(best_x)
            last_improve_t = time.time()
            no_improve_gens = 0

            inject = max(12, 2 * dim)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                x = random_point() if random.random() < 0.55 else halton_point()
                fx = safe_eval(x)
                if fx < best:
                    best, best_x = fx, x[:]
                    m = to_unit(best_x)
                    last_best = best
                if time.time() >= deadline:
                    return best
                xo = opposite(x)
                fo = safe_eval(xo)
                if fo < best:
                    best, best_x = fo, xo[:]
                    m = to_unit(best_x)
                    last_best = best

    return best
