import random, math, time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvements vs the provided DE variants:
      - Uses CMA-ES (strong on ill-conditioned / rotated landscapes) with:
          * diagonal-then-full covariance switching (cheap early, powerful later)
          * mirrored sampling (variance reduction)
          * rank-μ + rank-1 updates, evolution paths, step-size control
          * bound handling via reflection (keeps sampling distribution stable)
      - Interleaves a tiny coordinate/pattern local search on current best.
      - Automatic restarts (IPOP-like) on stagnation / low progress with increasing population.

    Returns:
        best (float): best objective value found within max_time seconds
    """

    # ------------------------- basic helpers -------------------------
    def f(x):
        return float(func(x))

    def reflect(v, lo, hi):
        if hi <= lo:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if (t <= w) else (hi - (t - w))

    widths = [max(0.0, bounds[i][1] - bounds[i][0]) for i in range(dim)]
    fixed = [bounds[i][1] <= bounds[i][0] for i in range(dim)]

    def rand_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else random.uniform(lo, hi)
        return x

    def center_vec():
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = lo if hi <= lo else 0.5 * (lo + hi)
        return x

    # small local search (very cheap, robust)
    def local_refine(x0, f0, base_step, sweeps=1):
        x = x0[:]
        fx = f0
        step = base_step[:]
        for _ in range(sweeps):
            order = list(range(dim))
            random.shuffle(order)
            improved = False
            base = x[:]
            for j in order:
                if fixed[j]:
                    x[j] = bounds[j][0]
                    continue
                s = step[j]
                if s <= 0.0:
                    continue
                lo, hi = bounds[j]
                xj = x[j]

                xp = x[:]
                xp[j] = reflect(xj + s, lo, hi)
                fp = f(xp)
                if fp < fx:
                    x, fx = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[j] = reflect(xj - s, lo, hi)
                fm = f(xm)
                if fm < fx:
                    x, fx = xm, fm
                    improved = True

            # simple pattern move if improved
            if improved:
                y = x[:]
                for j in range(dim):
                    if fixed[j]:
                        y[j] = bounds[j][0]
                    else:
                        lo, hi = bounds[j]
                        y[j] = reflect(x[j] + (x[j] - base[j]), lo, hi)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
            else:
                for j in range(dim):
                    step[j] *= 0.5
        return x, fx

    # ------------------------- linear algebra helpers -------------------------
    def dot(a, b):
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    def norm(a):
        return math.sqrt(dot(a, a))

    def mat_vec(M, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Mi = M[i]
            for j in range(n):
                s += Mi[j] * v[j]
            out[i] = s
        return out

    def eye(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def symmetrize(C):
        n = len(C)
        for i in range(n):
            for j in range(i+1, n):
                v = 0.5 * (C[i][j] + C[j][i])
                C[i][j] = v
                C[j][i] = v

    def chol_lower(C):
        # Cholesky of symmetric PD matrix; returns lower-triangular L s.t. C = L L^T
        n = len(C)
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = C[i][j]
                for k in range(j):
                    s -= L[i][k]*L[j][k]
                if i == j:
                    if s <= 1e-18:
                        return None
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
        return L

    def lower_mat_vec(L, z):
        n = len(z)
        out = [0.0]*n
        for i in range(n):
            s = 0.0
            Li = L[i]
            for j in range(i+1):
                s += Li[j]*z[j]
            out[i] = s
        return out

    # ------------------------- time -------------------------
    start = time.time()
    deadline = start + float(max_time)
    if max_time <= 0 or dim <= 0:
        x = rand_vec()
        return f(x)

    # ------------------------- initialization (best-so-far) -------------------------
    best = float("inf")
    best_x = None

    # Try a few deterministic-ish points early
    probes = []
    probes.append(center_vec())
    for k in range(min(dim, 10)):
        lo, hi = bounds[k]
        if hi > lo:
            x = center_vec()
            x[k] = lo
            probes.append(x)
            x = center_vec()
            x[k] = hi
            probes.append(x)
    for _ in range(8):
        probes.append(rand_vec())

    for x in probes:
        if time.time() >= deadline:
            return best
        fx = f(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        best_x = rand_vec()
        best = f(best_x)

    # ------------------------- CMA-ES restart loop -------------------------
    # IPOP-ish: increase lambda on each restart
    restart = 0
    stagnation_limit_base = 25 + 5 * dim

    # global step baseline
    avg_w = sum(widths) / max(1, dim)
    if avg_w <= 0.0:
        return best

    while time.time() < deadline:
        # --- strategy parameters ---
        lam = int(4 + 3 * math.log(dim + 1.0))  # base
        lam = max(8, lam)
        lam = min(80 + 6 * dim, lam * (2 ** min(6, restart)))  # grow with restarts, but cap
        mu = lam // 2

        # log weights
        w = [0.0] * mu
        for i in range(mu):
            w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        wsum = sum(w)
        for i in range(mu):
            w[i] /= wsum
        mueff = 1.0 / sum(wi * wi for wi in w)

        # time-fraction affects initial sigma: later restarts more exploitative
        tfrac = (time.time() - start) / max(1e-12, (deadline - start))
        sigma = max(1e-12, (0.22 * (1.0 - 0.65 * tfrac) + 0.02) * avg_w)
        sigma *= (0.7 ** restart)

        # init mean around global best with slight random jitter
        m = best_x[:]
        for i in range(dim):
            if not fixed[i] and widths[i] > 0:
                lo, hi = bounds[i]
                m[i] = reflect(m[i] + random.gauss(0.0, 0.05 * widths[i]), lo, hi)
            else:
                m[i] = bounds[i][0]

        # covariance / paths
        diag_only = True
        D = [1.0] * dim  # diagonal stds for diag mode
        C = None
        Bchol = None

        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        pc = [0.0] * dim
        ps = [0.0] * dim

        # bookkeeping
        best_restart = float("inf")
        no_improve = 0
        evals = 0
        gen = 0
        chol_update_every = 1
        switch_full_after = 6  # generations

        while time.time() < deadline:
            gen += 1
            # switch to full covariance after some progress / if dim is moderate
            if diag_only and gen >= switch_full_after and dim <= 60:
                diag_only = False
                C = eye(dim)
                # set initial C from D
                for i in range(dim):
                    C[i][i] = max(1e-12, D[i] * D[i])
                symmetrize(C)
                Bchol = chol_lower(C)
                chol_update_every = max(1, int(0.15 * dim))

            # ensure factorization in full mode
            if not diag_only and (Bchol is None):
                # try to repair by adding jitter to diagonal
                for i in range(dim):
                    C[i][i] += 1e-10 + 1e-12 * (i + 1)
                symmetrize(C)
                Bchol = chol_lower(C)
                if Bchol is None:
                    # fallback to diag-only
                    diag_only = True
                    D = [math.sqrt(max(1e-12, C[i][i])) for i in range(dim)]
                    C = None
                    Bchol = None

            # --- sample offspring (mirrored sampling) ---
            cand = []
            half = lam // 2
            for k in range(half):
                if time.time() >= deadline:
                    break
                z = [random.gauss(0.0, 1.0) for _ in range(dim)]
                if diag_only:
                    y = [D[i] * z[i] for i in range(dim)]
                else:
                    y = lower_mat_vec(Bchol, z)

                x1 = [0.0] * dim
                x2 = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    if hi <= lo:
                        x1[i] = lo
                        x2[i] = lo
                    else:
                        x1[i] = reflect(m[i] + sigma * y[i], lo, hi)
                        x2[i] = reflect(m[i] - sigma * y[i], lo, hi)

                f1 = f(x1); evals += 1
                if f1 < best:
                    best, best_x = f1, x1[:]
                cand.append((f1, x1, y))

                if time.time() >= deadline or len(cand) >= lam:
                    break
                f2 = f(x2); evals += 1
                if f2 < best:
                    best, best_x = f2, x2[:]
                cand.append((f2, x2, [-v for v in y]))

            if not cand:
                return best

            cand.sort(key=lambda t: t[0])
            if cand[0][0] < best_restart:
                best_restart = cand[0][0]
                no_improve = 0
            else:
                no_improve += 1

            # --- recombination ---
            m_old = m[:]
            # new mean
            m = [0.0] * dim
            for i in range(mu):
                fi, xi, _yi = cand[i]
                wi = w[i]
                for j in range(dim):
                    m[j] += wi * xi[j]

            # y_w = (m - m_old) / sigma
            y_w = [(m[j] - m_old[j]) / max(1e-18, sigma) for j in range(dim)]

            # --- step-size control (ps path) ---
            if diag_only:
                # approximate C^{-1/2} y_w as y_w / D
                invsqrt_y = [y_w[i] / max(1e-18, D[i]) for i in range(dim)]
            else:
                # approx C^{-1/2} y_w via solving L * v = y_w then using v (since C = L L^T)
                # forward solve L v = y_w
                n = dim
                v = [0.0] * n
                for i in range(n):
                    s = y_w[i]
                    Li = Bchol[i]
                    for j in range(i):
                        s -= Li[j] * v[j]
                    v[i] = s / max(1e-18, Li[i])
                invsqrt_y = v

            for i in range(dim):
                ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mueff) * invsqrt_y[i]

            ps_norm = norm(ps)
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))

            # --- covariance control (pc path + update) ---
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) < (1.4 + 2.0 / (dim + 1.0)) * chiN) else 0.0
            for i in range(dim):
                pc[i] = (1.0 - cc) * pc[i] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * y_w[i]

            if diag_only:
                # diagonal covariance update using selected steps in y-space
                # D^2 <- (1-c1-cmu)D^2 + c1*pc^2 + cmu*sum w_i*y_i^2
                for j in range(dim):
                    Dj2 = D[j] * D[j]
                    rank1 = pc[j] * pc[j]
                    rankmu = 0.0
                    for i in range(mu):
                        _fi, _xi, yi = cand[i]
                        rankmu += w[i] * (yi[j] * yi[j])
                    Dj2 = (1.0 - c1 - cmu) * Dj2 + c1 * rank1 + cmu * rankmu
                    D[j] = math.sqrt(max(1e-18, Dj2))
            else:
                # full covariance update
                # C <- (1 - c1 - cmu) C + c1 * (pc pc^T + (1-hsig)*cc*(2-cc)*C) + cmu * sum w_i * yi yi^T
                factor = (1.0 - c1 - cmu)
                for a in range(dim):
                    Ca = C[a]
                    for b in range(dim):
                        Ca[b] *= factor

                # rank-1
                add_rank1 = c1
                if hsig == 0.0:
                    add_rank1 = c1 * (1.0 - cc * (2.0 - cc))  # slightly damp rank-1 when no hsig
                for a in range(dim):
                    for b in range(a, dim):
                        C[a][b] += add_rank1 * pc[a] * pc[b]
                        if b != a:
                            C[b][a] = C[a][b]

                # rank-μ
                for i in range(mu):
                    _fi, _xi, yi = cand[i]
                    wi = cmu * w[i]
                    for a in range(dim):
                        ya = yi[a]
                        for b in range(a, dim):
                            C[a][b] += wi * ya * yi[b]
                            if b != a:
                                C[b][a] = C[a][b]

                # periodic refactorization
                if (gen % chol_update_every) == 0:
                    # add tiny jitter for stability
                    for i in range(dim):
                        C[i][i] += 1e-16
                    symmetrize(C)
                    Bchol = chol_lower(C)
                    if Bchol is None:
                        # fallback: increase jitter
                        for i in range(dim):
                            C[i][i] += 1e-10
                        symmetrize(C)
                        Bchol = chol_lower(C)
                        if Bchol is None:
                            # give up to diag-only from diagonal of C
                            diag_only = True
                            D = [math.sqrt(max(1e-12, C[i][i])) for i in range(dim)]
                            C = None
                            Bchol = None

            # --- occasional local search on global best ---
            if (gen % 10) == 0 and time.time() < deadline and best_x is not None:
                step = [max(1e-12, widths[j] * (0.015 * (1.0 - 0.7 * tfrac) + 0.003)) for j in range(dim)]
                rx, rf = local_refine(best_x, best, step, sweeps=1)
                if rf < best:
                    best, best_x = rf, rx[:]
                    # pull mean slightly towards improved best
                    for j in range(dim):
                        if not fixed[j]:
                            m[j] = 0.7 * m[j] + 0.3 * best_x[j]

            # --- restart conditions ---
            if sigma < 1e-14 * avg_w:
                break
            if no_improve >= stagnation_limit_base:
                break

        restart += 1

    return best
