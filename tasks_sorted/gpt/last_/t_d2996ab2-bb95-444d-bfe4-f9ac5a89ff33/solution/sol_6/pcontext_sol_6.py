import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (no external libs).

    Key upgrades vs prior version:
      1) True CMA-ES (full covariance) in a small active subspace (k<=min(dim,20)):
         - Much stronger on rotated/ill-conditioned landscapes than diagonal CMA.
         - Uses a lightweight covariance update with periodic eigendecomposition.
      2) Multi-start portfolio:
         - LHS-like exploration + opposition + heavy-tail perturbations.
         - Several short CMA bursts from diverse elites, not just best.
      3) Better bound handling for CMA sampling:
         - "Resample a few times then clip" reduces boundary bias vs pure clipping.
      4) Deterministic small cache to reduce repeats.

    Returns:
      best (float): best objective value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))
    eps = 1e-12

    # -------------------- utilities --------------------
    if dim <= 0:
        try:
            y = float(func([]))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    lo = [float(bounds[i][0]) for i in range(dim)]
    hi = [float(bounds[i][1]) for i in range(dim)]
    span = [max(0.0, hi[i] - lo[i]) for i in range(dim)]
    max_span = max(span) if span else 0.0
    if max_span <= eps:
        x0 = lo[:]
        try:
            y = float(func(x0))
            return y if math.isfinite(y) else float("inf")
        except Exception:
            return float("inf")

    def time_left():
        return time.time() < deadline

    def clip(v, a, b):
        return a if v < a else (b if v > b else v)

    def project(x):
        return [clip(float(x[i]), lo[i], hi[i]) for i in range(dim)]

    def center():
        return [(lo[i] + hi[i]) * 0.5 for i in range(dim)]

    def rand_uniform():
        return [random.uniform(lo[i], hi[i]) if span[i] > eps else lo[i] for i in range(dim)]

    # quantized cache
    qstep = []
    for i in range(dim):
        s = span[i]
        qstep.append(0.0 if s <= eps else max(1e-12, 2e-7 * s))
    cache = {}
    cache_keys = []
    CACHE_MAX = 45000

    def key_of(x):
        k = []
        for i in range(dim):
            if qstep[i] == 0.0:
                k.append(0)
            else:
                k.append(int(round((x[i] - lo[i]) / qstep[i])))
        return tuple(k)

    def cache_put(k, y):
        if k in cache:
            cache[k] = y
            return
        if len(cache) >= CACHE_MAX:
            # evict random ~2.5%
            ev = max(30, CACHE_MAX // 40)
            for _ in range(ev):
                if not cache_keys:
                    break
                idx = random.randrange(len(cache_keys))
                kk = cache_keys[idx]
                cache_keys[idx] = cache_keys[-1]
                cache_keys.pop()
                cache.pop(kk, None)
                if len(cache) < CACHE_MAX:
                    break
        cache[k] = y
        cache_keys.append(k)

    def safe_eval(x):
        x = project(x)
        k = key_of(x)
        y = cache.get(k)
        if y is not None:
            return y, x
        try:
            y = func(x)
            y = float("inf") if y is None else float(y)
            if not math.isfinite(y):
                y = float("inf")
        except Exception:
            y = float("inf")
        cache_put(k, y)
        return y, x

    # Box-Muller gaussian
    _spare = [None]
    def gauss01():
        z = _spare[0]
        if z is not None:
            _spare[0] = None
            return z
        u1 = max(1e-300, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2.0 * math.pi * u2)
        z1 = r * math.sin(2.0 * math.pi * u2)
        _spare[0] = z1
        return z0

    # LHS-like stratified (independent per dimension)
    def stratified(m, shift=0):
        x = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                x[i] = lo[i]
            else:
                b = (random.randrange(m) + shift) % m
                a = lo[i] + (b / m) * span[i]
                c = lo[i] + ((b + 1) / m) * span[i]
                x[i] = random.uniform(a, c)
        return x

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            if span[i] <= eps:
                y[i] = lo[i]
            else:
                o = lo[i] + hi[i] - x[i]
                mid = 0.5 * (lo[i] + hi[i])
                # randomized opposition (less brittle)
                y[i] = clip(mid + random.random() * (o - mid), lo[i], hi[i])
        return y

    # -------------------- elites --------------------
    best = float("inf")
    x_best = center()
    fb, xb = safe_eval(x_best)
    best, x_best = fb, xb

    ELITE_MAX = max(24, 7 * dim)
    elites = [(best, x_best[:])]

    def push_elite(fx, x):
        nonlocal best, x_best, elites
        if fx < best:
            best = fx
            x_best = x[:]
        elites.append((fx, x[:]))
        if len(elites) > 6 * ELITE_MAX:
            elites.sort(key=lambda t: t[0])
            elites[:] = elites[:ELITE_MAX]

    # -------------------- local polish: bounded coordinate search --------------------
    def pattern_search(x0, f0, step0, max_iter):
        x = x0[:]
        fx = f0
        step = step0
        coords = list(range(dim))
        it = 0
        while it < max_iter and time_left():
            it += 1
            improved = False
            random.shuffle(coords)
            for j in coords:
                if span[j] <= eps:
                    continue
                cur = x[j]
                for sgn in (1.0, -1.0):
                    xj = clip(cur + sgn * step, lo[j], hi[j])
                    if abs(xj - cur) <= 1e-18:
                        continue
                    xt = x[:]
                    xt[j] = xj
                    ft, xt = safe_eval(xt)
                    if ft < fx:
                        x, fx = xt, ft
                        improved = True
                        break
                if improved and not time_left():
                    break
            step *= (1.20 if improved else 0.55)
            if step <= 1e-12 * (max_span if max_span > 0 else 1.0):
                break
        return fx, x

    # -------------------- tiny linear algebra (no numpy) --------------------
    def mat_vec(A, v):
        n = len(v)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Ai = A[i]
            for j in range(n):
                s += Ai[j] * v[j]
            out[i] = s
        return out

    def outer(u, v):
        n = len(u)
        M = [[0.0]*n for _ in range(n)]
        for i in range(n):
            ui = u[i]
            Mi = M[i]
            for j in range(n):
                Mi[j] = ui * v[j]
        return M

    def mat_add_inplace(A, B, alpha=1.0):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            Bi = B[i]
            for j in range(n):
                Ai[j] += alpha * Bi[j]

    def mat_scale_inplace(A, alpha):
        n = len(A)
        for i in range(n):
            Ai = A[i]
            for j in range(n):
                Ai[j] *= alpha

    def identity(n):
        I = [[0.0]*n for _ in range(n)]
        for i in range(n):
            I[i][i] = 1.0
        return I

    # Jacobi eigen-decomposition for symmetric matrices (small n<=20)
    def jacobi_eigh(A, max_sweeps=30):
        n = len(A)
        V = identity(n)
        D = [row[:] for row in A]

        def max_offdiag(D):
            p = 0
            q = 1
            m = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    v = abs(D[i][j])
                    if v > m:
                        m = v
                        p, q = i, j
            return m, p, q

        for _ in range(max_sweeps):
            m, p, q = max_offdiag(D)
            if m < 1e-12:
                break
            app = D[p][p]
            aqq = D[q][q]
            apq = D[p][q]
            if abs(apq) < 1e-20:
                continue
            tau = (aqq - app) / (2.0 * apq)
            t = (1.0 / (abs(tau) + math.sqrt(1.0 + tau*tau)))
            if tau < 0.0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t*t)
            s = t * c

            # rotate D
            for k in range(n):
                if k != p and k != q:
                    dkp = D[k][p]
                    dkq = D[k][q]
                    D[k][p] = c*dkp - s*dkq
                    D[p][k] = D[k][p]
                    D[k][q] = c*dkq + s*dkp
                    D[q][k] = D[k][q]

            dpp = c*c*app - 2.0*s*c*apq + s*s*aqq
            dqq = s*s*app + 2.0*s*c*apq + c*c*aqq
            D[p][p] = dpp
            D[q][q] = dqq
            D[p][q] = 0.0
            D[q][p] = 0.0

            # rotate eigenvectors
            for k in range(n):
                vkp = V[k][p]
                vkq = V[k][q]
                V[k][p] = c*vkp - s*vkq
                V[k][q] = c*vkq + s*vkp

        evals = [D[i][i] for i in range(n)]
        # sort ascending
        idx = list(range(n))
        idx.sort(key=lambda i: evals[i])
        evals2 = [evals[i] for i in idx]
        V2 = [[V[r][i] for i in idx] for r in range(n)]
        return evals2, V2

    # -------------------- CMA-ES in active subspace --------------------
    def cma_active(m0, sigma0, active_idx, budget_evals, lam=None):
        nonlocal best, x_best

        n = len(active_idx)
        if n <= 0 or budget_evals <= 0 or not time_left():
            return

        if lam is None:
            lam = max(10, 4 + int(3 * math.log(n + 1.0)))
        mu = max(2, lam // 2)

        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        sw = sum(w)
        w = [wi / sw for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # standard-ish parameters
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
        chi_n = math.sqrt(n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n))

        # state
        m = [m0[i] for i in active_idx]
        sigma = max(1e-15, float(sigma0))
        C = identity(n)
        p_c = [0.0] * n
        p_s = [0.0] * n

        # factorization B*diag(d)*B^T for C; start with I
        B = identity(n)
        d = [1.0] * n
        inv_sqrt_C = identity(n)

        evals = 0
        gen = 0
        eig_every = max(2, int(1 + (n*n) / max(1, lam)))  # periodic

        # helper to sample and respect bounds with limited resampling
        def sample_x(y):
            # y is n-dim in active subspace; embed into full x
            x = x_best[:]  # use current best as base for inactive coords
            for k, idx in enumerate(active_idx):
                x[idx] = y[k]
            return x

        while evals < budget_evals and time_left():
            gen += 1

            if gen == 1 or (gen % eig_every == 0):
                # eigendecomposition of C
                evalsC, B = jacobi_eigh(C, max_sweeps=25)
                # guard
                d = [math.sqrt(max(1e-30, ev)) for ev in evalsC]
                # invsqrtC = B*diag(1/d)*B^T
                inv_sqrt_C = [[0.0]*n for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        s = 0.0
                        for k in range(n):
                            s += B[i][k] * (1.0/d[k]) * B[j][k]
                        inv_sqrt_C[i][j] = s

            pop = []
            for _ in range(lam):
                if evals >= budget_evals or not time_left():
                    break

                # z ~ N(0,I)
                z = [gauss01() for _ in range(n)]

                # y = m + sigma * B*diag(d)*z
                # first u = diag(d)*z
                u = [d[i] * z[i] for i in range(n)]
                Bu = mat_vec(B, u)
                y = [m[i] + sigma * Bu[i] for i in range(n)]

                # bounds: resample a few times if outside
                tries = 0
                ok = True
                while tries < 3:
                    ok = True
                    for kk, idx in enumerate(active_idx):
                        if y[kk] < lo[idx] or y[kk] > hi[idx]:
                            ok = False
                            break
                    if ok:
                        break
                    # resample z
                    z = [gauss01() for _ in range(n)]
                    u = [d[i] * z[i] for i in range(n)]
                    Bu = mat_vec(B, u)
                    y = [m[i] + sigma * Bu[i] for i in range(n)]
                    tries += 1

                if not ok:
                    # last resort clip (rare)
                    for kk, idx in enumerate(active_idx):
                        y[kk] = clip(y[kk], lo[idx], hi[idx])

                x_full = sample_x(y)
                fx, x_full = safe_eval(x_full)
                evals += 1
                push_elite(fx, x_full)

                # store y relative to m for covariance update (in y-space)
                yvec = [x_full[idx] for idx in active_idx]
                # (yvec - m)/sigma
                y_w = [(yvec[i] - m[i]) / max(1e-30, sigma) for i in range(n)]
                pop.append((fx, yvec, y_w))

            if len(pop) < 2:
                break

            pop.sort(key=lambda t: t[0])

            m_old = m[:]
            # new mean
            m = [0.0] * n
            y_w_mean = [0.0] * n
            for i in range(mu):
                wi = w[i]
                yi = pop[i][1]
                ywi = pop[i][2]
                for j in range(n):
                    m[j] += wi * yi[j]
                    y_w_mean[j] += wi * ywi[j]

            # p_s update: invsqrtC * (m-m_old)/sigma  == invsqrtC * y_w_mean
            inv_term = mat_vec(inv_sqrt_C, y_w_mean)
            ccs = math.sqrt(cs * (2.0 - cs) * mueff)
            for j in range(n):
                p_s[j] = (1.0 - cs) * p_s[j] + ccs * inv_term[j]

            # sigma update
            pnorm = math.sqrt(sum(v*v for v in p_s))
            sigma *= math.exp((cs / damps) * (pnorm / chi_n - 1.0))
            sigma = max(1e-18 * max_span, min(sigma, 0.5 * max_span))

            # hsig
            hsig = 1.0 if (pnorm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) / chi_n) < (1.4 + 2.0/(n+1.0)) else 0.0

            # p_c update
            ccc = math.sqrt(cc * (2.0 - cc) * mueff)
            for j in range(n):
                p_c[j] = (1.0 - cc) * p_c[j] + hsig * ccc * y_w_mean[j]

            # covariance update:
            # C <- (1-c1-cmu)C + c1 * (p_c p_c^T + (1-hsig)cc(2-cc)C) + cmu * sum(w_i * y_i y_i^T)
            oldC = C
            C = [row[:] for row in oldC]
            a = 1.0 - c1 - cmu
            if a < 0.0:
                a = 0.0
            mat_scale_inplace(C, a)

            # rank-1
            pcpc = outer(p_c, p_c)
            mat_add_inplace(C, pcpc, alpha=c1)

            if hsig < 0.5:
                mat_add_inplace(C, oldC, alpha=c1 * cc * (2.0 - cc))

            # rank-mu
            for i in range(mu):
                wi = w[i]
                yi = pop[i][2]  # already (x-m_old)/sigma approx; acceptable
                yy = outer(yi, yi)
                mat_add_inplace(C, yy, alpha=cmu * wi)

            # keep mean in bounds
            for kk, idx in enumerate(active_idx):
                m[kk] = clip(m[kk], lo[idx], hi[idx])

            if sigma <= 1e-14 * max_span:
                break

        # end CMA

    # choose active indices: prioritize largest spans (more room to optimize)
    def choose_active(kmax):
        idx = list(range(dim))
        idx.sort(key=lambda i: span[i], reverse=True)
        active = [i for i in idx if span[i] > eps]
        return active[:max(1, min(kmax, len(active)))]

    # -------------------- initial exploration --------------------
    init_n = max(120, 30 * dim)
    m_bins = max(7, int(math.sqrt(init_n)) + 2)

    for k in range(init_n):
        if not time_left():
            return best
        x = stratified(m_bins, shift=k) if (k % 2 == 0) else rand_uniform()
        fx, x = safe_eval(x)
        push_elite(fx, x)
        if (k % 3 == 0) and time_left():
            xo = opposition(x)
            fo, xo = safe_eval(xo)
            push_elite(fo, xo)

    # -------------------- main loop portfolio --------------------
    round_id = 0
    no_improve = 0

    while time_left():
        round_id += 1
        elites.sort(key=lambda t: t[0])
        elites = elites[:min(len(elites), ELITE_MAX)]
        prev_best = best

        # active subspace size
        kact = min(dim, 18 if dim > 18 else dim)
        active_idx = choose_active(kact)

        base_sigma = 0.22 * max_span / math.sqrt(max(1.0, len(active_idx)))
        base_sigma = max(base_sigma, 1e-9 * max_span)

        # multiple short CMA bursts from diverse elites
        burst = max(50, 12 * len(active_idx))
        starts = 1 if len(elites) < 3 else 3
        for s in range(starts):
            if not time_left():
                break
            if s == 0:
                m0 = x_best[:]
                sig = base_sigma / (1.0 + 0.10 * round_id)
            else:
                # pick a non-best elite with bias to good ones
                j = 1 + int((random.random() ** 1.6) * (len(elites) - 1))
                j = min(j, len(elites) - 1)
                m0 = elites[j][1]
                sig = base_sigma * (0.65 + 0.35 * random.random())
            cma_active(m0, sig, active_idx, budget_evals=burst // starts)

        # local polish
        if time_left():
            f0, x0 = safe_eval(x_best[:])
            step0 = max(1e-7 * max_span, (0.05 / (1.0 + 0.10 * round_id)) * max_span)
            fl, xl = pattern_search(x0, f0, step0, max_iter=max(12, 2 * dim))
            push_elite(fl, xl)

        # stagnation handling: heavy-tail & restart-like exploration
        if best < prev_best:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 2 and time_left():
            burst2 = max(60, 14 * dim)
            for _ in range(burst2):
                if not time_left():
                    return best
                r = random.random()
                if r < 0.40:
                    x = rand_uniform()
                    if random.random() < 0.5:
                        x = opposition(x)
                else:
                    x = x_best[:]
                    for j in range(dim):
                        if span[j] <= eps:
                            x[j] = lo[j]
                        else:
                            u = random.random()
                            step = (0.12 * span[j]) * math.tan(math.pi * (u - 0.5))  # Cauchy-like
                            x[j] = clip(x[j] + step, lo[j], hi[j])
                fx, x = safe_eval(x)
                push_elite(fx, x)
            no_improve = 0

    return best
