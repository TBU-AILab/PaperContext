import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvement focus (vs your last best ~L-SHADE+coord refine):
      1) Adds CMA-ES-style local optimizer (separable + low-rank covariance) for strong end-game exploitation.
         - Uses a small population and rank-mu update.
         - Works well when DE has found a good basin.
      2) Keeps a robust global phase: L-SHADE (current-to-pbest/1 + archive) + occasional rand/1.
      3) Uses an evaluation cache (quantized) to save calls during local search.
      4) Smart schedule: more DE early, progressively more CMA near end; also triggers CMA on stagnation.
      5) Local coordinate pattern search kept as a cheap fallback when CMA budget is tight.

    Returns:
        best (float): best objective value found within max_time seconds
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    # ---------------- utilities ----------------
    def time_left():
        return time.time() < deadline

    def frac_time():
        den = max(1e-12, (deadline - t0))
        x = (time.time() - t0) / den
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def clamp01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def reflect_into_bounds(x):
        # reflection to keep "momentum" near borders (better than hard clamp for DE/CMA)
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if lo == hi:
                x[i] = lo
                continue
            span = hi - lo
            if span <= 0.0:
                x[i] = lo
                continue
            v = x[i]
            if v < lo or v > hi:
                u = (v - lo) % (2.0 * span)
                if u < 0.0:
                    u += 2.0 * span
                if u > span:
                    u = 2.0 * span - u
                v = lo + u
            x[i] = v
        return x

    # Box-Muller normal
    _has_spare = False
    _spare = 0.0
    def randn():
        nonlocal _has_spare, _spare
        if _has_spare:
            _has_spare = False
            return _spare
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-12:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare = z1
        _has_spare = True
        return z0

    def cauchy(loc, scale):
        u = random.random()
        return loc + scale * math.tan(math.pi * (u - 0.5))

    def choose_distinct(k, n, banned):
        banned = set(banned)
        out = []
        tries = 0
        while len(out) < k and tries < 80 * k:
            r = random.randrange(n)
            tries += 1
            if r in banned:
                continue
            banned.add(r)
            out.append(r)
        if len(out) < k:
            pool = [i for i in range(n) if i not in banned]
            random.shuffle(pool)
            out.extend(pool[:(k - len(out))])
        return out

    # -------------- cache (quantized) --------------
    q = 20000 if dim <= 10 else (12000 if dim <= 30 else 8000)
    cache = {}
    cache_keys = []
    cache_max = 35000

    def key_of(x):
        k = []
        for i in range(dim):
            s = spans[i]
            if s <= 0.0:
                k.append(0)
            else:
                u = (x[i] - lows[i]) / s
                if u < 0.0: u = 0.0
                if u > 1.0: u = 1.0
                k.append(int(u * q + 0.5))
        return tuple(k)

    def eval_cached(x):
        k = key_of(x)
        v = cache.get(k)
        if v is not None:
            return v
        fx = safe_eval(x)
        if len(cache) >= cache_max:
            m = max(1, cache_max // 100)  # evict ~1%
            for _ in range(m):
                if not cache_keys:
                    break
                kk = cache_keys.pop(0)
                cache.pop(kk, None)
        cache[k] = fx
        cache_keys.append(k)
        return fx

    # ---------------- cheap local search fallback ----------------
    def pattern_refine(x0, f0, eval_budget, init_rel_step):
        if x0 is None:
            return x0, f0
        x = x0[:]
        f = f0
        step = [max(1e-16, init_rel_step * spans[j]) for j in range(dim)]
        noimp = 0
        while eval_budget > 0 and time_left():
            improved = False
            axes = list(range(dim))
            random.shuffle(axes)
            for j in axes:
                if eval_budget <= 0 or not time_left():
                    break
                if step[j] <= 1e-16 * (spans[j] + 1.0):
                    continue
                base = x[j]

                cand = x[:]
                cand[j] = base + step[j]
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    improved = True
                    step[j] *= 1.25
                    continue

                if eval_budget <= 0 or not time_left():
                    break

                cand = x[:]
                cand[j] = base - step[j]
                reflect_into_bounds(cand)
                fc = eval_cached(cand)
                eval_budget -= 1
                if fc < f:
                    x, f = cand, fc
                    improved = True
                    step[j] *= 1.25
                else:
                    step[j] *= 0.60

            if improved:
                noimp = 0
            else:
                noimp += 1
                if noimp >= 2:
                    for j in range(dim):
                        step[j] *= 0.75
                    noimp = 0

            tiny = True
            for j in range(dim):
                if step[j] > 1e-14 * (spans[j] + 1.0):
                    tiny = False
                    break
            if tiny:
                break
        return x, f

    # ---------------- CMA-ES (local) ----------------
    # Light-weight CMA with low-rank covariance (rank-1 + rank-mu) and eigen-decomp via power iteration-ish
    # We implement full covariance (O(d^2)) but only run it late / on stagnation with small budgets.
    def cma_local(x_start, f_start, eval_budget, sigma_rel=0.25):
        if x_start is None or eval_budget <= 0:
            return x_start, f_start

        # dimensionless internal representation: y in R^d; x = lo + span * y, y in [0,1]
        # This improves conditioning across different bounds.
        def to_y(x):
            y = [0.0] * dim
            for i in range(dim):
                s = spans[i]
                if s <= 0.0:
                    y[i] = 0.0
                else:
                    y[i] = (x[i] - lows[i]) / s
            return y

        def to_x(y):
            x = [0.0] * dim
            for i in range(dim):
                x[i] = lows[i] + y[i] * spans[i]
            reflect_into_bounds(x)
            return x

        def y_clip(y):
            for i in range(dim):
                if y[i] < 0.0: y[i] = 0.0
                elif y[i] > 1.0: y[i] = 1.0
            return y

        # Parameters (small-pop CMA)
        lam = max(8, min(4 * dim + 8, 40))
        mu = lam // 2

        # log weights
        w = [0.0] * mu
        for i in range(mu):
            w[i] = math.log(mu + 0.5) - math.log(i + 1.0)
        wsum = sum(w)
        if wsum <= 0:
            w = [1.0 / mu] * mu
        else:
            w = [wi / wsum for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)

        # strategy parameters
        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

        # state
        m = to_y(x_start)
        sigma = max(1e-12, sigma_rel)  # in y-space (0..1)
        pc = [0.0] * dim
        ps = [0.0] * dim

        # covariance C and its decomposition B*D (C = B diag(D^2) B^T)
        # Initialize with identity.
        C = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            C[i][i] = 1.0
        B = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            B[i][i] = 1.0
        D = [1.0] * dim
        inv_sqrt_C = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            inv_sqrt_C[i][i] = 1.0

        # expected norm of N(0,I)
        chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        def mat_vec(M, v):
            out = [0.0] * dim
            for i in range(dim):
                s = 0.0
                Mi = M[i]
                for j in range(dim):
                    s += Mi[j] * v[j]
                out[i] = s
            return out

        def vec_add(a, b, scale=1.0):
            return [a[i] + scale * b[i] for i in range(dim)]

        def vec_norm(v):
            return math.sqrt(sum(vi * vi for vi in v))

        def outer(u, v):
            M = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                ui = u[i]
                for j in range(dim):
                    M[i][j] = ui * v[j]
            return M

        def mat_add_inplace(A, Bm, scale):
            for i in range(dim):
                Ai = A[i]
                Bi = Bm[i]
                for j in range(dim):
                    Ai[j] += scale * Bi[j]

        def symmetrize(A):
            for i in range(dim):
                for j in range(i + 1, dim):
                    v = 0.5 * (A[i][j] + A[j][i])
                    A[i][j] = v
                    A[j][i] = v

        def eigen_decomp_symmetric(Cm):
            # Jacobi eigenvalue algorithm (robust, no libs); O(d^3) but used sparingly.
            # Returns B (eigenvectors) and D (sqrt eigenvalues).
            # Copy
            A = [row[:] for row in Cm]
            V = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                V[i][i] = 1.0

            def max_offdiag(A):
                p = 0
                q = 1 if dim > 1 else 0
                mval = 0.0
                for i in range(dim):
                    for j in range(i + 1, dim):
                        v = abs(A[i][j])
                        if v > mval:
                            mval = v
                            p, q = i, j
                return p, q, mval

            iters = 0
            max_iters = 5 * dim * dim + 10
            while iters < max_iters:
                p, q, off = max_offdiag(A)
                if off < 1e-12:
                    break
                app = A[p][p]
                aqq = A[q][q]
                apq = A[p][q]
                if abs(apq) < 1e-18:
                    iters += 1
                    continue
                tau = (aqq - app) / (2.0 * apq)
                t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
                if tau < 0.0:
                    t = -t
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                # rotate A
                for k in range(dim):
                    if k != p and k != q:
                        akp = A[k][p]
                        akq = A[k][q]
                        A[k][p] = c * akp - s * akq
                        A[p][k] = A[k][p]
                        A[k][q] = s * akp + c * akq
                        A[q][k] = A[k][q]

                app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq
                aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq
                A[p][p] = app_new
                A[q][q] = aqq_new
                A[p][q] = 0.0
                A[q][p] = 0.0

                # rotate V
                for k in range(dim):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq

                iters += 1

            evals = [A[i][i] for i in range(dim)]
            # sort by eigenvalue descending (not required, but stable)
            order = list(range(dim))
            order.sort(key=lambda i: evals[i], reverse=True)
            Bm = [[V[i][j] for j in order] for i in range(dim)]
            Ds = []
            for idx in order:
                ev = evals[idx]
                if ev < 1e-20:
                    ev = 1e-20
                Ds.append(math.sqrt(ev))
            return Bm, Ds

        def update_inv_sqrt_C():
            # invsqrtC = B diag(1/D) B^T
            invD = [1.0 / di for di in D]
            # temp = B * diag(invD)
            temp = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                for j in range(dim):
                    temp[i][j] = B[i][j] * invD[j]
            # invsqrt = temp * B^T
            for i in range(dim):
                for j in range(dim):
                    s = 0.0
                    for k in range(dim):
                        s += temp[i][k] * B[j][k]
                    inv_sqrt_C[i][j] = s

        # periodic eigendecomposition
        evals_done = 0
        eig_period = max(1, (dim * dim) // 6)  # in evaluations

        best_x = x_start[:]
        best_f = f_start

        # Start if f_start not evaluated in cache
        if best_f == float("inf"):
            best_f = eval_cached(best_x)

        while eval_budget > 0 and time_left():
            # eigendecomposition occasionally
            if evals_done == 0 or (evals_done % eig_period == 0):
                symmetrize(C)
                B, D = eigen_decomp_symmetric(C)
                update_inv_sqrt_C()

            # sample lambda offspring
            arz = []
            ary = []
            arx = []
            arf = []

            # precompute BD = B * diag(D)
            BD = [[0.0] * dim for _ in range(dim)]
            for i in range(dim):
                for j in range(dim):
                    BD[i][j] = B[i][j] * D[j]

            for _ in range(lam):
                if eval_budget <= 0 or not time_left():
                    break
                z = [randn() for _ in range(dim)]
                # y = m + sigma * (BD * z)
                BDz = [0.0] * dim
                for i in range(dim):
                    s = 0.0
                    BDi = BD[i]
                    for j in range(dim):
                        s += BDi[j] * z[j]
                    BDz[i] = s
                y = [m[i] + sigma * BDz[i] for i in range(dim)]
                y_clip(y)
                x = to_x(y)
                fx = eval_cached(x)

                arz.append(z)
                ary.append(y)
                arx.append(x)
                arf.append(fx)

                eval_budget -= 1
                evals_done += 1

                if fx < best_f:
                    best_f = fx
                    best_x = x[:]

            if not arf:
                break

            # select best mu
            order = list(range(len(arf)))
            order.sort(key=lambda i: arf[i])
            sel = order[:max(1, min(mu, len(order)))]

            old_m = m[:]
            m = [0.0] * dim
            for k, idx in enumerate(sel):
                wk = w[k] if k < len(w) else 0.0
                yk = ary[idx]
                for i in range(dim):
                    m[i] += wk * yk[i]

            # evolution paths
            ydiff = [m[i] - old_m[i] for i in range(dim)]
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * invsqrtC * (ydiff/sigma)
            ystep = [ydiff[i] / max(1e-18, sigma) for i in range(dim)]
            invC_ystep = mat_vec(inv_sqrt_C, ystep)
            coeff = math.sqrt(c_sigma * (2.0 - c_sigma) * mueff)
            ps = [ (1.0 - c_sigma) * ps[i] + coeff * invC_ystep[i] for i in range(dim)]

            # heuristic hsig
            ps_norm = vec_norm(ps)
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (evals_done / lam + 1.0))) / chiN) < (1.4 + 2.0 / (dim + 1.0)) else 0.0

            coeff_pc = math.sqrt(c_c * (2.0 - c_c) * mueff)
            pc = [ (1.0 - c_c) * pc[i] + hsig * coeff_pc * ydiff[i] / max(1e-18, sigma) for i in range(dim)]

            # covariance update
            # C = (1 - c1 - cmu) C + c1 * (pc pc^T + (1-hsig)*cc*(2-cc)C) + cmu * sum w_i * (y_i - old_m)(...)^T / sigma^2
            # using selected y's
            factor = (1.0 - c1 - cmu)
            for i in range(dim):
                Ci = C[i]
                for j in range(dim):
                    Ci[j] *= factor

            # rank-1
            pcpc = outer(pc, pc)
            mat_add_inplace(C, pcpc, c1)

            if hsig < 0.5:
                mat_add_inplace(C, [[(c1 * c_c * (2.0 - c_c)) * (1.0 if i == j else 0.0) for j in range(dim)] for i in range(dim)], 1.0)

            # rank-mu
            for k, idx in enumerate(sel):
                wk = w[k] if k < len(w) else 0.0
                if wk <= 0.0:
                    continue
                dy = [ (ary[idx][i] - old_m[i]) / max(1e-18, sigma) for i in range(dim) ]
                mat_add_inplace(C, outer(dy, dy), cmu * wk)

            symmetrize(C)

            # step-size adaptation
            sigma *= math.exp((c_sigma / d_sigma) * (ps_norm / chiN - 1.0))
            # keep sigma sane in y-space
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 0.8:
                sigma = 0.8

        return best_x, best_f

    # ---------------- DE (global) initialization ----------------
    NP_init = max(26, min(140, 10 * dim + 40))
    NP_min = max(8, min(28, 3 * dim + 6))
    NP = NP_init

    H = max(8, min(30, NP_init // 2))
    MF = [0.6] * H
    MCR = [0.5] * H
    mem_ptr = 0

    archive = []
    archive_max = NP_init

    p_min, p_max = 0.05, 0.20

    pop = [rand_vec() for _ in range(NP)]
    fit = [float("inf")] * NP

    best = float("inf")
    best_x = None

    for i in range(NP):
        if not time_left():
            return best
        fi = eval_cached(pop[i])
        fit[i] = fi
        if fi < best:
            best = fi
            best_x = pop[i][:]

    def target_np():
        ft = frac_time()
        return int(round(NP_init - ft * (NP_init - NP_min)))

    def diversity_estimate():
        if NP < 2:
            return 0.0
        m = 6 if NP >= 10 else 3
        s = 0.0
        for _ in range(m):
            a = random.randrange(NP)
            b = random.randrange(NP)
            if a == b:
                b = (b + 1) % NP
            xa, xb = pop[a], pop[b]
            d2 = 0.0
            for j in range(dim):
                t = (xa[j] - xb[j]) / (spans[j] + 1e-300)
                d2 += t * t
            s += math.sqrt(d2 / max(1, dim))
        return s / m

    stagn = 0
    last_best = best
    last_cma_best = best
    cma_runs = 0

    while time_left():
        ft = frac_time()

        idx_sorted = sorted(range(NP), key=lambda i: fit[i])
        if fit[idx_sorted[0]] < best:
            best = fit[idx_sorted[0]]
            best_x = pop[idx_sorted[0]][:]

        if best < last_best - 1e-12:
            last_best = best
            stagn = 0
        else:
            stagn += 1

        # --- schedule: run CMA late or on stagnation (but not too often) ---
        run_cma = False
        if best_x is not None:
            if ft > 0.72 and (cma_runs < 6) and (best < last_cma_best - 1e-10 or ft > 0.90):
                run_cma = True
            elif stagn >= max(14, 5 + dim // 2) and ft > 0.35 and cma_runs < 6:
                run_cma = True

        if run_cma and time_left():
            # allocate a chunk of remaining time as eval budget (conservative)
            # Use at least a few lambdas, but don't starve DE.
            remaining = max(0.0, deadline - time.time())
            # assume unknown eval cost; use bounded eval count
            eval_budget = int(min(3000, max(80, 18 * dim + 80)))
            # if very near end, smaller
            if remaining < 0.15 * max_time:
                eval_budget = int(min(eval_budget, max(40, 10 * dim + 40)))

            bx, bf = cma_local(best_x, best, eval_budget, sigma_rel=0.25 if ft < 0.90 else 0.15)
            cma_runs += 1
            last_cma_best = min(last_cma_best, bf)
            if bf < best:
                best, best_x = bf, bx[:]
                stagn = 0
                last_best = best

        # extra cheap refinement near end
        if best_x is not None and ft > 0.80 and (stagn % 3 == 0):
            bx, bf = pattern_refine(best_x, best, eval_budget=max(10, 4 * dim + 25), init_rel_step=0.05)
            if bf < best:
                best, best_x = bf, bx[:]
                stagn = 0
                last_best = best

        # stagnation recovery
        if stagn >= max(22, 6 + dim):
            k = max(2, NP // 4)
            worst = idx_sorted[-k:]
            for wi in worst:
                if not time_left():
                    return best
                r = random.random()
                if best_x is not None and r < 0.60:
                    x = best_x[:]
                    rad = 0.22
                    for d in range(dim):
                        x[d] += rad * spans[d] * randn()
                    reflect_into_bounds(x)
                elif best_x is not None and r < 0.82:
                    x = [best_x[d] + (best_x[d] - pop[wi][d]) for d in range(dim)]
                    for d in range(dim):
                        x[d] += 0.02 * spans[d] * randn()
                    reflect_into_bounds(x)
                else:
                    x = rand_vec()
                pop[wi] = x
                fit[wi] = eval_cached(x)
            archive.clear()
            stagn = 0

        # DE exploration probability based on diversity and time
        div = diversity_estimate()
        p_explore = 0.10
        if div < 0.08:
            p_explore = 0.45
        elif div < 0.15:
            p_explore = 0.25
        if ft > 0.65:
            p_explore *= 0.6

        p = p_max - (p_max - p_min) * ft
        pcount = max(2, int(math.ceil(p * NP)))

        SCR, SF, dF = [], [], []
        union = pop + archive
        union_n = len(union)

        for i in range(NP):
            if not time_left():
                return best

            xi = pop[i]
            fi = fit[i]

            r = random.randrange(H)
            mu_cr = MCR[r]
            mu_f = MF[r]

            CR = clamp01(mu_cr + 0.10 * randn())

            F = cauchy(mu_f, 0.10)
            tries = 0
            while (F <= 0.0 or F > 1.0) and tries < 16:
                F = cauchy(mu_f, 0.10)
                tries += 1
            if F <= 0.0:
                F = 0.3 + 0.2 * random.random()
            if F > 1.0:
                F = 1.0
            F = min(1.0, max(1e-6, F * (0.95 + 0.10 * random.random())))

            use_explore = (random.random() < p_explore)

            if not use_explore:
                pbest_idx = idx_sorted[random.randrange(pcount)]
                x_pbest = pop[pbest_idx]
                r1 = choose_distinct(1, NP, banned={i, pbest_idx})[0]

                r2u = None
                for _ in range(50):
                    cand = random.randrange(union_n)
                    if cand < NP and cand in (i, pbest_idx, r1):
                        continue
                    r2u = cand
                    break
                if r2u is None:
                    r2u = random.randrange(union_n)

                x_r1 = pop[r1]
                x_r2 = union[r2u]

                donor = [0.0] * dim
                for j in range(dim):
                    donor[j] = xi[j] + F * (x_pbest[j] - xi[j]) + F * (x_r1[j] - x_r2[j])
            else:
                r1, r2, r3 = choose_distinct(3, NP, banned={i})
                x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                donor = [0.0] * dim
                for j in range(dim):
                    donor[j] = x1[j] + F * (x2[j] - x3[j])

            jrand = random.randrange(dim)
            trial = xi[:]
            for j in range(dim):
                if j == jrand or random.random() < CR:
                    trial[j] = donor[j]
            reflect_into_bounds(trial)

            ftrial = eval_cached(trial)
            if ftrial <= fi:
                if len(archive) < archive_max:
                    archive.append(xi[:])
                else:
                    archive[random.randrange(archive_max)] = xi[:]

                pop[i] = trial
                fit[i] = ftrial

                if ftrial < best:
                    best = ftrial
                    best_x = trial[:]
                    stagn = 0
                    last_best = best

                df = fi - ftrial
                if df < 0.0:
                    df = 0.0
                SCR.append(CR)
                SF.append(F)
                dF.append(df)

        if dF:
            wsum = sum(dF)
            if wsum <= 1e-18:
                w = [1.0 / len(dF)] * len(dF)
            else:
                w = [di / wsum for di in dF]

            mcr = 0.0
            for wi, cri in zip(w, SCR):
                mcr += wi * cri

            num = 0.0
            den = 0.0
            for wi, fi_ in zip(w, SF):
                num += wi * fi_ * fi_
                den += wi * fi_
            mf = (num / den) if den > 1e-12 else 0.5

            MCR[mem_ptr] = clamp01(mcr)
            MF[mem_ptr] = min(1.0, max(1e-6, mf))
            mem_ptr = (mem_ptr + 1) % H

        targ = target_np()
        if targ < NP:
            idx_sorted = sorted(range(NP), key=lambda i: fit[i])
            keep = idx_sorted[:targ]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            NP = targ
            archive_max = max(NP, NP_min)
            if len(archive) > archive_max:
                random.shuffle(archive)
                archive = archive[:archive_max]

    return best
