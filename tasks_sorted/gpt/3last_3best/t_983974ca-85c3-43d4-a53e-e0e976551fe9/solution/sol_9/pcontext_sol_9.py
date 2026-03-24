import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimizer (no external libs).

    Improvement over your best (~27.95):
      - Keep JADE current-to-pbest/1 + archive (strong global engine)
      - Add *Eigen coordinate system* exploitation:
          periodically estimate a covariance from elites and sample/mutate in rotated space
          (cheap "CMA-ES flavor" without full CMA-ES)
      - Add *lightweight Nelder–Mead* (bounded) in the last part of the budget (fast local squeeze)
      - Use *ring/topology DE* option sometimes (improves diversity on rugged landscapes)
      - Reduce expensive full sorts: use partial top-k and occasional full sort only

    Returns:
        best fitness (float)
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------- edge cases ----------
    if dim <= 0:
        try:
            v = func([])
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]

    # fix inverted / degenerate
    spans = [0.0] * dim
    for i in range(dim):
        lo, hi = lows[i], highs[i]
        if lo > hi:
            lo, hi = hi, lo
            lows[i], highs[i] = lo, hi
        s = hi - lo
        if s <= 0.0:
            s = 1.0
        spans[i] = s

    # ---------- helpers ----------
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

    def repair_reflect(x):
        # reflect into bounds; if still out -> random reinsert
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            xi = x[i]
            if xi < lo:
                xi = lo + (lo - xi)
                if xi > hi:
                    xi = lo + random.random() * (hi - lo) if hi > lo else lo
            elif xi > hi:
                xi = hi - (xi - hi)
                if xi < lo:
                    xi = lo + random.random() * (hi - lo) if hi > lo else lo
            x[i] = xi
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    def opposite(x):
        xo = [0.0] * dim
        for i in range(dim):
            xo[i] = lows[i] + highs[i] - x[i]
        return repair_reflect(xo)

    def corner_vec(jitter=0.02):
        x = [lows[i] if random.random() < 0.5 else highs[i] for i in range(dim)]
        if jitter > 0.0:
            for i in range(dim):
                x[i] += random.gauss(0.0, jitter * spans[i])
        return repair_reflect(x)

    def lhs_points(n):
        perms = []
        for d in range(dim):
            p = list(range(n))
            random.shuffle(p)
            perms.append(p)
        invn = 1.0 / float(n)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for d in range(dim):
                u = (perms[d][k] + random.random()) * invn
                x[d] = lows[d] + u * (highs[d] - lows[d])
            pts.append(x)
        return pts

    def randn_clip2():
        while True:
            z = random.gauss(0.0, 1.0)
            if -2.0 <= z <= 2.0:
                return z

    def cauchy():
        return math.tan(math.pi * (random.random() - 0.5))

    def topk_indices(fits, k):
        best_list = []
        for idx, f in enumerate(fits):
            if len(best_list) < k:
                best_list.append((f, idx))
                if len(best_list) == k:
                    best_list.sort(key=lambda t: t[0])
            else:
                if f < best_list[-1][0]:
                    j = k - 1
                    while j > 0 and f < best_list[j - 1][0]:
                        j -= 1
                    best_list.insert(j, (f, idx))
                    best_list.pop()
        return [idx for _, idx in best_list]

    def pick_excluding(n, forbid):
        j = random.randrange(n)
        while j in forbid:
            j = random.randrange(n)
        return j

    # ----- linear algebra (no numpy): Jacobi eigen for symmetric matrix -----
    def jacobi_eigen_sym(A, iters=40):
        """Return eigenvectors V (cols) and eigenvalues diag as list for symmetric A."""
        n = len(A)
        # V = I
        V = [[0.0] * n for _ in range(n)]
        for i in range(n):
            V[i][i] = 1.0

        def max_offdiag(M):
            p = 0
            q = 1 if n > 1 else 0
            m = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    v = abs(M[i][j])
                    if v > m:
                        m = v
                        p, q = i, j
            return p, q, m

        for _ in range(iters):
            p, q, m = max_offdiag(A)
            if m < 1e-12:
                break
            app = A[p][p]
            aqq = A[q][q]
            apq = A[p][q]

            if apq == 0.0:
                continue

            tau = (aqq - app) / (2.0 * apq)
            # t = sign(tau)/(abs(tau)+sqrt(1+tau^2)) for stability
            if tau >= 0.0:
                t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
            else:
                t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c

            # rotate A
            A[p][p] = app - t * apq
            A[q][q] = aqq + t * apq
            A[p][q] = 0.0
            A[q][p] = 0.0

            for k in range(n):
                if k != p and k != q:
                    aik = A[k][p]
                    akq = A[k][q]
                    A[k][p] = aik * c - akq * s
                    A[p][k] = A[k][p]
                    A[k][q] = aik * s + akq * c
                    A[q][k] = A[k][q]

            # rotate V
            for k in range(n):
                vip = V[k][p]
                viq = V[k][q]
                V[k][p] = vip * c - viq * s
                V[k][q] = vip * s + viq * c

        evals = [A[i][i] for i in range(n)]
        return V, evals

    def mat_vec(M, v):
        n = len(M)
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Mi = M[i]
            for j in range(n):
                s += Mi[j] * v[j]
            out[i] = s
        return out

    def matT_vec(M, v):
        n = len(M)
        out = [0.0] * n
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += M[i][j] * v[i]
            out[j] = s
        return out

    # ---------- parameters ----------
    pop_size = int(14 + 6 * math.log(dim + 1.0))
    pop_size = max(24, min(96, pop_size))
    archive_max = pop_size

    # JADE adaptation
    c_adapt = 0.1
    mu_F = 0.6
    mu_CR = 0.9

    # local coordinate polish (cheap)
    min_step = 1e-15
    polish_every = 9
    polish_coords = min(dim, max(8, int(0.40 * dim)))
    polish_step = [max(min_step, 0.012 * spans[i]) for i in range(dim)]

    # eigen exploitation
    elite_k = max(6, min(pop_size, 8 + pop_size // 6))
    eig_every = 10
    eig_samples = max(6, min(18, 2 + dim // 3))
    eig_vecs = None     # V
    eig_vals = None     # evals
    eig_scale = [0.20 * spans[i] for i in range(dim)]  # used if eigen fails

    # stagnation
    last_improve_t = time.time()
    stagnate_time = max(0.25 * max_time, 0.8)

    # ---------- initialization ----------
    init_until = min(deadline, t0 + 0.18 * max_time)

    candidates = []
    n_lhs = max(10, min(pop_size, int(12 + 6 * math.log(dim + 1.0))))
    candidates.extend(lhs_points(n_lhs))
    for _ in range(max(2, pop_size // 6)):
        candidates.append(corner_vec(0.02))

    while len(candidates) < pop_size:
        x = rand_vec()
        candidates.append(x)
        if len(candidates) < pop_size:
            candidates.append(opposite(x))

    pop, fits = [], []
    best = float("inf")
    best_x = None

    for x in candidates:
        if time.time() >= init_until and len(pop) >= max(10, pop_size // 2):
            break
        x = repair_reflect(list(x))
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        if f < best:
            best, best_x = f, list(x)
            last_improve_t = time.time()
        if len(pop) >= pop_size:
            break

    while len(pop) < pop_size and time.time() < deadline:
        x = repair_reflect(rand_vec())
        f = safe_eval(x)
        pop.append(x)
        fits.append(f)
        if f < best:
            best, best_x = f, list(x)
            last_improve_t = time.time()

    if best_x is None:
        best_x = repair_reflect(rand_vec())
        best = safe_eval(best_x)
        last_improve_t = time.time()

    archive = []

    # ---------- coordinate polish ----------
    def polish_best(x, fx):
        nonlocal best, best_x, last_improve_t, polish_step
        idxs = list(range(dim))
        random.shuffle(idxs)
        idxs = idxs[:polish_coords]

        improved_any = False
        for i in idxs:
            if time.time() >= deadline:
                break
            si = polish_step[i]
            if si < min_step:
                si = min_step
            base = x[i]

            x[i] = base + si
            repair_reflect(x)
            f1 = safe_eval(x)

            x[i] = base - si
            repair_reflect(x)
            f2 = safe_eval(x)

            x[i] = base
            if f1 < fx or f2 < fx:
                improved_any = True
                if f1 <= f2:
                    x[i] = base + si
                    repair_reflect(x)
                    fx = f1
                else:
                    x[i] = base - si
                    repair_reflect(x)
                    fx = f2

                polish_step[i] = min(0.35 * spans[i], polish_step[i] * 1.25)
                if fx < best:
                    best, best_x = fx, list(x)
                    last_improve_t = time.time()
            else:
                polish_step[i] = max(min_step, polish_step[i] * 0.78)

        if not improved_any:
            for i in idxs:
                polish_step[i] = max(min_step, polish_step[i] * 0.93)
        return x, fx

    # ---------- eigen update + sampling ----------
    def update_eigensystem():
        nonlocal eig_vecs, eig_vals
        if dim > 60:
            # too expensive, skip
            eig_vecs, eig_vals = None, None
            return

        k = elite_k
        elite_idx = topk_indices(fits, k)
        pts = [pop[i] for i in elite_idx]

        # mean
        m = [0.0] * dim
        invk = 1.0 / float(len(pts))
        for x in pts:
            for d in range(dim):
                m[d] += x[d]
        for d in range(dim):
            m[d] *= invk

        # covariance (symmetric)
        C = [[0.0] * dim for _ in range(dim)]
        for x in pts:
            dx = [x[d] - m[d] for d in range(dim)]
            for i in range(dim):
                di = dx[i]
                Ci = C[i]
                for j in range(i, dim):
                    Ci[j] += di * dx[j]
        for i in range(dim):
            for j in range(i, dim):
                C[i][j] *= invk
                C[j][i] = C[i][j]

        # small ridge for stability
        ridge = 1e-12
        for i in range(dim):
            C[i][i] += ridge

        V, ev = jacobi_eigen_sym(C, iters=35)
        # ensure non-negative
        for i in range(len(ev)):
            if ev[i] < 0.0:
                ev[i] = 0.0
        eig_vecs, eig_vals = V, ev

    def eigen_sampler():
        nonlocal best, best_x, last_improve_t
        if time.time() >= deadline:
            return
        if eig_vecs is None or eig_vals is None:
            return

        # time shrink: more exploit late
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        shrink = 1.0 - 0.60 * min(1.0, tfrac)
        if shrink < 0.22:
            shrink = 0.22

        bx = best_x
        # draw in eigen coordinates: z ~ N(0, diag(ev))
        # step = V * (sqrt(ev)*g)
        for _ in range(eig_samples):
            if time.time() >= deadline:
                return
            g = [random.gauss(0.0, 1.0) for _ in range(dim)]
            y = [0.0] * dim
            for i in range(dim):
                y[i] = math.sqrt(eig_vals[i] + 1e-18) * g[i]
            step = mat_vec(eig_vecs, y)
            x = [bx[d] + shrink * step[d] for d in range(dim)]
            repair_reflect(x)
            f = safe_eval(x)
            if f < best:
                best, best_x = f, list(x)
                last_improve_t = time.time()
            # inject by replacing a likely-worse individual
            j = random.randrange(pop_size)
            for __ in range(3):
                kidx = random.randrange(pop_size)
                if fits[kidx] > fits[j]:
                    j = kidx
            if f < fits[j]:
                pop[j] = x
                fits[j] = f

    # ---------- bounded Nelder–Mead (small simplex around best) ----------
    def nelder_mead_local(x0, f0, time_budget):
        nonlocal best, best_x, last_improve_t
        endt = min(deadline, time.time() + max(0.0, time_budget))
        n = dim
        if n <= 0:
            return x0, f0

        # simplex size decays with time
        tfrac = (time.time() - t0) / max(1e-12, float(max_time))
        base = 0.06 - 0.045 * min(1.0, tfrac)
        if base < 0.008:
            base = 0.008

        # build simplex: x0 + ei*step
        simp = [list(x0)]
        fs = [float(f0)]
        for i in range(n):
            if time.time() >= endt:
                break
            x = list(x0)
            x[i] += base * spans[i]
            repair_reflect(x)
            simp.append(x)
            fs.append(safe_eval(x))

        m = len(simp)
        if m < 2:
            return x0, f0

        # coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        def sort_simplex():
            nonlocal simp, fs
            idx = list(range(len(fs)))
            idx.sort(key=lambda i: fs[i])
            simp = [simp[i] for i in idx]
            fs = [fs[i] for i in idx]

        def centroid(exclude_last=True):
            k = len(simp) - 1 if exclude_last else len(simp)
            c = [0.0] * n
            inv = 1.0 / float(k)
            for i in range(k):
                xi = simp[i]
                for d in range(n):
                    c[d] += xi[d]
            for d in range(n):
                c[d] *= inv
            return c

        sort_simplex()
        while time.time() < endt:
            sort_simplex()
            if fs[0] < best:
                best, best_x = fs[0], list(simp[0])
                last_improve_t = time.time()

            x_best = simp[0]
            f_best = fs[0]
            x_worst = simp[-1]
            f_worst = fs[-1]
            x_second = simp[-2]
            f_second = fs[-2]

            c = centroid(True)

            # reflect
            xr = [c[d] + alpha * (c[d] - x_worst[d]) for d in range(n)]
            repair_reflect(xr)
            fr = safe_eval(xr)
            if time.time() >= endt:
                break

            if fr < f_best:
                # expand
                xe = [c[d] + gamma * (xr[d] - c[d]) for d in range(n)]
                repair_reflect(xe)
                fe = safe_eval(xe)
                if fe < fr:
                    simp[-1], fs[-1] = xe, fe
                else:
                    simp[-1], fs[-1] = xr, fr
            elif fr < f_second:
                simp[-1], fs[-1] = xr, fr
            else:
                # contract
                if fr < f_worst:
                    # outside contraction
                    xc = [c[d] + rho * (xr[d] - c[d]) for d in range(n)]
                else:
                    # inside contraction
                    xc = [c[d] - rho * (c[d] - x_worst[d]) for d in range(n)]
                repair_reflect(xc)
                fc = safe_eval(xc)
                if fc < f_worst:
                    simp[-1], fs[-1] = xc, fc
                else:
                    # shrink
                    for i in range(1, len(simp)):
                        if time.time() >= endt:
                            break
                        xi = simp[i]
                        xn = [x_best[d] + sigma * (xi[d] - x_best[d]) for d in range(n)]
                        repair_reflect(xn)
                        simp[i] = xn
                        fs[i] = safe_eval(xn)

            # stop if simplex is tiny in fitness spread
            fspread = fs[-1] - fs[0]
            if fspread < 1e-12:
                break

        sort_simplex()
        return simp[0], fs[0]

    # ---------- main loop (JADE + eigen exploitation) ----------
    gen = 0
    last_fullsort_gen = 0
    idx_sorted_cache = None

    while time.time() < deadline:
        gen += 1
        now = time.time()
        tfrac = (now - t0) / max(1e-12, float(max_time))

        # late-stage: do NM + quick polish and return (strong finishing)
        if tfrac > 0.82:
            # small time slice for NM
            nm_budget = (deadline - time.time()) * 0.70
            nelder_mead_local(best_x, best, nm_budget)
            polish_best(list(best_x), best)
            return best

        # occasional full sort for stable pbest selection & eigen updates
        if idx_sorted_cache is None or (gen - last_fullsort_gen) >= 6:
            idx_sorted_cache = list(range(pop_size))
            idx_sorted_cache.sort(key=lambda i: fits[i])
            last_fullsort_gen = gen

        # exploitation add-ons
        if gen % polish_every == 0:
            polish_best(list(best_x), best)

        if gen % eig_every == 0:
            update_eigensystem()
            eigen_sampler()

        # stagnation immigrants
        if time.time() - last_improve_t > stagnate_time:
            idx_sorted_cache = list(range(pop_size))
            idx_sorted_cache.sort(key=lambda i: fits[i])
            last_fullsort_gen = gen
            worst_k = max(3, pop_size // 6)
            for t in range(worst_k):
                if time.time() >= deadline:
                    return best
                wi = idx_sorted_cache[-1 - t]
                r = random.random()
                if r < 0.55:
                    x = list(best_x)
                    for d in range(dim):
                        x[d] += random.gauss(0.0, 0.15 * spans[d])
                    repair_reflect(x)
                elif r < 0.75:
                    x = opposite(best_x)
                else:
                    x = corner_vec(0.08) if random.random() < 0.5 else rand_vec()
                    repair_reflect(x)
                f = safe_eval(x)
                pop[wi] = x
                fits[wi] = f
                if f < best:
                    best, best_x = f, list(x)
                    last_improve_t = time.time()
            last_improve_t = time.time()

        # adaptive pbest rate
        p_best_rate = 0.28 - 0.20 * min(1.0, tfrac)
        if p_best_rate < 0.08:
            p_best_rate = 0.08
        p_num = max(2, int(math.ceil(p_best_rate * pop_size)))

        # sometimes use ring-neighborhood differential vectors (diversity)
        use_ring = (random.random() < 0.25)

        SF, SCR, dW = [], [], []

        # precompute pbest pool cheaply
        # use cached sorted indices
        pbest_pool = idx_sorted_cache[:p_num]

        for i in range(pop_size):
            if time.time() >= deadline:
                return best

            xi = pop[i]
            fi = fits[i]

            CR = mu_CR + 0.10 * randn_clip2()
            if CR < 0.0:
                CR = 0.0
            elif CR > 1.0:
                CR = 1.0

            F = mu_F + 0.10 * cauchy()
            tries = 0
            while F <= 0.0 and tries < 8:
                F = mu_F + 0.10 * cauchy()
                tries += 1
            if F <= 0.0:
                F = 0.1
            if F > 1.0:
                F = 1.0

            pbest_idx = pbest_pool[random.randrange(len(pbest_pool))]
            xpbest = pop[pbest_idx]

            if use_ring:
                # ring neighbors
                r1 = (i + 1 + random.randrange(pop_size - 1)) % pop_size
                r2 = (i - 1 - random.randrange(pop_size - 1)) % pop_size
                if r1 == i:
                    r1 = (i + 1) % pop_size
                if r2 == i or r2 == r1:
                    r2 = (i + pop_size - 1) % pop_size
                x1 = pop[r1]
                # x2 from archive or neighbor
                if archive and random.random() < 0.5:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    x2 = pop[r2]
            else:
                r1 = pick_excluding(pop_size, {i})
                x1 = pop[r1]
                if archive and random.random() < 0.5:
                    x2 = archive[random.randrange(len(archive))]
                else:
                    r2 = pick_excluding(pop_size, {i, r1})
                    x2 = pop[r2]

            # with some probability, do mutation in eigen coordinates (if available)
            if eig_vecs is not None and eig_vals is not None and dim <= 60 and random.random() < (0.20 + 0.35 * tfrac):
                # transform to eigen coords: z = V^T x
                zi = matT_vec(eig_vecs, xi)
                zp = matT_vec(eig_vecs, xpbest)
                z1 = matT_vec(eig_vecs, x1)
                z2 = matT_vec(eig_vecs, x2)
                zv = [zi[d] + F * (zp[d] - zi[d]) + F * (z1[d] - z2[d]) for d in range(dim)]
                v = mat_vec(eig_vecs, zv)
            else:
                v = [xi[d] + F * (xpbest[d] - xi[d]) + F * (x1[d] - x2[d]) for d in range(dim)]

            repair_reflect(v)

            jrand = random.randrange(dim)
            u = [v[d] if (d == jrand or random.random() < CR) else xi[d] for d in range(dim)]
            fu = safe_eval(u)

            if fu <= fi:
                archive.append(list(xi))
                if len(archive) > archive_max:
                    del archive[random.randrange(len(archive))]

                pop[i] = u
                fits[i] = fu

                SF.append(F)
                SCR.append(CR)
                imp = fi - fu
                if imp <= 0.0:
                    imp = 1e-12
                dW.append(imp)

                if fu < best:
                    best, best_x = fu, list(u)
                    last_improve_t = time.time()

        if SF:
            mu_CR = (1.0 - c_adapt) * mu_CR + c_adapt * (sum(SCR) / float(len(SCR)))
            num = 0.0
            den = 0.0
            for Fv, w in zip(SF, dW):
                num += w * Fv * Fv
                den += w * Fv
            if den > 0.0:
                mu_F = (1.0 - c_adapt) * mu_F + c_adapt * (num / den)

            mu_F = min(0.95, max(0.05, mu_F))
            mu_CR = min(0.98, max(0.05, mu_CR))

        # light best jitter injection
        if gen % 7 == 0 and time.time() < deadline:
            inject = 2 if dim <= 40 else 1
            scale = 0.10 - 0.07 * min(1.0, tfrac)
            if scale < 0.012:
                scale = 0.012
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                j = random.randrange(pop_size)
                for __ in range(3):
                    kidx = random.randrange(pop_size)
                    if fits[kidx] > fits[j]:
                        j = kidx
                x = list(best_x)
                for d in range(dim):
                    x[d] += random.gauss(0.0, scale * spans[d])
                repair_reflect(x)
                f = safe_eval(x)
                if f < fits[j]:
                    pop[j] = x
                    fits[j] = f
                if f < best:
                    best, best_x = f, list(x)
                    last_improve_t = time.time()

    return best
