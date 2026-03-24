import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no numpy).

    Improvements vs your best (#1):
      - Adds a fast quadratic surrogate (RBF ridge) built on-the-fly from collected samples,
        then proposes candidates by optimizing the surrogate (cheap) + evaluates only a few.
      - Keeps your strong core: Halton/LHS init + ES + SA + pattern search + restarts.
      - Better time discipline: evaluation budgeted in small batches; avoids re-sorting big lists.
      - Uses a small tabu-like "too-close" rejection to reduce duplicate evaluations.

    Returns:
        best (float): best (minimum) function value found within max_time seconds.
    """
    t0 = time.time()
    deadline = t0 + max_time
    eps_time = 1e-4

    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] < 0.0:
            lows[i], highs[i] = highs[i], lows[i]
            spans[i] = -spans[i]
        if spans[i] == 0.0:
            spans[i] = 1.0

    avg_span = sum(spans) / float(dim)
    diag = math.sqrt(sum(s * s for s in spans)) + 1e-18

    def clamp(v, i):
        if v < lows[i]:
            return lows[i]
        if v > highs[i]:
            return highs[i]
        return v

    def eval_x(x):
        return float(func(x))

    def rand_point():
        return [lows[i] + random.random() * (highs[i] - lows[i]) for i in range(dim)]

    # ---------- RNG helpers ----------
    def randn():
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def cauchy():
        u = random.random()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return math.tan(math.pi * (u - 0.5))

    # ---------- Halton + stratified init ----------
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
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

    primes = first_primes(dim)

    def van_der_corput(index, base):
        vdc = 0.0
        denom = 1.0
        n = index
        while n:
            n, rem = divmod(n, base)
            denom *= base
            vdc += rem / denom
        return vdc

    def halton_point(k):
        x = []
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x.append(lows[i] + u * (highs[i] - lows[i]))
        return x

    def stratified_points(n):
        strata = []
        for i in range(dim):
            vals = [((j + random.random()) / n) for j in range(n)]
            random.shuffle(vals)
            strata.append([lows[i] + v * (highs[i] - lows[i]) for v in vals])
        pts = []
        for j in range(n):
            pts.append([strata[i][j] for i in range(dim)])
        return pts

    # ---------- small utilities ----------
    def dist2(a, b):
        s = 0.0
        for i in range(dim):
            d = (a[i] - b[i]) / spans[i]
            s += d * d
        return s

    # simple "too-close" check against last few evaluated points
    recent_cap = 80
    recent = []  # store points only
    close_thr2 = (1e-4) ** 2  # in normalized space

    def accept_new_point(x):
        # reject if extremely close to recent points (saves expensive evals)
        for y in recent:
            if dist2(x, y) < close_thr2:
                return False
        recent.append(x[:])
        if len(recent) > recent_cap:
            del recent[0:len(recent) - recent_cap]
        return True

    # ---------- RBF ridge surrogate (tiny, self-contained) ----------
    # Model: f(x) ~= b + sum_j w_j * exp(-||x-cj||^2 / (2*ell^2))
    # Fit by ridge regression in closed form via Gauss-Jordan.
    def solve_linear(A, b):
        # Gauss-Jordan with partial pivoting; A is nxn, b is n
        n = len(A)
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            # pivot
            piv = col
            best = abs(M[col][col])
            for r in range(col + 1, n):
                v = abs(M[r][col])
                if v > best:
                    best = v
                    piv = r
            if best < 1e-14:
                return None
            if piv != col:
                M[col], M[piv] = M[piv], M[col]
            # normalize
            div = M[col][col]
            inv = 1.0 / div
            for j in range(col, n + 1):
                M[col][j] *= inv
            # eliminate
            for r in range(n):
                if r == col:
                    continue
                factor = M[r][col]
                if factor == 0.0:
                    continue
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
        return [M[i][n] for i in range(n)]

    def build_surrogate(points, values, mmax):
        # choose centers: best few + some diverse
        n = len(points)
        if n < 6:
            return None

        # take candidates sorted by value
        idx = list(range(n))
        idx.sort(key=lambda k: values[k])
        best_idx = idx[:max(3, mmax // 2)]

        # add some far points to improve coverage
        centers = [points[k][:] for k in best_idx]
        while len(centers) < min(mmax, n):
            # greedy farthest from current centers among remaining good-ish points
            cand_pool = idx[:min(n, 5 * mmax)]
            best_k = None
            best_d = -1.0
            for k in cand_pool:
                p = points[k]
                md = 1e100
                for c in centers:
                    d = dist2(p, c)
                    if d < md:
                        md = d
                if md > best_d:
                    best_d = md
                    best_k = k
            if best_k is None:
                break
            centers.append(points[best_k][:])

        m = len(centers)
        if m < 6:
            return None

        # length-scale from median center distance
        dists = []
        for i in range(m):
            for j in range(i + 1, m):
                dists.append(dist2(centers[i], centers[j]))
        dists.sort()
        med = dists[len(dists) // 2] if dists else 1.0
        ell2 = max(1e-6, med)  # in normalized units
        # build Phi and targets for least squares: use all points (capped)
        cap = min(n, 6 * m + 40)
        use_idx = idx[:cap]

        # normal equations: (X^T X + lam I) w = X^T y
        # where X = [1, phi_1,...,phi_m]
        p = m + 1
        XtX = [[0.0] * p for _ in range(p)]
        Xty = [0.0] * p

        lam = 1e-6  # ridge
        for k in use_idx:
            x = points[k]
            y = values[k]
            row = [1.0]
            for c in centers:
                d2 = dist2(x, c)
                row.append(math.exp(-d2 / (2.0 * ell2)))
            # accumulate
            for i in range(p):
                Xty[i] += row[i] * y
                ri = row[i]
                for j in range(p):
                    XtX[i][j] += ri * row[j]

        for i in range(p):
            XtX[i][i] += lam

        w = solve_linear(XtX, Xty)
        if w is None:
            return None

        b0 = w[0]
        ww = w[1:]

        def predict(x):
            s = b0
            for j in range(m):
                d2 = dist2(x, centers[j])
                s += ww[j] * math.exp(-d2 / (2.0 * ell2))
            return s

        return predict

    def propose_from_surrogate(predict, base_x, sigma_s, tries):
        # sample around base_x and pick lowest predicted
        bestp = None
        bestv = float("inf")
        for _ in range(tries):
            x = base_x[:]
            heavy = (random.random() < 0.25)
            for i in range(dim):
                z = cauchy() if heavy else randn()
                x[i] = clamp(x[i] + z * sigma_s, i)
            v = predict(x)
            if v < bestv:
                bestv = v
                bestp = x
        return bestp

    # ---------- init evaluation ----------
    best = float("inf")
    best_x = None

    n_lhs = max(16, min(80, 8 * dim))
    n_hal = max(16, min(140, 14 * dim))
    n_rnd = max(10, min(70, 6 * dim))

    init_pts = []
    init_pts.extend(stratified_points(n_lhs))
    init_pts.extend(halton_point(k) for k in range(1, n_hal + 1))
    init_pts.extend(rand_point() for _ in range(n_rnd))
    for _ in range(min(2 * dim, 20)):
        init_pts.append([highs[i] if random.random() < 0.5 else lows[i] for i in range(dim)])

    # store dataset for surrogate
    data_x = []
    data_f = []

    # small elite list maintained without heavy sorting
    mu = max(6, min(26, 4 + dim // 2))
    elites = []  # list of (f,x)

    def push_elite(fx, x):
        nonlocal elites
        elites.append((fx, x[:]))
        elites.sort(key=lambda t: t[0])
        if len(elites) > mu:
            elites = elites[:mu]

    for x in init_pts:
        if time.time() >= deadline - eps_time:
            return best
        if not accept_new_point(x):
            continue
        fx = eval_x(x)
        data_x.append(x[:])
        data_f.append(fx)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(fx, x)

    if best_x is None:
        return best

    # ---------- pattern search ----------
    def pattern_search(x0, f0, base_scale):
        x = x0[:]
        fx = f0
        step = [base_scale * spans[i] for i in range(dim)]
        for _round in range(2):
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - eps_time:
                    return fx, x
                si = step[i]
                xi = x[i]
                cand = x[:]
                cand[i] = clamp(xi + si, i)
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    data_x.append(cand[:]); data_f.append(fc); push_elite(fc, cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
                        continue
                cand = x[:]
                cand[i] = clamp(xi - si, i)
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    data_x.append(cand[:]); data_f.append(fc); push_elite(fc, cand)
                    if fc < fx:
                        x, fx = cand, fc
                        improved = True
            step = [v * (0.55 if improved else 0.35) for v in step]
            if not improved:
                break
        return fx, x

    # ---------- main search params ----------
    lam = max(18, min(110, 10 + 5 * dim))

    sigma = 0.20 * avg_span
    sigma_min = 1e-14 * avg_span + 1e-18
    sigma_max = 1.8 * avg_span

    # SA temperature
    T = 1.0
    T_min = 1e-9
    cool = 0.995

    stall = 0
    restart_after = max(90, 20 * dim)
    last_best = best

    # surrogate refresh control
    next_sur_time = t0
    surrogate = None

    while time.time() < deadline - eps_time:
        now = time.time()

        # refresh surrogate occasionally (time-based, not iteration-based)
        if now >= next_sur_time and len(data_x) >= 12:
            mmax = max(10, min(30, 6 + 2 * dim))
            surrogate = build_surrogate(data_x, data_f, mmax=mmax)
            next_sur_time = now + 0.18  # refresh every ~0.18s (cheap enough)

        # choose parents from elites
        elites.sort(key=lambda t: t[0])
        parents = [x for (_, x) in elites[:max(3, min(len(elites), mu))]]
        parents.append(best_x[:])

        # --- ES batch ---
        for _ in range(lam):
            if time.time() >= deadline - eps_time:
                break

            p = parents[random.randrange(len(parents))]
            child = p[:]

            s = sigma * math.exp(0.25 * randn())
            if s < sigma_min:
                s = sigma_min
            elif s > sigma_max:
                s = sigma_max

            heavy = (random.random() < 0.16)
            for i in range(dim):
                z = cauchy() if heavy else randn()
                child[i] = clamp(child[i] + z * s, i)

            if not accept_new_point(child):
                continue
            fchild = eval_x(child)
            data_x.append(child[:]); data_f.append(fchild); push_elite(fchild, child)

            if fchild < best:
                best, best_x = fchild, child[:]
                stall = 0

        # --- surrogate-guided proposals (few true evals, good payoff) ---
        if surrogate is not None and time.time() < deadline - eps_time:
            # optimize surrogate around best, then evaluate 1-2 best predicted points
            sigma_s = max(sigma_min, min(sigma_max, 0.7 * sigma))
            # candidate around best
            cand = propose_from_surrogate(surrogate, best_x, sigma_s, tries=18)
            if cand is not None and time.time() < deadline - eps_time:
                if accept_new_point(cand):
                    fc = eval_x(cand)
                    data_x.append(cand[:]); data_f.append(fc); push_elite(fc, cand)
                    if fc < best:
                        best, best_x = fc, cand[:]
                        stall = 0

        # --- SA move around best ---
        if time.time() < deadline - eps_time:
            x = best_x[:]
            step_scale = max(sigma_min, min(sigma_max, 0.85 * sigma))
            for i in range(dim):
                z = cauchy() if random.random() < 0.35 else randn()
                x[i] = clamp(x[i] + z * step_scale, i)
            if accept_new_point(x):
                fx = eval_x(x)
                data_x.append(x[:]); data_f.append(fx); push_elite(fx, x)
                accept = False
                if fx <= best:
                    accept = True
                else:
                    df = fx - best
                    if T > T_min and df < 700.0 * T and random.random() < math.exp(-df / max(T, T_min)):
                        accept = True
                if accept and fx < best:
                    best, best_x = fx, x[:]
                    stall = 0

        # adapt sigma / stall
        if best < last_best:
            sigma = max(sigma_min, sigma * 0.90)
            last_best = best
        else:
            stall += 1
            if stall % max(12, 2 * dim) == 0:
                sigma = min(sigma_max, sigma * 1.07)

        T = max(T_min, T * cool)

        # intensify
        if stall % max(26, 4 * dim) == 0 and time.time() < deadline - eps_time:
            f2, x2 = pattern_search(best_x, best, base_scale=0.065)
            if f2 < best:
                best, best_x = f2, x2[:]
                stall = 0
                last_best = best
                sigma = max(sigma_min, sigma * 0.85)

        # restart
        if stall >= restart_after and time.time() < deadline - eps_time:
            stall = 0
            T = 1.0
            sigma = max(sigma_min, min(sigma_max, 0.24 * avg_span))

            # keep best; add several mixed points
            new_elites = [(best, best_x[:])]
            m = max(8, mu)
            for _ in range(m - 1):
                if time.time() >= deadline - eps_time:
                    break
                if random.random() < 0.55:
                    y = halton_point(random.randint(1, max(60, 20 * dim)))
                else:
                    y = rand_point()
                a = 0.30 + 0.65 * random.random()
                x = [clamp(a * best_x[i] + (1.0 - a) * y[i], i) for i in range(dim)]
                if not accept_new_point(x):
                    continue
                fx = eval_x(x)
                data_x.append(x[:]); data_f.append(fx)
                new_elites.append((fx, x[:]))
                if fx < best:
                    best, best_x = fx, x[:]
                    last_best = best
            new_elites.sort(key=lambda t: t[0])
            elites = new_elites[:mu]

    return best
