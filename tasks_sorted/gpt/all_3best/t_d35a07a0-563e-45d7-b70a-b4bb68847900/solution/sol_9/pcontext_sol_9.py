import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded black-box minimizer (self-contained; no numpy).

    Practical upgrade over your current best (active diag-CMA + DE with reflection):
      - Switches CMA to an *unconstrained y-space* with a smooth tanh box map (better CMA geometry).
      - Keeps *active* diagonal covariance update + mirrored sampling.
      - Adds a *time-bounded surrogate-assisted* candidate filter (tiny RBF on a small elite set):
        propose many cheap candidates, evaluate only a few most promising.
      - Uses *multi-armed operator mix* (CMA sample, DE/current-to-best, best-centered TR sample),
        scheduled by remaining time, plus light late coordinate polish.
      - Robust restarts with archive reinjection.

    Returns:
      best (float): best objective value found within max_time
    """

    # -------------------- time --------------------
    start = time.time()
    deadline = start + max(0.0, float(max_time) if max_time is not None else 0.0)
    if dim <= 0:
        return float("inf")

    # -------------------- objective wrapper --------------------
    def eval_f(x):
        try:
            y = float(func(x))
        except Exception:
            return float("inf")
        if y != y or y == float("inf") or y == float("-inf"):
            return float("inf")
        return y

    # -------------------- bound transform: y -> x via tanh --------------------
    mids = [0.0] * dim
    halfspans = [1.0] * dim
    spans = [1.0] * dim
    for j in range(dim):
        lo, hi = bounds[j]
        if hi < lo:
            lo, hi = hi, lo
        mid = 0.5 * (lo + hi)
        hs = 0.5 * (hi - lo)
        if hs <= 0.0:
            hs = 1.0
        mids[j] = mid
        halfspans[j] = hs
        spans[j] = 2.0 * hs

    def to_x(y):
        x = [0.0] * dim
        for j in range(dim):
            x[j] = mids[j] + halfspans[j] * math.tanh(y[j])
        return x

    def atanh(u):
        if u <= -0.999999999999:
            u = -0.999999999999
        elif u >= 0.999999999999:
            u = 0.999999999999
        return 0.5 * math.log((1.0 + u) / (1.0 - u))

    def to_y(x):
        y = [0.0] * dim
        for j in range(dim):
            u = (x[j] - mids[j]) / halfspans[j]
            y[j] = atanh(u)
        return y

    # -------------------- RNG helpers --------------------
    def gauss():
        u1 = max(1e-16, random.random())
        u2 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def rand_x():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite_x(x):
        xo = [0.0] * dim
        for j in range(dim):
            lo, hi = bounds[j]
            xo[j] = lo + hi - x[j]
        return xo

    # scrambled Halton for seeding
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

    def pick3(n, exclude):
        while True:
            a = random.randrange(n)
            if a != exclude:
                break
        while True:
            b = random.randrange(n)
            if b != exclude and b != a:
                break
        while True:
            c = random.randrange(n)
            if c != exclude and c != a and c != b:
                break
        return a, b, c

    # -------------------- tiny RBF surrogate (small, fast) --------------------
    # Fit weights by ridge regression on kernel matrix using Gauss-Jordan.
    def rbf_fit(centers, yvals, gamma, ridge):
        m = len(centers)
        if m == 0:
            return None
        A = [[0.0] * m for _ in range(m)]
        b = [float(v) for v in yvals]
        for i in range(m):
            ci = centers[i]
            for j in range(m):
                cj = centers[j]
                d2 = 0.0
                for k in range(dim):
                    t = ci[k] - cj[k]
                    d2 += t * t
                A[i][j] = math.exp(-gamma * d2)
            A[i][i] += ridge

        # Solve A w = b
        aug = [A[i] + [b[i]] for i in range(m)]
        for col in range(m):
            piv = col
            bestv = abs(aug[col][col])
            for r in range(col + 1, m):
                v = abs(aug[r][col])
                if v > bestv:
                    bestv = v
                    piv = r
            if bestv < 1e-14:
                return None
            if piv != col:
                aug[col], aug[piv] = aug[piv], aug[col]
            invp = 1.0 / aug[col][col]
            for c in range(col, m + 1):
                aug[col][c] *= invp
            for r in range(m):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor == 0.0:
                    continue
                for c in range(col, m + 1):
                    aug[r][c] -= factor * aug[col][c]

        w = [aug[i][m] for i in range(m)]
        return w

    def rbf_predict(w, centers, x, gamma):
        if w is None:
            return float("inf")
        s = 0.0
        for i in range(len(centers)):
            ci = centers[i]
            d2 = 0.0
            for k in range(dim):
                t = x[k] - ci[k]
                d2 += t * t
            s += w[i] * math.exp(-gamma * d2)
        return s

    # -------------------- initialization / archive --------------------
    archive_cap = max(60, min(240, 14 * dim + 60))
    archive = []  # (f, x, y)

    best = float("inf")
    best_x = None
    best_y = None
    last_improve = start

    shift = [random.random() for _ in range(dim)]
    kseed = 1
    init_budget = max(18, 14 * dim)

    for i in range(init_budget):
        if time.time() >= deadline:
            return best
        if i % 4 == 0:
            x = rand_x()
        else:
            x = halton_point(kseed, shift)
            kseed += 1
        fx = eval_f(x)
        y = to_y(x)
        archive.append((fx, list(x), list(y)))
        if fx < best:
            best, best_x, best_y = fx, list(x), list(y)
            last_improve = time.time()

        if time.time() >= deadline:
            break
        if random.random() < 0.55:
            xo = opposite_x(x)
            fxo = eval_f(xo)
            yo = to_y(xo)
            archive.append((fxo, list(xo), list(yo)))
            if fxo < best:
                best, best_x, best_y = fxo, list(xo), list(yo)
                last_improve = time.time()

    if not archive:
        return eval_f(rand_x())

    archive.sort(key=lambda t: t[0])
    archive = archive[:archive_cap]
    best, best_x, best_y = archive[0][0], list(archive[0][1]), list(archive[0][2])

    # -------------------- diag-CMA-ES (diagonal) in y-space --------------------
    lam0 = max(12, min(48, 4 + int(3.0 * math.sqrt(dim)) + 2 * dim))
    lam = lam0
    ipop_mult = 2
    mu = lam // 2

    def setup_weights(mu_):
        wpos_raw = [max(0.0, math.log(mu_ + 0.5) - math.log(i + 1.0)) for i in range(mu_)]
        s = sum(wpos_raw)
        if s <= 0.0:
            wpos = [1.0 / mu_] * mu_
        else:
            wpos = [v / s for v in wpos_raw]
        mueff_ = 1.0 / sum(wi * wi for wi in wpos)
        # negative weights (active): same count, sum to -1
        wneg_raw = [-(abs(math.log(mu_ + 0.5) - math.log(i + 1.0))) for i in range(mu_)]
        sn = -sum(wneg_raw)
        if sn <= 0.0:
            wneg = [-1.0 / mu_] * mu_
        else:
            wneg = [v / sn for v in wneg_raw]
        return wpos, wneg, mueff_

    w_pos, w_neg, mueff = setup_weights(mu)

    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    mean = list(best_y)
    diagC = [1.0] * dim
    ps = [0.0] * dim
    pc = [0.0] * dim
    sigma = 0.9  # y-space

    min_diag = [1e-30] * dim
    max_diag = [1e6] * dim

    # DE parameters (x-space)
    F_base = 0.60
    CR_base = 0.85
    donor_cap = max(18, min(100, 6 * dim + 28))

    # restart logic
    stag_base = max(0.35, 0.10 * float(max_time))
    stag = stag_base
    restarts = 0

    def sigma_collapse():
        tiny = 0
        for j in range(dim):
            if sigma * math.sqrt(max(min_diag[j], diagC[j])) < 1e-12:
                tiny += 1
        return tiny >= max(1, int(0.85 * dim))

    # late polish (coordinate) in y space
    def coord_polish_y(y0, f0, step, tries):
        y = list(y0)
        f = f0
        for _ in range(tries):
            j = random.randrange(dim)
            s = step[j]
            if s <= 0.0:
                continue
            yp = list(y)
            yp[j] += s
            fp = eval_f(to_x(yp))
            if fp < f:
                return yp, fp
            ym = list(y)
            ym[j] -= s
            fm = eval_f(to_x(ym))
            if fm < f:
                return ym, fm
        return y, f

    # surrogate scaling
    avg_span = sum(spans) / float(dim)
    gamma = 1.0 / max(1e-12, (0.35 * avg_span) ** 2)
    ridge = 1e-8

    # -------------------- main loop --------------------
    while time.time() < deadline:
        now = time.time()
        tfrac = 0.0 if deadline <= start else (now - start) / (deadline - start)
        if tfrac < 0.0:
            tfrac = 0.0
        elif tfrac > 1.0:
            tfrac = 1.0

        archive.sort(key=lambda t: t[0])
        archive = archive[:archive_cap]
        donors = archive[:min(donor_cap, len(archive))]
        if not donors:
            break

        # restart
        if (now - last_improve) > stag or sigma_collapse():
            restarts += 1
            lam = min(220, lam * ipop_mult)
            mu = lam // 2
            w_pos, w_neg, mueff = setup_weights(mu)

            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs

            ps = [0.0] * dim
            pc = [0.0] * dim
            diagC = [1.0] * dim
            sigma = max(0.28, 1.15 - 0.65 * tfrac)

            if best_y is None:
                mean = [gauss() for _ in range(dim)]
            else:
                mean = [best_y[j] + gauss() * (0.40 + 0.30 * random.random()) for j in range(dim)]

            # reinject points
            inject = max(12, 3 * dim)
            for _ in range(inject):
                if time.time() >= deadline:
                    break
                if best_y is not None and random.random() < 0.75:
                    y = [best_y[j] + gauss() * 1.0 for j in range(dim)]
                    x = to_x(y)
                else:
                    x = rand_x()
                    y = to_y(x)
                fx = eval_f(x)
                archive.append((fx, list(x), list(y)))
                if fx < best:
                    best, best_x, best_y = fx, list(x), list(y)
                    last_improve = time.time()

            archive.sort(key=lambda t: t[0])
            archive = archive[:archive_cap]
            stag = stag_base * (1.0 + 0.18 * restarts)
            continue

        sqrtC = [math.sqrt(max(min_diag[j], diagC[j])) for j in range(dim)]

        # --- build a small surrogate on current elites sometimes ---
        model_w = None
        centers = None
        if len(archive) >= max(16, 3 * dim) and random.random() < (0.30 + 0.35 * tfrac):
            m = min(18, max(10, 2 * dim))
            centers = [archive[i][1] for i in range(min(m, len(archive)))]
            yvals = [archive[i][0] for i in range(len(centers))]
            model_w = rbf_fit(centers, yvals, gamma, ridge)
            if model_w is None:
                centers = None

        # We will generate a pool of candidate proposals, score by surrogate if present,
        # then evaluate only top-k.
        pool = max(lam, 18 + 4 * dim)
        pool = int(pool * (0.75 + 0.55 * tfrac))  # slightly more late
        eval_k = max(4, min(lam, 6 + dim // 3))
        eval_k = min(eval_k, pool)

        mirror_prob = 0.35 * (1.0 - tfrac) + 0.10
        use_de_prob = 0.20 * (1.0 - tfrac) + 0.06
        use_tr_prob = 0.22 + 0.22 * tfrac  # best-centered trust-region samples later

        proposals = []  # (score, kind, data)
        # kind: "CMA" -> (y,z), "DE"->x, "TR"->y

        for _ in range(pool):
            r = random.random()
            if r < use_de_prob and best_x is not None and len(donors) >= 6:
                idx = random.randrange(len(donors))
                _, xi, _ = donors[idx]
                a, b, c = pick3(len(donors), idx)
                xb = donors[b][1]
                xc = donors[c][1]

                F = F_base + 0.30 * (random.random() - 0.5)
                if F < 0.25: F = 0.25
                if F > 0.95: F = 0.95
                CR = CR_base + 0.25 * (random.random() - 0.5)
                if CR < 0.10: CR = 0.10
                if CR > 0.98: CR = 0.98

                x = [0.0] * dim
                jrand = random.randrange(dim)
                for j in range(dim):
                    if random.random() < CR or j == jrand:
                        xj = xi[j] + F * (best_x[j] - xi[j]) + F * (xb[j] - xc[j])
                    else:
                        xj = xi[j]
                    lo, hi = bounds[j]
                    if xj < lo: xj = lo
                    elif xj > hi: xj = hi
                    x[j] = xj

                if centers is not None and model_w is not None:
                    score = rbf_predict(model_w, centers, x, gamma)
                else:
                    # cheap heuristic: closer to best late
                    d2 = 0.0
                    for j in range(dim):
                        t = (x[j] - best_x[j]) / max(1e-12, spans[j])
                        d2 += t * t
                    score = d2
                proposals.append((score, "DE", x))

            elif r < (use_de_prob + use_tr_prob) and best_y is not None:
                # trust-region in y around best_y, shrinking over time
                shrink = 0.95 - 0.65 * tfrac
                y = [0.0] * dim
                for j in range(dim):
                    y[j] = best_y[j] + gauss() * (shrink * sigma * sqrtC[j])
                x = to_x(y)
                if centers is not None and model_w is not None:
                    score = rbf_predict(model_w, centers, x, gamma)
                else:
                    score = random.random()
                proposals.append((score, "TR", y))

            else:
                # CMA sample around mean (y-space)
                z = [gauss() for _ in range(dim)]
                y = [mean[j] + sigma * sqrtC[j] * z[j] for j in range(dim)]
                x = to_x(y)
                if centers is not None and model_w is not None:
                    score = rbf_predict(model_w, centers, x, gamma)
                else:
                    # prefer smaller steps late, larger early
                    zn = 0.0
                    for v in z:
                        zn += v * v
                    score = math.sqrt(zn) * (0.30 + 0.70 * (1.0 - tfrac))
                proposals.append((score, "CMA", (y, z)))

                # mirrored partner sometimes
                if random.random() < mirror_prob:
                    ym = [mean[j] - sigma * sqrtC[j] * z[j] for j in range(dim)]
                    xm = to_x(ym)
                    if centers is not None and model_w is not None:
                        scorem = rbf_predict(model_w, centers, xm, gamma)
                    else:
                        scorem = score + 1e-9
                    proposals.append((scorem, "CMA", (ym, z)))

        proposals.sort(key=lambda t: t[0])
        chosen = proposals[:eval_k]

        cand = []  # evaluated CMA candidates for tell: (f,y,x)
        # evaluate chosen
        for _, kind, data in chosen:
            if time.time() >= deadline:
                break
            if kind == "DE":
                x = data
                fx = eval_f(x)
                y = to_y(x)
            elif kind == "TR":
                y = data
                x = to_x(y)
                fx = eval_f(x)
            else:  # "CMA"
                y, _z = data
                x = to_x(y)
                fx = eval_f(x)

            archive.append((fx, list(x), list(y)))
            if fx < best:
                best, best_x, best_y = fx, list(x), list(y)
                last_improve = time.time()
            cand.append((fx, y, x))

        if not cand:
            break

        archive.sort(key=lambda t: t[0])
        archive = archive[:archive_cap]

        # If we evaluated too few CMA-ish points, add a few direct CMA evaluations for stable tell
        # (cheap safety to keep CMA update meaningful).
        need = max(0, min(mu, 6) - len(cand))
        for _ in range(need):
            if time.time() >= deadline:
                break
            z = [gauss() for _ in range(dim)]
            y = [mean[j] + sigma * sqrtC[j] * z[j] for j in range(dim)]
            x = to_x(y)
            fx = eval_f(x)
            archive.append((fx, list(x), list(y)))
            if fx < best:
                best, best_x, best_y = fx, list(x), list(y)
                last_improve = time.time()
            cand.append((fx, y, x))

        cand.sort(key=lambda t: t[0])

        # --- CMA tell update (use best top_mu from cand) ---
        old_mean = list(mean)
        top_mu = min(mu, len(cand))

        mean = [0.0] * dim
        for i in range(top_mu):
            wi = w_pos[i]
            yi = cand[i][1]
            for j in range(dim):
                mean[j] += wi * yi[j]

        inv_sigma = 1.0 / max(1e-300, sigma)

        y_step = [0.0] * dim
        for j in range(dim):
            y_step[j] = (mean[j] - old_mean[j]) * inv_sigma / max(1e-300, sqrtC[j])

        c_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        ps_norm2 = 0.0
        for j in range(dim):
            ps[j] = (1.0 - cs) * ps[j] + c_ps * y_step[j]
            ps_norm2 += ps[j] * ps[j]
        ps_norm = math.sqrt(ps_norm2)

        sigma *= math.exp((cs / damps) * ((ps_norm / max(1e-300, chi_n)) - 1.0))
        if sigma < 0.08: sigma = 0.08
        if sigma > 6.0: sigma = 6.0

        hsig = 1.0 if ps_norm < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
        c_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for j in range(dim):
            pc[j] = (1.0 - cc) * pc[j] + hsig * c_pc * y_step[j]

        z2_pos = [0.0] * dim
        for i in range(top_mu):
            wi = w_pos[i]
            yi = cand[i][1]
            for j in range(dim):
                zj = (yi[j] - old_mean[j]) * inv_sigma / max(1e-300, sqrtC[j])
                z2_pos[j] += wi * (zj * zj)

        # active negative update from worst of evaluated cand
        neg_count = top_mu
        z2_neg = [0.0] * dim
        worst_start = max(0, len(cand) - neg_count)
        count_worst = len(cand) - worst_start
        for ii in range(count_worst):
            wi = w_neg[ii]  # negative
            yi = cand[worst_start + ii][1]
            for j in range(dim):
                zj = (yi[j] - old_mean[j]) * inv_sigma / max(1e-300, sqrtC[j])
                z2_neg[j] += wi * (zj * zj)

        for j in range(dim):
            Cjj = diagC[j]
            Cjj = (1.0 - c1 - cmu) * Cjj + c1 * (pc[j] * pc[j]) + cmu * (z2_pos[j] + 0.32 * z2_neg[j])
            if Cjj < min_diag[j]:
                Cjj = min_diag[j]
            elif Cjj > max_diag[j]:
                Cjj = max_diag[j]
            diagC[j] = Cjj

        # late coordinate polish
        if tfrac > 0.72 and best_y is not None and random.random() < 0.28 and time.time() < deadline:
            step = [(0.18 - 0.10 * (tfrac - 0.72) / 0.28) * sigma * math.sqrt(max(min_diag[j], diagC[j]))
                    for j in range(dim)]
            yb, fb = coord_polish_y(best_y, best, step, tries=max(1, min(9, dim)))
            if fb < best:
                best, best_y = fb, list(yb)
                best_x = to_x(best_y)
                last_improve = time.time()
                archive.append((best, list(best_x), list(best_y)))

    return best
