import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (self-contained, no external libs).

    What is improved vs your last set:
      - Uses *two-phase* search: (1) diversified seeding, (2) aggressive exploit/explore loop.
      - Exploitation uses a robust, fast local model: online quadratic ridge regression
        (diagonal + pairwise terms optional via random feature pairs) fitted on a small
        trust-region neighborhood around current best.
      - Exploration uses a tuned CMA-like diagonal evolution path update (sep-CMA flavor)
        BUT with *best-directed* sampling and mirror sampling to reduce wasted evaluations.
      - Adds a real evaluation cache to avoid duplicates.
      - Adds a strong "coordinate + random-subspace pattern search" local improver.
      - Budget/time aware throughout.

    Returns:
        best (float): best fitness found within max_time seconds.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    fixed = [span[i] == 0.0 for i in range(dim)]
    active = [i for i in range(dim) if not fixed[i]]
    adim = len(active)

    # ---------------- helpers ----------------
    def safe_eval(x):
        try:
            v = float(func(x))
            if math.isnan(v) or math.isinf(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def reflect_to_bounds(x):
        # reflection to keep continuity near borders
        for i in range(dim):
            if fixed[i]:
                x[i] = lo[i]
                continue
            a, b = lo[i], hi[i]
            w = b - a
            if w <= 0.0:
                x[i] = a
                continue
            y = x[i] - a
            y = y % (2.0 * w)
            if y > w:
                y = 2.0 * w - y
            x[i] = a + y

    def rand_uniform():
        x = [0.0] * dim
        for i in range(dim):
            x[i] = lo[i] if fixed[i] else lo[i] + random.random() * span[i]
        return x

    def pack_active(xfull):
        return [xfull[i] for i in active]

    def unpack_active(xact, template_full):
        x = template_full[:]
        for k, i in enumerate(active):
            x[i] = xact[k]
        return x

    def vec_norm(v):
        return math.sqrt(max(0.0, sum(a*a for a in v)))

    def clamp(v, a, b):
        return a if v < a else (b if v > b else v)

    if adim == 0:
        x0 = [lo[i] for i in range(dim)]
        return safe_eval(x0)

    avg_span = sum(span[i] for i in active) / max(1, adim)
    base = avg_span if avg_span > 0.0 else 1.0

    # ---------------- evaluation cache ----------------
    # scale-aware rounding key
    def key_of(x):
        k = []
        for i in range(dim):
            if fixed[i]:
                k.append(0)
            else:
                s = span[i] if span[i] != 0.0 else 1.0
                k.append(int(round((x[i] - lo[i]) / s * 1e12)))
        return tuple(k)

    cache = {}
    def eval_cached(x):
        reflect_to_bounds(x)
        k = key_of(x)
        v = cache.get(k)
        if v is None:
            v = safe_eval(x)
            cache[k] = v
        return v

    # ---------------- archive ----------------
    best = float("inf")
    best_x = rand_uniform()
    # store (f, xfull)
    archive = []
    ARC_MAX = max(120, min(900, 180 + 55 * adim))

    def arc_add(x, f):
        nonlocal best, best_x
        archive.append((f, x[:]))
        if f < best:
            best = f
            best_x = x[:]

    def arc_trim():
        if len(archive) <= ARC_MAX:
            return
        archive.sort(key=lambda t: t[0])
        keep_best = max(50, ARC_MAX // 2)
        kept = archive[:keep_best]
        rest = archive[keep_best:]
        # keep diverse random remainder
        while len(kept) < ARC_MAX and rest:
            kept.append(rest.pop(random.randrange(len(rest))))
        archive[:] = kept

    # ---------------- seeding: LHS-like + opposites + a few corners ----------------
    seed_n = min(40 + 6 * adim, 220)

    strata = []
    for i in range(dim):
        if fixed[i]:
            strata.append([lo[i]] * seed_n)
        else:
            perm = list(range(seed_n))
            random.shuffle(perm)
            vals = []
            for j in range(seed_n):
                u = (perm[j] + random.random()) / seed_n
                vals.append(lo[i] + u * span[i])
            strata.append(vals)

    for j in range(seed_n):
        if time.time() >= deadline:
            return best
        x = [strata[i][j] for i in range(dim)]
        if j & 1:
            for i in active:
                x[i] = lo[i] + (hi[i] - x[i])
        f = eval_cached(x)
        arc_add(x, f)

    # add a few "near-corner" points (often helps constrained problems)
    extra = min(8 + 2 * adim, 40)
    for _ in range(extra):
        if time.time() >= deadline:
            return best
        x = best_x[:]
        for i in active:
            # choose near lo or hi with slight random offset
            if random.random() < 0.5:
                x[i] = lo[i] + (random.random() ** 2) * span[i] * 0.05
            else:
                x[i] = hi[i] - (random.random() ** 2) * span[i] * 0.05
        f = eval_cached(x)
        arc_add(x, f)

    arc_trim()

    # ---------------- sep-CMA-ish state (active only) ----------------
    m_full = best_x[:]
    m = pack_active(m_full)
    Cdiag = [1.0] * adim
    ps = [0.0] * adim
    pc = [0.0] * adim

    lam0 = max(14, min(70, 10 + 2 * adim + int(3 * math.log(adim + 1.0))))
    lam = lam0
    sigma = 0.28 * base

    def make_weights(mu):
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        w = [wi / s for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)
        return w, mueff

    def sample_heavy():
        # gaussian with occasional cauchy tails
        if random.random() < 0.08:
            u = random.random()
            return math.tan(math.pi * (u - 0.5))
        return random.gauss(0.0, 1.0)

    # ---------------- local search step sizes ----------------
    ls_step = [0.06 * s for s in span]
    ls_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # ---------------- local surrogate: quadratic ridge in trust region ----------------
    # We fit on normalized active coordinates around best within radius.
    # Feature set: bias + linear + diagonal quadratic + (a few) random pairwise terms.
    PAIRS = min(12, max(0, adim // 2))  # small number of cross terms
    pair_idx = []
    if adim >= 2 and PAIRS > 0:
        # fixed random pairs (stable)
        seen = set()
        tries = 0
        while len(pair_idx) < PAIRS and tries < 200:
            a = random.randrange(adim)
            b = random.randrange(adim)
            if a == b:
                tries += 1
                continue
            if a > b:
                a, b = b, a
            if (a, b) not in seen:
                seen.add((a, b))
                pair_idx.append((a, b))
            tries += 1

    def build_features(u):
        # u is normalized vector in R^adim (roughly within [-1,1])
        feat = [1.0]
        feat.extend(u)
        feat.extend([ui * ui for ui in u])
        for a, b in pair_idx:
            feat.append(u[a] * u[b])
        return feat

    def fit_surrogate(center_full, radius, max_pts):
        # pick nearest points to center, fit ridge regression by solving normal equations
        if not archive:
            return None

        # compute distances and take nearest
        cx = pack_active(center_full)
        dlist = []
        for f, x in archive:
            xa = pack_active(x)
            d2 = 0.0
            for k in range(adim):
                d = xa[k] - cx[k]
                d2 += d * d
            dlist.append((d2, f, xa))
        dlist.sort(key=lambda t: t[0])

        k = min(len(dlist), max(12 + 3 * adim, min(max_pts, 80 + 5 * adim)))
        # trust region filter (but keep at least some points)
        keep = []
        r2 = max(1e-30, radius * radius)
        for i in range(k):
            if dlist[i][0] <= 9.0 * r2 or len(keep) < max(10, 2 * adim):
                keep.append(dlist[i])

        if len(keep) < max(10, 2 * adim):
            return None

        # scale: normalize u = (x-center)/radius
        # weighted by exp(-d2/(2*r^2)) to emphasize locality
        p = 1 + adim + adim + len(pair_idx)
        A = [[0.0] * p for _ in range(p)]
        b = [0.0] * p
        ridge = 1e-6  # small ridge

        for d2, y, xa in keep:
            w = math.exp(-d2 / (2.0 * r2))
            u = [(xa[k] - cx[k]) / max(1e-30, radius) for k in range(adim)]
            phi = build_features(u)
            wy = w * y
            for i in range(p):
                bi = phi[i]
                b[i] += bi * wy
                wi = w * bi
                row = A[i]
                for j in range(p):
                    row[j] += wi * phi[j]

        for i in range(p):
            A[i][i] += ridge

        # solve A*theta=b with Gaussian elimination (small p)
        # returns theta or None
        theta = b[:]
        M = [A[i][:] for i in range(p)]
        for i in range(p):
            # pivot
            piv = i
            bestabs = abs(M[i][i])
            for r in range(i+1, p):
                ar = abs(M[r][i])
                if ar > bestabs:
                    bestabs = ar
                    piv = r
            if bestabs < 1e-18:
                return None
            if piv != i:
                M[i], M[piv] = M[piv], M[i]
                theta[i], theta[piv] = theta[piv], theta[i]
            invp = 1.0 / M[i][i]
            for j in range(i, p):
                M[i][j] *= invp
            theta[i] *= invp
            for r in range(p):
                if r == i:
                    continue
                fac = M[r][i]
                if fac == 0.0:
                    continue
                for j in range(i, p):
                    M[r][j] -= fac * M[i][j]
                theta[r] -= fac * theta[i]

        # return predictor closure
        def predict(xfull):
            xa = pack_active(xfull)
            u = [(xa[k] - cx[k]) / max(1e-30, radius) for k in range(adim)]
            phi = build_features(u)
            s = 0.0
            for i in range(p):
                s += theta[i] * phi[i]
            return s

        return predict

    # ---------------- main loop ----------------
    gen = 0
    stall = 0
    last_best = best
    restarts = 0

    trust = 0.45 * base
    trust_min = 1e-14 * base
    trust_max = 1.8 * base

    while time.time() < deadline:
        gen += 1
        arc_trim()

        # CMA params
        mu = max(3, lam // 2)
        weights, mueff = make_weights(mu)
        cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
        cs = (mueff + 2.0) / (adim + mueff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

        # fit local surrogate around best
        if stall > max(6, 2 * adim):
            trust = max(trust_min, trust * 0.985)
        else:
            trust = min(trust_max, trust * 1.01)

        predictor = None
        if len(archive) >= max(20, 3 * adim):
            predictor = fit_surrogate(best_x, trust, max_pts=120 + 6 * adim)

        # probabilities
        # more surrogate usage when stalled; always keep some true exploration
        p_global = 0.03 if stall < max(10, 2 * adim) else 0.08
        p_surr = 0.25 if predictor is not None else 0.0
        if stall > max(14, 3 * adim) and predictor is not None:
            p_surr = 0.45

        # build population
        pop = []  # (f, xact)
        need = lam

        while need > 0:
            if time.time() >= deadline:
                return best

            r = random.random()

            if r < p_global:
                xfull = rand_uniform()
                f = eval_cached(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull)))
                need -= 1
                continue

            if r < p_global + p_surr and predictor is not None:
                # sample a batch near best; select by surrogate then evaluate best few
                # (1) generate candidates
                cand = []
                n_cand = 18 + 3 * adim
                center = best_x
                for _ in range(n_cand):
                    x = center[:]
                    # random direction within trust radius
                    g = [random.gauss(0.0, 1.0) for _ in range(adim)]
                    ng = vec_norm(g)
                    if ng <= 0.0:
                        continue
                    rad = trust * (random.random() ** (1.0 / max(1, adim)))
                    for t, ii in enumerate(active):
                        x[ii] += rad * (g[t] / ng)
                    # add mild coordinate noise
                    if random.random() < 0.35:
                        ii = random.choice(active)
                        x[ii] += random.gauss(0.0, 0.25 * trust)
                    reflect_to_bounds(x)
                    pred = predictor(x)
                    cand.append((pred, x))
                if cand:
                    cand.sort(key=lambda t: t[0])
                    # evaluate the top predicted one (greedy) + occasional second
                    pick = cand[0][1]
                    f = eval_cached(pick)
                    arc_add(pick, f)
                    pop.append((f, pack_active(pick)))
                    need -= 1
                    continue
                # fallback to CMA if candidate gen failed

            # CMA-like sampling with mirroring, and slight pull toward best
            z = [sample_heavy() for _ in range(adim)]
            for sign in (1.0, -1.0):
                if need <= 0:
                    break
                y = [math.sqrt(max(1e-30, Cdiag[k])) * (sign * z[k]) for k in range(adim)]
                # best-directed component (helps on many problems)
                pull = 0.15 + 0.20 * random.random()
                xact = [m[k] + sigma * y[k] + pull * (pack_active(best_x)[k] - m[k]) for k in range(adim)]
                xfull = unpack_active(xact, m_full)
                reflect_to_bounds(xfull)
                f = eval_cached(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull)))
                need -= 1

        pop.sort(key=lambda t: t[0])

        # update stall
        if best < last_best - 1e-15:
            last_best = best
            stall = 0
        else:
            stall += 1

        # CMA recombination update
        old_m = m[:]
        new_m = [0.0] * adim
        for i in range(mu):
            wi = weights[i]
            xi = pop[i][1]
            for k in range(adim):
                new_m[k] += wi * xi[k]
        m = new_m
        m_full = unpack_active(m, m_full)
        reflect_to_bounds(m_full)
        m = pack_active(m_full)

        invsigma = 1.0 / max(1e-30, sigma)
        y_w = [(m[k] - old_m[k]) * invsigma for k in range(adim)]
        invsqrtCy = [y_w[k] / math.sqrt(max(1e-30, Cdiag[k])) for k in range(adim)]

        coeff_ps = math.sqrt(cs * (2.0 - cs) * mueff)
        for k in range(adim):
            ps[k] = (1.0 - cs) * ps[k] + coeff_ps * invsqrtCy[k]

        ps_norm = vec_norm(ps)
        left = ps_norm / math.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen))
        hsig = 1.0 if left < (1.4 + 2.0 / (adim + 1.0)) * chiN else 0.0

        coeff_pc = math.sqrt(cc * (2.0 - cc) * mueff)
        for k in range(adim):
            pc[k] = (1.0 - cc) * pc[k] + hsig * coeff_pc * y_w[k]

        factor = 1.0 - c1 - cmu
        if factor < 0.0:
            factor = 0.0

        y_mu2 = [0.0] * adim
        for i in range(mu):
            xi = pop[i][1]
            wi = weights[i]
            for k in range(adim):
                yik = (xi[k] - old_m[k]) * invsigma
                y_mu2[k] += wi * (yik * yik)

        for k in range(adim):
            Ck = Cdiag[k] * factor
            Ck += c1 * (pc[k] * pc[k])
            if hsig == 0.0:
                Ck += c1 * cc * (2.0 - cc) * Cdiag[k] * 0.2
            Ck += cmu * y_mu2[k]
            Cdiag[k] = clamp(Ck, 1e-30, 1e30)

        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        sigma = clamp(sigma, 1e-16 * base, 3.2 * base + 1e-12)

        # -------- local improvement: coordinate + random subspace pattern search --------
        if stall > max(7, 2 * adim) and time.time() + 0.002 < deadline:
            trials = min(18 + dim, 44)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                if random.random() < 0.70:
                    d = random.randrange(dim)
                    if fixed[d]:
                        continue
                    sd = max(ls_min[d], ls_step[d])
                    x[d] += sd if random.random() < 0.5 else -sd
                else:
                    # random subspace move
                    ksub = 1 if adim == 1 else min(adim, 1 + int(1 + math.sqrt(adim) * random.random()))
                    idx = random.sample(active, ksub)
                    g = [random.gauss(0.0, 1.0) for _ in range(ksub)]
                    ng = math.sqrt(sum(v*v for v in g))
                    if ng <= 0.0:
                        continue
                    sc = (0.12 + 0.70 * random.random()) * (0.25 * sigma + 0.03 * base)
                    for t, ii in enumerate(idx):
                        x[ii] += sc * (g[t] / ng)
                reflect_to_bounds(x)
                fx = eval_cached(x)
                arc_add(x, fx)
                if fx < best:
                    # snap mean to new best, reset paths mildly
                    m_full = best_x[:]
                    m = pack_active(m_full)
                    ps = [0.0] * adim
                    pc = [0.0] * adim
                    stall = 0
                    break
                else:
                    # slowly shrink coord steps
                    for d in range(dim):
                        if not fixed[d] and ls_step[d] > ls_min[d]:
                            ls_step[d] *= 0.996

        # -------- restart logic --------
        if stall > max(26, 6 * adim) and time.time() + 0.01 < deadline:
            restarts += 1
            stall = 0
            # modest IPOP-like growth
            lam = min(200, int(lam0 * (1.0 + 0.40 * restarts)))
            lam = max(lam, 14)

            # reset CMA state around best with broadened sigma/trust
            m_full = best_x[:]
            m = pack_active(m_full)
            ps = [0.0] * adim
            pc = [0.0] * adim
            Cdiag = [1.0] * adim
            sigma = max(sigma, 0.65 * base)
            trust = max(trust, 0.75 * base)

            # inject a few fresh points
            inject = min(12 + 2 * adim, 40)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                x = rand_uniform()
                f = eval_cached(x)
                arc_add(x, f)

            # refresh local step sizes
            for d in range(dim):
                if not fixed[d]:
                    ls_step[d] = max(ls_step[d], 0.05 * span[d])

    return best
