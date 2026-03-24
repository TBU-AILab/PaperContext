import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimizer (self-contained, no external libs).

    Improvements vs previous variants:
      - Keeps the strong sep-CMA-ES core (fast local convergence),
      - Adds a proper "ask/tell" style evaluation cache + budget-aware operation,
      - Stronger restart logic with *two* restart types:
          (A) IPOP-like (bigger population) when clearly stuck,
          (B) small "sigma reset around best" when only step-size collapsed,
      - Adds an explicit DE/current-to-best/1/bin operator (often very effective),
      - Adds a small elite archive used for DE mutation + "mean shift" proposals,
      - Uses mirrored sampling (pairing z and -z) to reduce noise in the search,
      - More robust parameter scaling by per-dimension spans.

    Returns: best fitness found within max_time seconds.
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
            x[i] = lo[i] if fixed[i] else (lo[i] + random.random() * span[i])
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

    def clamp01(u):
        if u < 0.0:
            return 0.0
        if u > 1.0:
            return 1.0
        return u

    # Degenerate
    if adim == 0:
        x0 = [lo[i] for i in range(dim)]
        return safe_eval(x0)

    # scale reference
    avg_span = sum(span[i] for i in active) / max(1, adim)
    base = avg_span if avg_span > 0.0 else 1.0

    # cache (avoid double eval if duplicates happen)
    # simple rounding-based key; good enough and cheap
    def key_of(x):
        # scale-aware rounding: ~1e-12 relative to span
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
        if k in cache:
            return cache[k]
        v = safe_eval(x)
        cache[k] = v
        return v

    # ---------------- archive ----------------
    best = float("inf")
    best_x = rand_uniform()

    # Store evaluated points for DE and restarts: (f, x)
    ARC_MAX = max(100, min(650, 140 + 45 * adim))
    archive = []

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
        keep_best = max(40, ARC_MAX // 2)
        kept = archive[:keep_best]
        rest = archive[keep_best:]
        # keep some random for diversity
        while len(kept) < ARC_MAX and rest:
            j = random.randrange(len(rest))
            kept.append(rest.pop(j))
        archive[:] = kept

    # ---------------- seeding (LHS-like + opposites) ----------------
    seed_n = min(34 + 6 * adim, 180)
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

    # ---------------- sep-CMA-ES state ----------------
    m_full = best_x[:]
    m = pack_active(m_full)

    # diagonal covariance
    Cdiag = [1.0] * adim
    ps = [0.0] * adim
    pc = [0.0] * adim

    # population size
    lam0 = max(12, min(54, 8 + 2 * adim + int(3 * math.log(adim + 1.0))))
    lam = lam0
    sigma = 0.33 * base

    def make_weights(mu):
        w = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        s = sum(w)
        w = [wi / s for wi in w]
        mueff = 1.0 / sum(wi * wi for wi in w)
        return w, mueff

    gen = 0
    stall = 0
    last_best = best
    restarts = 0
    mini_resets = 0

    # local refinement
    ls_step = [0.05 * s for s in span]
    ls_min = [1e-12 * (s if s > 0 else 1.0) for s in span]

    # operator probabilities
    p_global = 0.025
    p_de = 0.16
    p_shift = 0.06  # mean-shift / archive-elite based proposal

    # success tracking for adapting operator rates
    succ = {"cma": 1.0, "de": 1.0, "glob": 0.4, "shift": 0.6}

    def update_rates():
        nonlocal p_global, p_de, p_shift
        ssum = succ["cma"] + succ["de"] + succ["glob"] + succ["shift"]
        if ssum <= 0:
            return
        t_de = succ["de"] / ssum
        t_gl = succ["glob"] / ssum
        t_sh = succ["shift"] / ssum

        # Smooth toward targets
        p_de = 0.88 * p_de + 0.12 * (0.06 + 0.42 * t_de)
        p_global = 0.88 * p_global + 0.12 * (0.008 + 0.10 * t_gl)
        p_shift = 0.88 * p_shift + 0.12 * (0.02 + 0.18 * t_sh)

        p_de = clamp01(p_de)
        p_global = clamp01(p_global)
        p_shift = clamp01(p_shift)

        # always leave room for CMA
        s = p_de + p_global + p_shift
        if s > 0.85:
            scale = 0.85 / s
            p_de *= scale
            p_global *= scale
            p_shift *= scale

    def sample_heavy():
        # mostly Gaussian; occasional Cauchy for rare long jumps
        if random.random() < 0.07:
            u = random.random()
            return math.tan(math.pi * (u - 0.5))  # Cauchy
        return random.gauss(0.0, 1.0)

    def elite_points(k):
        if not archive:
            return []
        archive.sort(key=lambda t: t[0])
        return [archive[i][1] for i in range(min(k, len(archive)))]

    def de_current_to_best():
        n = len(archive)
        if n < 8:
            return None
        # choose base "current" from mid-quality to keep diversity
        archive.sort(key=lambda t: t[0])
        # pick xi from top 60% (not only best)
        xi = archive[random.randrange(max(3, n // 3), n)][1]
        # choose r1,r2
        r1, r2 = random.sample(range(n), 2)
        x1 = archive[r1][1]
        x2 = archive[r2][1]

        F = 0.45 + 0.55 * random.random()
        CR = 0.55 + 0.40 * random.random()

        trial = xi[:]
        jrand = random.randrange(adim)
        for kk, i in enumerate(active):
            if kk == jrand or random.random() < CR:
                # current-to-best/1: xi + F*(best-xi) + F*(x1-x2)
                trial[i] = xi[i] + F * (best_x[i] - xi[i]) + F * (x1[i] - x2[i])
        # small jitter sometimes
        if random.random() < 0.25:
            i = random.choice(active)
            trial[i] += random.gauss(0.0, 0.005 * span[i] if span[i] > 0 else 0.005)
        reflect_to_bounds(trial)
        return trial

    def shift_candidate():
        # combine mean with an elite direction (cheap exploitation)
        elites = elite_points(6)
        if len(elites) < 2:
            return None
        xa = random.choice(elites)
        xb = random.choice(elites)
        x = m_full[:]
        a = 0.6 + 0.35 * random.random()
        b = 0.8 * (random.random() - 0.5)
        for i in active:
            x[i] = (1.0 - a) * x[i] + a * xa[i] + b * (xa[i] - xb[i])
            # small noise scaled by sigma and span
            x[i] += random.gauss(0.0, 0.15 * sigma + 0.002 * span[i])
        reflect_to_bounds(x)
        return x

    # ---------------- main loop ----------------
    while time.time() < deadline:
        gen += 1
        if gen % 7 == 0:
            update_rates()

        # CMA params
        mu = max(3, lam // 2)
        weights, mueff = make_weights(mu)

        cc = (4.0 + mueff / adim) / (adim + 4.0 + 2.0 * mueff / adim)
        cs = (mueff + 2.0) / (adim + mueff + 5.0)
        c1 = 2.0 / ((adim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((adim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (adim + 1.0)) - 1.0) + cs
        chiN = math.sqrt(adim) * (1.0 - 1.0 / (4.0 * adim) + 1.0 / (21.0 * adim * adim))

        # adapt exploration when stalled
        if stall > max(10, 2 * adim):
            p_global_eff = min(0.10, p_global + 0.04)
            p_de_eff = min(0.55, p_de + 0.12)
            p_shift_eff = min(0.18, p_shift + 0.06)
        else:
            p_global_eff, p_de_eff, p_shift_eff = p_global, p_de, p_shift

        # population: (f, x_act, src)
        pop = []

        # mirrored sampling: for half of CMA samples, also sample negative direction
        # This reduces random drift and often improves quality per eval.
        need = lam
        while need > 0:
            if time.time() >= deadline:
                return best

            r = random.random()
            if r < p_global_eff:
                xfull = rand_uniform()
                f = eval_cached(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull), "glob"))
                need -= 1
                continue

            if r < p_global_eff + p_shift_eff:
                xfull = shift_candidate()
                if xfull is None:
                    xfull = rand_uniform()
                f = eval_cached(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull), "shift"))
                need -= 1
                continue

            if r < p_global_eff + p_shift_eff + p_de_eff:
                xfull = de_current_to_best()
                if xfull is None:
                    xfull = rand_uniform()
                f = eval_cached(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull), "de"))
                need -= 1
                continue

            # CMA sample (mirrored pair if we have room)
            z = [sample_heavy() for _ in range(adim)]
            for sign in (1.0, -1.0):
                if need <= 0:
                    break
                y = [math.sqrt(max(1e-30, Cdiag[k])) * (sign * z[k]) for k in range(adim)]
                xact = [m[k] + sigma * y[k] for k in range(adim)]
                xfull = unpack_active(xact, m_full)
                reflect_to_bounds(xfull)
                f = eval_cached(xfull)
                arc_add(xfull, f)
                pop.append((f, pack_active(xfull), "cma"))
                need -= 1

        arc_trim()
        pop.sort(key=lambda t: t[0])

        # operator success credit: who produced gen-best improvement?
        if pop and pop[0][0] < last_best - 1e-15:
            succ[pop[0][2]] = succ.get(pop[0][2], 0.5) + 1.0

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
            if Ck < 1e-30:
                Ck = 1e-30
            elif Ck > 1e30:
                Ck = 1e30
            Cdiag[k] = Ck

        sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
        if sigma < 1e-16 * base:
            sigma = 1e-16 * base
        if sigma > 3.0 * base + 1e-12:
            sigma = 3.0 * base + 1e-12

        # --------- cheap local refinement when stalled ----------
        if stall > max(7, 2 * adim) and time.time() + 0.002 < deadline:
            trials = min(14 + dim, 36)
            for _ in range(trials):
                if time.time() >= deadline:
                    return best
                x = best_x[:]
                if random.random() < 0.78:
                    d = random.randrange(dim)
                    if fixed[d]:
                        continue
                    sd = max(ls_min[d], ls_step[d])
                    x[d] += sd if random.random() < 0.5 else -sd
                else:
                    # random direction in active subspace scaled by sigma
                    g = [random.gauss(0.0, 1.0) for _ in range(adim)]
                    ng = vec_norm(g)
                    if ng <= 0:
                        continue
                    sc = (0.18 + 0.72 * random.random()) * (0.35 * sigma + 0.02 * base)
                    for kk, i in enumerate(active):
                        x[i] += sc * (g[kk] / ng)
                reflect_to_bounds(x)
                fx = eval_cached(x)
                arc_add(x, fx)
                if fx < best:
                    # snap mean to improved best
                    m_full = best_x[:]
                    m = pack_active(m_full)
                    ps = [0.0] * adim
                    pc = [0.0] * adim
                    stall = 0
                    succ["cma"] += 0.3
                    break
                else:
                    for d in range(dim):
                        if not fixed[d] and ls_step[d] > ls_min[d]:
                            ls_step[d] *= 0.996

        # --------- restart / reset logic ----------
        # (1) mini-reset when sigma collapsed (too small) but not improving
        if stall > max(14, 3 * adim) and sigma < 1e-10 * base and time.time() + 0.004 < deadline:
            mini_resets += 1
            stall = 0
            # increase sigma and slightly randomize mean around best
            sigma = max(sigma, (0.25 + 0.15 * mini_resets) * base)
            for kk, i in enumerate(active):
                m_full[i] = best_x[i] + random.gauss(0.0, 0.08 * span[i] + 0.15 * sigma)
            reflect_to_bounds(m_full)
            m = pack_active(m_full)
            ps = [0.0] * adim
            pc = [0.0] * adim
            # keep some learned scaling but don't let it be extreme
            for k in range(adim):
                Cdiag[k] = min(50.0, max(0.02, Cdiag[k]))
            succ["de"] += 0.2
            succ["glob"] += 0.1

        # (2) full restart when truly stuck
        if stall > max(26, 6 * adim) and time.time() + 0.01 < deadline:
            restarts += 1
            stall = 0
            # IPOP-like growth
            lam = min(160, int(lam0 * (1.0 + 0.45 * restarts)))
            if lam < 12:
                lam = 12

            # re-center around best but with a broader sigma
            m_full = best_x[:]
            m = pack_active(m_full)
            ps = [0.0] * adim
            pc = [0.0] * adim
            Cdiag = [1.0] * adim

            sigma = max(0.55 * base, sigma)

            # inject a few random evaluations to refresh archive
            inject = min(10 + 2 * adim, 34)
            for _ in range(inject):
                if time.time() >= deadline:
                    return best
                x = rand_uniform()
                f = eval_cached(x)
                arc_add(x, f)
            arc_trim()

            # refresh local steps
            for d in range(dim):
                if not fixed[d]:
                    ls_step[d] = max(ls_step[d], 0.05 * span[d])

    return best
